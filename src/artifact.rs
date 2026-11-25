//! tokenizer — BPE, in-process so the engine is self-contained (no Python at serve time).
//! `tokenizer.json` is **lifted** through wirefold's JSON parser (`parse_surface`), not   CITATION-EXEMPT: HuggingFace `tokenizers` artifact, a data file, not a repo path
//! hand-scanned — the anti-parser-diff discipline. Two regimes, both read off the config:
//!
//! * **byte-level** (SmolLM `ByteLevel`, Qwen cl100k `Split`): no normalizer, every byte becomes a
//!   printable char via `bytes_to_unicode`, and the decoder is that map's inverse.
//! * **SentencePiece / Llama-2** (Bonsai): `pre_tokenizer: null` — the **normalizer** does the work
//!   (`Prepend "▁"` then `Replace " " → "▁"`), the whole span is one BPE word, and a piece with no
//!   vocab entry decomposes to `<0xNN>` **byte-fallback** tokens (so a piece can never be dropped).
//!   The decoder is the declared chain `Replace "▁"→" "` → `ByteFallback` → `Fuse` → `Strip`.
//!
//! encode = split off added tokens → normalize → pre-tokenize → merge adjacent symbols by rank
//! (lowest rank, then leftmost) until none apply → look the pieces up. decode inverts it.
//!
//! **Known gap, confirmed real (2026-08-13, `LiquidAI/LFM2.5-230M`): the `post_processor` stage
//! is not read at all.** A checkpoint whose `tokenizer.json` declares a `TemplateProcessing`
//! post-processor (BOS/EOS/pair-template insertion — LFM2's wraps every `single` encode in a
//! leading `<|startoftext|>`) will NOT get those tokens from `encode()`. `encode()` is raw BPE
//! only. Confirmed by direct byte-for-byte diff against a live `transformers.AutoTokenizer`: with
//! the fix below, every BPE-derived id matched exactly; only the template-inserted BOS was
//! missing. Adding it is the caller's job until this is implemented — see
//! `examples/lfm2_bench.rs` for the workaround (read `bos_token_id` off `config.json`, prepend it).

// Documentation backlog: this module predates the crate-wide `missing_docs` lint (Cargo.toml
// [lints.rust]). Scoped to the module rather than silenced crate-wide; document new public items.
#![allow(missing_docs)]

use apeiron::map::hashtrinity::Table;
use wirefold::codec::syntax::{Atom, SyntaxTerm};
use wirefold::codec::{parse_surface, Surface};

/// The SentencePiece space marker, U+2581 LOWER ONE EIGHTH BLOCK.
const SP_SPACE: &str = "\u{2581}";

/// The separator that joins a merge pair `(a, b)` into the single `Table` key `a ‖ SEP ‖ b`.
///
/// `Table`'s key must be one `Coord` value. A `(String, String)` tuple *is* `Coord`, but then every
/// probe has to materialize the tuple — which is the two `String` clones per probe the old
/// std-map-on-a-string-pair was paying inside the merge loop. Joining into one key lets a probe
/// write into a caller-owned buffer and allocate nothing at all (see [`Tokenizer::rank_of`]).
///
/// The encoding `(a, b) ↦ a ‖ SEP ‖ b` is **injective whenever SEP occurs in neither piece**: the
/// first SEP in the joined key then splits it back uniquely. That side condition is not assumed — it
/// is asserted for every merge pair at load ([`Tokenizer::from_file`]), so a vocabulary that violated
/// it would fail loudly rather than alias two merges. U+0000 is the right choice: in a byte-level
/// regime the pieces are `bytes_to_unicode` chars, whose alphabet starts at U+0021, and in the
/// SentencePiece regime a NUL byte is spelled `<0x00>`, five printable chars.
const RANK_SEP: char = '\u{0}';

/// The pre-tokenization split the checkpoint declares. `Gpt2` is `ByteLevel(use_regex)` (SmolLM/Llama);
/// `Cl100k` is the `Split` with the GPT-4 regex (Qwen) where a letter run absorbs one preceding
/// non-letter char; `Metaspace` is the SentencePiece regime (`pre_tokenizer: null` + a `▁` normalizer),
/// where there is no split at all and the whole span is one word. Together these cover essentially
/// every current BPE model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreTok {
    Gpt2 {
        individual_digits: bool,
    },
    /// `digit_cap` is the `\p{N}` quantifier's upper bound, READ from the JSON's own regex text,
    /// not assumed. Qwen's cl100k pattern is bare `\p{N}` (cap 1, single digit); LFM2/Llama-3's is
    /// `\p{N}{1,3}` (cap 3). Confirmed the divergence for real: on `LiquidAI/LFM2.5-230M`,
    /// hardcoding cap 1 here split "189" into "1"/"8"/"9" (ids 526,533,534) where the real
    /// tokenizer emits one id (3405) — every occurrence of a 3-digit run in held-out text, verified
    /// against a live `transformers.AutoTokenizer`, 2026-08-13.
    Cl100k {
        digit_cap: usize,
    },
    Metaspace,
}

/// A normalizer stage. Only the two SentencePiece stages are modelled; `NFC`/`NFKC` and friends parse
/// to nothing — which is exactly what the byte-level regimes already assumed, so they are unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Norm {
    Prepend(String),
    Replace { from: String, to: String },
}

/// A decoder stage, read off `decoder`. `ByteLevel` is the GPT-2 inverse (the historical path, and the
/// default when nothing parses); the rest are the SentencePiece chain.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Dec {
    ByteLevel,
    Replace {
        from: String,
        to: String,
    },
    ByteFallback,
    Fuse,
    Strip {
        content: char,
        start: usize,
        stop: usize,
    },
}

/// A loaded BPE tokenizer: vocab (token → id), merge ranks, special tokens, the byte↔unicode maps, the
/// declared normalizer/decoder chains, and the pre-tokenization regime.
pub struct Tokenizer {
    vocab: Table<String, u32>,
    id_to_tok: Vec<String>,
    /// merge ranks, keyed by the pair joined with [`RANK_SEP`] — see that constant for why the key is
    /// one string and not the `(String, String)` tuple.
    ranks: Table<String, u32>,
    specials: Vec<(String, u32)>,
    b2u: [char; 256],
    /// The inverse of `b2u`, as a **dense array indexed by the char itself**, not a map. `b2u` is a
    /// bijection onto 256 chars that all land below U+0144, so the char *is* its own address and the
    /// inverse needs no key, no fold and no probe — `Regen`'s "the address is the key" taken to the
    /// case where the address is already dense. Indices outside the alphabet hold `None`.
    u2b: Box<[Option<u8>]>,
    pre: PreTok,
    norms: Vec<Norm>,
    decs: Vec<Dec>,
    /// `model.byte_fallback` **and** a regime where the symbols are raw text — a piece with no vocab
    /// entry then becomes its `<0xNN>` bytes. Off for byte-level checkpoints; see `from_file`.
    byte_fallback: bool,
    /// ids of `<0x00>`..`<0xFF>`, when `byte_fallback` applies and the vocab declares them.
    byte_tok: [Option<u32>; 256],
    /// id of `model.unk_token` — the last resort before a piece would be dropped. Same gate.   CITATION-EXEMPT: a HuggingFace tokenizer JSON field, not a declaration
    unk_id: Option<u32>,
}

/// Read the pre-tokenizer regime from the config: no pre-tokenizer **and** a `▁` normalizer →
/// `Metaspace` (SentencePiece/Llama-2); a `Split` with an explicit `Regex` → `Cl100k`; otherwise
/// `ByteLevel(use_regex)` = GPT-2, with `individual_digits` if a `Digits` stage precedes it.
fn detect_pretok(t: &SyntaxTerm, norms: &[Norm]) -> PreTok {
    let mut split_regex_pattern: Option<String> = None;
    let mut individual_digits = false;
    let pt_absent = matches!(field(t, "pre_tokenizer"), None | Some(SyntaxTerm::Nb));
    let metaspace_norm = norms.iter().any(|n| match n {
        Norm::Prepend(p) => p == SP_SPACE,
        Norm::Replace { from, to } => from == " " && to == SP_SPACE,
    });
    if pt_absent && metaspace_norm {
        return PreTok::Metaspace;
    }
    // A `Split` stage whose pattern is a literal `" "` STRING (not a `Regex`) is a no-op once a
    // metaspace normalizer has already run: the normalizer replaces every ASCII space with "▁"
    // BEFORE the pre-tokenizer sees the text, so a literal-space Split can never match anything.
    // Confirmed real, not a hypothetical: Gemma3's tokenizer.json (`functiongemma-270m-it`) ships
    // exactly this shape (`pre_tokenizer: {type: Split, pattern: {String: " "}, behavior:
    // MergedWithPrevious}` + `normalizer: {type: Replace, pattern: {String: " "}, content: "▁"}`,
    // decoder chain `Replace ▁→" "` -> `ByteFallback` -> `Fuse`, the exact SentencePiece/Llama-2
    // chain this crate already models). Without this, `pt_absent` is false (a real stage IS
    // present), so detection fell through to `Gpt2{individual_digits}` — wrong regime entirely:
    // measured on real held-out text, 135 emitted ids where the real tokenizer emits 79.
    let literal_space_split_noop = {
        let mut only_literal_split = true;
        let mut saw_any_stage = false;
        if let Some(pt) = field(t, "pre_tokenizer") {
            let stages = match field(pt, "pretokenizers") {
                Some(SyntaxTerm::Fourth(v)) => v.clone(),
                _ => vec![pt.clone()],
            };
            for p in &stages {
                saw_any_stage = true;
                let is_literal_space_split = field(p, "type").and_then(as_str) == Some("Split")
                    && field(p, "pattern")
                        .and_then(|x| field(x, "String"))
                        .and_then(as_str)
                        == Some(" ");
                if !is_literal_space_split {
                    only_literal_split = false;
                }
            }
        }
        saw_any_stage && only_literal_split
    };
    if literal_space_split_noop && metaspace_norm {
        return PreTok::Metaspace;
    }
    if let Some(pt) = field(t, "pre_tokenizer") {
        let stages = match field(pt, "pretokenizers") {
            Some(SyntaxTerm::Fourth(v)) => v.clone(),
            _ => vec![pt.clone()],
        };
        for p in &stages {
            match field(p, "type").and_then(as_str) {
                Some("Split") => {
                    if let Some(re) = field(p, "pattern")
                        .and_then(|x| field(x, "Regex"))
                        .and_then(as_str)
                    {
                        split_regex_pattern = Some(re.to_string());
                    }
                }
                Some("Digits") => {
                    individual_digits = matches!(
                        field(p, "individual_digits"),
                        Some(SyntaxTerm::B(Atom::Bool(true)))
                    );
                }
                _ => {}
            }
        }
    }
    if let Some(pattern) = split_regex_pattern {
        PreTok::Cl100k {
            digit_cap: digit_quantifier_cap(&pattern),
        }
    } else {
        PreTok::Gpt2 { individual_digits }
    }
}

/// Reads the `\p{N}` run cap straight out of the declared regex TEXT — not a general regex parse,
/// a literal scan for the one quantifier shape these tokenizer families actually use. Qwen's cl100k
/// pattern has bare `\p{N}` (cap 1); Llama-3/LFM2's has `\p{N}{1,3}` (cap 3). Unrecognized shapes
/// (`\p{N}+`, no cap found) fall back to 1 — the historical, still-correct behavior for every
/// checkpoint this crate has been measured against before LFM2.
fn digit_quantifier_cap(pattern: &str) -> usize {
    let needle = "\\p{N}{1,";
    if let Some(start) = pattern.find(needle) {
        let rest = &pattern[start + needle.len()..];
        if let Some(end) = rest.find('}') {
            if let Ok(n) = rest[..end].parse::<usize>() {
                return n.max(1);
            }
        }
    }
    1
}

fn field<'a>(t: &'a SyntaxTerm, k: &str) -> Option<&'a SyntaxTerm> {
    match t {
        SyntaxTerm::Bu(v) => v.iter().find(|(kk, _)| kk == k).map(|(_, x)| x),
        _ => None,
    }
}
fn as_str(t: &SyntaxTerm) -> Option<&str> {
    match t {
        SyntaxTerm::B(Atom::Str(s)) => Some(s),
        _ => None,
    }
}
fn as_u32(t: &SyntaxTerm) -> Option<u32> {
    match t {
        SyntaxTerm::B(Atom::Num(s)) => s.parse().ok(),
        _ => None,
    }
}

/// Re-decode a string that carries **one Latin-1 char per UTF-8 byte** back into real scalars:
/// `Ġ` spelled as the two chars U+00C4 U+00A0 becomes the one char U+0120.
///
/// **Do not apply this to [`parse_surface`] output.** It exists for input that is genuinely
/// byte-per-char. `parse_surface` used to be such a source — its JSON string reader widened each
/// byte to a `char` — and this shim was how the loader repaired the damage. That reader now
/// accumulates bytes and decodes UTF-8 once at the end, so it already yields real scalars, and
/// running this over them is a *second* decode: on Qwen2.5's vocabulary 6264 of 151643 keys are
/// silently rewritten (`Ã©`, the `bytes_to_unicode` spelling of `é`'s two UTF-8 bytes, collapses to
/// the single char `é`) and 91 groups of distinct tokens alias onto one key. The byte-level regime
/// never *wanted* bytes-as-chars — `bytes_to_unicode`'s alphabet is 256 Unicode scalars, which is
/// exactly what a correct JSON reader hands over.
pub fn undo_latin1(s: &str) -> String {
    if s.chars().all(|c| (c as u32) < 256) {
        String::from_utf8(s.chars().map(|c| c as u8).collect()).unwrap_or_else(|_| s.to_string())
    } else {
        s.to_string()
    }
}

/// The GPT-2 reversible byte→unicode map: printable bytes map to themselves, the rest to U+0100+n,
/// so every byte is a single printable char (the `Ġ`=space you see in the vocab).
fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend((b'!' as u32)..=(b'~' as u32));
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        out[*b as usize] = char::from_u32(*c).unwrap();
    }
    out
}

/// `Replace`'s literal pattern (`{"String": " "}`) and its replacement, if both are present.
fn replace_pair(t: &SyntaxTerm) -> Option<(String, String)> {
    let from = field(t, "pattern")
        .and_then(|p| field(p, "String"))
        .and_then(as_str)?;
    let to = field(t, "content").and_then(as_str)?;
    Some((from.to_string(), to.to_string()))
}

/// Lift the `normalizer` stage tree. A `Sequence` flattens; anything we do not model (`NFC`, …)
/// contributes nothing, which is the behaviour the byte-level regimes have always had.
fn parse_norm(t: &SyntaxTerm) -> Vec<Norm> {
    match field(t, "type").and_then(as_str) {
        Some("Sequence") => match field(t, "normalizers") {
            Some(SyntaxTerm::Fourth(v)) => v.iter().flat_map(parse_norm).collect(),
            _ => Vec::new(),
        },
        Some("Prepend") => field(t, "prepend")
            .and_then(as_str)
            .map(|s| vec![Norm::Prepend(s.to_string())])
            .unwrap_or_default(),
        Some("Replace") => replace_pair(t)
            .map(|(from, to)| vec![Norm::Replace { from, to }])
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Lift the `decoder` stage tree, same discipline as [`parse_norm`].
fn parse_dec(t: &SyntaxTerm) -> Vec<Dec> {
    match field(t, "type").and_then(as_str) {
        Some("Sequence") => match field(t, "decoders") {
            Some(SyntaxTerm::Fourth(v)) => v.iter().flat_map(parse_dec).collect(),
            _ => Vec::new(),
        },
        Some("ByteLevel") => vec![Dec::ByteLevel],
        Some("ByteFallback") => vec![Dec::ByteFallback],
        Some("Fuse") => vec![Dec::Fuse],
        Some("Replace") => replace_pair(t)
            .map(|(from, to)| vec![Dec::Replace { from, to }])
            .unwrap_or_default(),
        Some("Strip") => {
            let content = field(t, "content")
                .and_then(as_str)
                .and_then(|s| s.chars().next())
                .unwrap_or(' ');
            let start = field(t, "start").and_then(as_u32).unwrap_or(0) as usize;
            let stop = field(t, "stop").and_then(as_u32).unwrap_or(0) as usize;
            vec![Dec::Strip {
                content,
                start,
                stop,
            }]
        }
        _ => Vec::new(),
    }
}

/// A minimal binary min-heap over packed `(rank << 32) | pos` keys — the BPE merge queue, so a merge
/// costs a `log n` sift instead of a full rescan. Not the standard library's priority queue: the
/// trinity rule keeps std containers out of the fold libs and `apeiron::map::hashtrinity::Table` has no
/// ordered/priority face, so the twenty lines live here. Packing means the queue never touches a
/// `String`.
///
/// LOOP-EXEMPT: a priority queue is an ORDER, not a fold. `Table` is keyed retrieval with no ordering
/// face at all, so there is nothing here to convert to; the structure is written out in full above
/// rather than imported.
struct MergeQueue(Vec<u64>);

impl MergeQueue {
    fn with_capacity(n: usize) -> MergeQueue {
        MergeQueue(Vec::with_capacity(n))
    }
    fn push(&mut self, rank: u32, pos: usize) {
        let h = &mut self.0;
        h.push(((rank as u64) << 32) | pos as u64);
        let mut i = h.len() - 1;
        while i > 0 {
            let p = (i - 1) / 2;
            if h[p] <= h[i] {
                break;
            }
            h.swap(p, i);
            i = p;
        }
    }
    /// Pop the lowest `(rank, pos)` — HuggingFace's tie-break: lowest rank first, then leftmost.
    fn pop(&mut self) -> Option<(u32, usize)> {
        let h = &mut self.0;
        let top = *h.first()?;
        let last = h.pop().expect("non-empty");
        if !h.is_empty() {
            h[0] = last;
            let n = h.len();
            let mut i = 0;
            loop {
                let (l, r) = (2 * i + 1, 2 * i + 2);
                let mut m = i;
                if l < n && h[l] < h[m] {
                    m = l;
                }
                if r < n && h[r] < h[m] {
                    m = r;
                }
                if m == i {
                    break;
                }
                h.swap(i, m);
                i = m;
            }
        }
        Some(((top >> 32) as u32, (top & 0xffff_ffff) as usize))
    }
}

impl Tokenizer {
    /// Load a HuggingFace `tokenizer.json`.   CITATION-EXEMPT: HuggingFace `tokenizers` artifact, a data file, not a repo path
    pub fn from_file(path: &str) -> Tokenizer {
        let raw = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("read {path}"));
        let t = parse_surface(Surface::Json, &raw).expect("parse tokenizer.json");
        let model = field(&t, "model").expect("tokenizer.json: model");

        // The fold on both tables is `insert_with(key, value, apeiron::map::hashtrinity::set)`: `f(new, old) = new`,
        // i.e. LAST-WINS, which is what a JSON object and a rank-ordered merge list already mean and
        // exactly what the std map's `insert` did here. Nothing is accumulated, so no other `f`
        // applies — but unlike `insert` the merge is explicit, so an accumulating face is one `f` away.
        let vsize = match field(model, "vocab") {
            Some(SyntaxTerm::Bu(v)) => v.len(),
            _ => 0,
        };
        let mut vocab: Table<String, u32> = Table::with_capacity(vsize);
        if let Some(SyntaxTerm::Bu(v)) = field(model, "vocab") {
            for (k, val) in v {
                if let Some(id) = as_u32(val) {
                    vocab.insert_with(k.clone(), id, apeiron::map::hashtrinity::set);
                }
            }
        }

        // merges: either "a b" strings (classic) or [a, b] pairs (newer). Index = rank.
        let msize = match field(model, "merges") {
            Some(SyntaxTerm::Fourth(ms)) => ms.len(),
            _ => 0,
        };
        let mut ranks: Table<String, u32> = Table::with_capacity(msize);
        if let Some(SyntaxTerm::Fourth(ms)) = field(model, "merges") {
            for (rank, m) in ms.iter().enumerate() {
                let (a, b) = match m {
                    SyntaxTerm::B(Atom::Str(s)) => {
                        let mut it = s.splitn(2, ' ');
                        (
                            it.next().unwrap_or("").to_string(),
                            it.next().unwrap_or("").to_string(),
                        )
                    }
                    SyntaxTerm::Fourth(p) if p.len() == 2 => (
                        as_str(&p[0]).unwrap_or("").to_string(),
                        as_str(&p[1]).unwrap_or("").to_string(),
                    ),
                    _ => continue,
                };
                // The side condition that makes `a ‖ SEP ‖ b` injective (see `RANK_SEP`). Checked, not
                // assumed: without it two different merges could alias onto one rank.
                assert!(
                    !a.contains(RANK_SEP) && !b.contains(RANK_SEP),
                    "{path}: merge {rank} piece contains U+0000, which the rank key uses as its separator"
                );
                let mut key = String::with_capacity(a.len() + 1 + b.len());
                key.push_str(&a);
                key.push(RANK_SEP);
                key.push_str(&b);
                ranks.insert_with(key, rank as u32, apeiron::map::hashtrinity::set);
            }
        }

        let maxid = vocab.iter().map(|(_, &id)| id).max().unwrap_or(0) as usize;
        let mut id_to_tok = vec![String::new(); maxid + 1];
        for (k, &id) in vocab.iter() {
            if (id as usize) < id_to_tok.len() {
                id_to_tok[id as usize] = k.clone();
            }
        }

        // special (added) tokens — matched literally in the raw text, decoded to their content.
        let mut specials = Vec::new();
        if let Some(SyntaxTerm::Fourth(ats)) = field(&t, "added_tokens") {
            for at in ats {
                if let (Some(id), Some(c)) = (
                    field(at, "id").and_then(as_u32),
                    field(at, "content").and_then(as_str),
                ) {
                    let c = c.to_string();
                    specials.push((c.clone(), id));
                    if (id as usize) >= id_to_tok.len() {
                        id_to_tok.resize((id as usize) + 1, String::new());
                    }
                    id_to_tok[id as usize] = c;
                }
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // longest-match first

        let norms = field(&t, "normalizer").map(parse_norm).unwrap_or_default();
        let pre = detect_pretok(&t, &norms);

        // The decoder must invert the encoder we actually run — a chain that does not match the
        // pre-tokenization regime yields self-inconsistent text, which is worse than a wrong-but-
        // reversible one. So the declared chain is used only where we also run the matching encoder
        // (SentencePiece); every byte-level checkpoint keeps `ByteLevel`, its exact historical path.
        let mut decs = if pre == PreTok::Metaspace {
            field(&t, "decoder").map(parse_dec).unwrap_or_default()
        } else {
            Vec::new()
        };
        if decs.is_empty() {
            decs = vec![Dec::ByteLevel];
        }

        // byte fallback: `<0x00>`..`<0xFF>` are ordinary vocab entries the model reserves for raw bytes.
        // It only means anything where the symbols ARE the text's bytes. In a byte-level regime they
        // live in the `b2u` alphabet instead, so decomposing one would splice that *marker's* UTF-8
        // into the stream rather than the text's — so it stays off there, exactly as it always was.
        let byte_fallback = pre == PreTok::Metaspace
            && matches!(
                field(model, "byte_fallback"),
                Some(SyntaxTerm::B(Atom::Bool(true)))
            );
        let mut byte_tok = [None; 256];
        if byte_fallback {
            for (b, slot) in byte_tok.iter_mut().enumerate() {
                *slot = vocab.get(&format!("<{:#04X}>", b as u8)).copied();
            }
        }
        let unk_id = (pre == PreTok::Metaspace)
            .then(|| {
                field(model, "unk_token")
                    .and_then(as_str)
                    .and_then(|u| vocab.get(&u.to_string()).copied())
            })
            .flatten();

        let b2u = bytes_to_unicode();
        // Invert the bijection into a dense array sized by its own image — no map, no fold, no probe.
        let span = b2u.iter().map(|&c| c as usize).max().unwrap_or(0) + 1;
        let mut u2b = vec![None; span].into_boxed_slice();
        for (b, &c) in b2u.iter().enumerate() {
            u2b[c as usize] = Some(b as u8);
        }
        Tokenizer {
            vocab,
            id_to_tok,
            ranks,
            specials,
            b2u,
            u2b,
            pre,
            norms,
            decs,
            byte_fallback,
            byte_tok,
            unk_id,
        }
    }

    /// The merge rank of the adjacent pair `(a, b)`, if the table declares one.
    ///
    /// `key` is a caller-owned scratch buffer, reused across every probe of one word, so the whole
    /// merge loop allocates **nothing**: the tuple-keyed map this replaced had to clone both `String`s
    /// on every probe — including the probes that miss, which are most of them.
    #[inline]
    fn rank_of(&self, a: &str, b: &str, key: &mut String) -> Option<u32> {
        key.clear();
        key.push_str(a);
        key.push(RANK_SEP);
        key.push_str(b);
        self.ranks.get(key).copied()
    }

    /// Apply the declared normalizer chain — for SentencePiece, `Prepend "▁"` then `Replace " " → "▁"`.
    /// Only the SentencePiece encoder calls this, so the byte-level regimes are untouched by it.
    /// `Prepend` is a no-op on empty input; HuggingFace guards it the same way.
    fn normalize(&self, text: &str) -> String {
        let mut out = text.to_string();
        for n in &self.norms {
            match n {
                Norm::Prepend(p) => {
                    if !out.is_empty() {
                        out.insert_str(0, p);
                    }
                }
                Norm::Replace { from, to } => {
                    if !from.is_empty() {
                        out = out.replace(from.as_str(), to.as_str());
                    }
                }
            }
        }
        out
    }

    /// The end-of-turn token id, if the vocabulary declares one — searched among the added
    /// (special) tokens by the conventional names, chat end-of-turn first. What
    /// `tests/end_to_end_chat.rs` gates on (`finish_reason == "stop"` needs the eos wired).   CITATION-EXEMPT: deleted by `d84bda3`, 25 crates → 12
    /// The id of a named special token, or `None` if this vocabulary has no such token. The
    /// presence of one is how a chat DIALECT is detected — `<|start_header_id|>` is in Llama 3's
    /// vocabulary and in no other family's — so this is a membership test, not a lookup that is
    /// expected to succeed.
    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.specials
            .iter()
            .find(|(c, _)| c.as_str() == name)
            .map(|(_, id)| *id)
    }

    pub fn eos_id(&self) -> Option<usize> {
        [
            "<|im_end|>",
            "<|eot_id|>",
            "<|endoftext|>",
            "</s>",
            "<|end|>",
        ]
        .iter()
        .find_map(|name| {
            self.specials
                .iter()
                .find(|(c, _)| c.as_str() == *name)
                .map(|(_, id)| *id as usize)
        })
    }

    /// Number of ids (max id + 1).
    pub fn vocab_size(&self) -> usize {
        self.id_to_tok.len()
    }

    /// Encode text → token ids. Special tokens are split out first (longest-match), the rest goes
    /// through the pre-tokenizer + BPE.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    fn encode_into(&self, text: &str, out: &mut Vec<usize>) {
        if text.is_empty() {
            return;
        }
        let mut hit: Option<(usize, usize, u32)> = None; // (byte-pos, len, id)
        for (c, id) in &self.specials {
            if let Some(p) = text.find(c.as_str()) {
                if hit.map_or(true, |(bp, _, _)| p < bp) {
                    hit = Some((p, c.len(), *id));
                }
            }
        }
        if let Some((p, len, id)) = hit {
            self.bpe_span(&text[..p], out);
            out.push(id as usize);
            self.encode_into(&text[p + len..], out);
        } else {
            self.bpe_span(text, out);
        }
    }

    fn bpe_span(&self, text: &str, out: &mut Vec<usize>) {
        if self.pre == PreTok::Metaspace {
            return self.bpe_span_metaspace(text, out);
        }
        let pieces = match self.pre {
            PreTok::Cl100k { digit_cap } => pretokenize_cl100k(text, digit_cap),
            PreTok::Gpt2 { individual_digits } => pretokenize_gpt2(text, individual_digits),
            PreTok::Metaspace => unreachable!("handled above"),
        };
        for piece in pieces {
            let mut sym = self.init_syms_bytes(&piece);
            self.merge(&mut sym);
            for s in sym {
                self.push_piece(s, out);
            }
        }
    }

    /// The initial BPE symbols of one byte-level piece: one `b2u` char per byte.
    fn init_syms_bytes(&self, piece: &str) -> Vec<String> {
        piece
            .bytes()
            .map(|b| self.b2u[b as usize].to_string())
            .collect()
    }

    /// The initial BPE symbols of one already-normalized SentencePiece span: one char each, with a
    /// char the vocab does not know decomposed to its `<0xNN>` bytes (then `<unk>`) right here.
    fn init_syms_sp(&self, norm: &str) -> Vec<String> {
        let mut sym: Vec<String> = Vec::with_capacity(norm.len());
        for ch in norm.chars() {
            // `Table` has no `contains_key`; occupancy is `get(..).is_some()`. The symbol is built
            // owned either way — the vocab-hit branch pushed it, and it is what a miss inspects.
            let s = ch.to_string();
            if self.vocab.get(&s).is_some() {
                sym.push(s);
            } else {
                self.fallback_syms(&s, &mut sym);
            }
        }
        sym
    }

    /// Benchmark hook for `examples/trinity_tok.rs`: the initial symbol lists the merge loop starts
    /// from, one per BPE word, for `text`. Shares `init_syms_*` with the real encoder, so the A/B
    /// replays the encoder's actual key stream at its actual shape. Not part of the serving surface.
    #[doc(hidden)]
    pub fn bench_words(&self, text: &str) -> Vec<Vec<String>> {
        if self.pre == PreTok::Metaspace {
            let norm = self.normalize(text);
            return if norm.is_empty() {
                Vec::new()
            } else {
                vec![self.init_syms_sp(&norm)]
            };
        }
        let pieces = match self.pre {
            PreTok::Cl100k { digit_cap } => pretokenize_cl100k(text, digit_cap),
            PreTok::Gpt2 { individual_digits } => pretokenize_gpt2(text, individual_digits),
            PreTok::Metaspace => unreachable!("handled above"),
        };
        pieces.iter().map(|p| self.init_syms_bytes(p)).collect()
    }

    /// Benchmark hook, as above: the two loaded tables as flat pair lists — `(piece, id)` for the
    /// vocab, `((a, b), rank)` for the merges — so an A/B can build a std map and a `Table` from the
    /// SAME data in one binary. Also the collision audit: `ranks` is keyed on `a ‖ RANK_SEP ‖ b`, and
    /// splitting the key back at that separator is exactly the injectivity claim, so a caller that
    /// re-derives the pairs and finds a duplicate has falsified it. Not part of the serving surface.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn bench_tables(&self) -> (Vec<(String, u32)>, Vec<((String, String), u32)>) {
        let v = self.vocab.iter().map(|(k, &id)| (k.clone(), id)).collect();
        let r = self
            .ranks
            .iter()
            .map(|(k, &rank)| {
                let (a, b) = k
                    .split_once(RANK_SEP)
                    .expect("rank key carries its separator");
                ((a.to_string(), b.to_string()), rank)
            })
            .collect();
        (v, r)
    }

    /// Whether this checkpoint's merge loop is the queued (SentencePiece, one word per span) one.
    #[doc(hidden)]
    pub fn bench_is_metaspace(&self) -> bool {
        self.pre == PreTok::Metaspace
    }

    /// SentencePiece/Llama-2: there is no pre-tokenizer, so the **whole span** is one BPE word. The
    /// normalizer has already turned every space into `▁`; the initial symbols are single chars, and a
    /// char the vocab does not know decomposes to its `<0xNN>` bytes right here — which is what makes a
    /// silently-dropped piece impossible.
    fn bpe_span_metaspace(&self, text: &str, out: &mut Vec<usize>) {
        let norm = self.normalize(text);
        if norm.is_empty() {
            return;
        }
        let mut sym = self.init_syms_sp(&norm);
        self.merge_queued(&mut sym);
        for s in sym {
            self.push_piece(s, out);
        }
    }

    /// The `<0xNN>` decomposition of a piece the vocab does not know, as symbols. All-or-nothing: if any
    /// byte lacks a token the piece falls to `<unk>` instead, exactly as HuggingFace's BPE does.
    fn fallback_syms(&self, s: &str, sym: &mut Vec<String>) {
        if self.byte_fallback && s.bytes().all(|b| self.byte_tok[b as usize].is_some()) {
            sym.extend(s.bytes().map(|b| format!("<{b:#04X}>")));
        } else if let Some(u) = self.unk_id {
            sym.push(self.id_to_tok[u as usize].clone());
        }
    }

    /// Resolve one merged symbol to an id. The vocab is the fast path; a miss decomposes to
    /// byte-fallback tokens, then to `<unk>` — so in the SentencePiece regime a piece can never be
    /// silently dropped. Both fallbacks are off in the byte-level regimes (see `from_file`), where the
    /// symbols are `b2u` chars that a GPT-2-family vocab always contains, so this is their historical
    /// "look it up, else drop", unchanged.
    ///
    /// Takes the symbol **owned**: a `Table<String, _>` probe needs a `&String`, and the merge already
    /// produced one — so passing it by value costs nothing where borrowing it would have forced a
    /// re-allocation per emitted token.
    fn push_piece(&self, s: String, out: &mut Vec<usize>) {
        if let Some(&id) = self.vocab.get(&s) {
            out.push(id as usize);
            return;
        }
        if self.byte_fallback {
            if let Some(ids) = s
                .bytes()
                .map(|b| self.byte_tok[b as usize])
                .collect::<Option<Vec<u32>>>()
            {
                out.extend(ids.into_iter().map(|i| i as usize));
                return;
            }
        }
        if let Some(u) = self.unk_id {
            out.push(u as usize);
        }
    }

    /// BPE over one word, priority-queue driven: pop the lowest-rank (then leftmost) adjacent pair,
    /// merge it, push the two pairs it creates. The symbols live in a doubly-linked list over the
    /// original `Vec`, so a merge is O(1) plus the sift. [`merge`]'s full rescan is O(n²), which is fine
    /// for a pre-tokenized piece of a few chars but not for the SentencePiece regime, where one "word"
    /// is the entire span — a 128k-token prompt.
    fn merge_queued(&self, sym: &mut Vec<String>) {
        let n = sym.len();
        if n < 2 {
            return;
        }
        let mut next: Vec<i64> = (0..n as i64)
            .map(|i| if i + 1 < n as i64 { i + 1 } else { -1 })
            .collect();
        let mut prev: Vec<i64> = (0..n as i64).map(|i| i - 1).collect();
        let mut dead = vec![false; n];
        let mut q = MergeQueue::with_capacity(n);
        let mut key = String::new();
        for i in 0..n - 1 {
            if let Some(r) = self.rank_of(&sym[i], &sym[i + 1], &mut key) {
                q.push(r, i);
            }
        }
        while let Some((rank, i)) = q.pop() {
            if dead[i] || next[i] < 0 {
                continue;
            }
            let j = next[i] as usize;
            // Re-validate: a queue entry expires when either side has since been merged into something
            // else. Comparing the *current* pair's rank against the entry's is HuggingFace's own check.
            if self.rank_of(&sym[i], &sym[j], &mut key) != Some(rank) {
                continue;
            }
            let tail = std::mem::take(&mut sym[j]);
            sym[i].push_str(&tail);
            dead[j] = true;
            let k = next[j];
            next[i] = k;
            if k >= 0 {
                prev[k as usize] = i as i64;
                if let Some(r) = self.rank_of(&sym[i], &sym[k as usize], &mut key) {
                    q.push(r, i);
                }
            }
            let p = prev[i];
            if p >= 0 {
                if let Some(r) = self.rank_of(&sym[p as usize], &sym[i], &mut key) {
                    q.push(r, p as usize);
                }
            }
        }
        // Walk the list back into a dense vector. Index 0 is never merged away (a merge only kills
        // `next[i]`, which is strictly greater than `i`), so it is always the head.
        let mut outv = Vec::with_capacity(n);
        let mut cur = 0i64;
        while cur >= 0 {
            outv.push(std::mem::take(&mut sym[cur as usize]));
            cur = next[cur as usize];
        }
        *sym = outv;
    }

    fn merge(&self, sym: &mut Vec<String>) {
        let mut key = String::new();
        while sym.len() >= 2 {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..sym.len() - 1 {
                if let Some(r) = self.rank_of(&sym[i], &sym[i + 1], &mut key) {
                    if best.map_or(true, |(_, br)| r < br) {
                        best = Some((i, r));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            sym[i] = format!("{}{}", sym[i], sym[i + 1]);
            sym.remove(i + 1);
        }
    }

    /// Decode ids → text by running the decoder chain the config declares. For a byte-level checkpoint
    /// that chain is the single `ByteLevel` stage — the historical behaviour, unchanged. For
    /// SentencePiece it is `Replace "▁"→" "` → `ByteFallback` → `Fuse` → `Strip " " 1 0`.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut toks: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.id_to_tok.get(id))
            .cloned()
            .collect();
        for d in &self.decs {
            toks = match d {
                Dec::ByteLevel => vec![self.byte_level_decode(&toks)],
                Dec::Replace { from, to } => toks
                    .iter()
                    .map(|t| t.replace(from.as_str(), to.as_str()))
                    .collect(),
                Dec::ByteFallback => byte_fallback_decode(toks),
                Dec::Fuse => vec![toks.concat()],
                Dec::Strip {
                    content,
                    start,
                    stop,
                } => toks
                    .iter()
                    .map(|t| strip_token(t, *content, *start, *stop))
                    .collect(),
            };
        }
        toks.concat()
    }

    /// The GPT-2 byte↔unicode inverse: every char that came from a byte maps back to it; special-token
    /// content (chars outside the map) passes through as its own UTF-8.
    fn byte_level_decode(&self, toks: &[String]) -> String {
        let mut bytes = Vec::new();
        for tok in toks {
            for ch in tok.chars() {
                if let Some(b) = self.u2b.get(ch as usize).copied().flatten() {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// `ByteFallback` decoder: a run of `<0xNN>` tokens is one raw byte string, re-read as UTF-8. A run
/// that is not valid UTF-8 becomes one replacement char per byte (HuggingFace's rule) — never a panic.
fn byte_fallback_decode(toks: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(toks.len());
    let mut run: Vec<u8> = Vec::new();
    let flush = |run: &mut Vec<u8>, out: &mut Vec<String>| {
        if run.is_empty() {
            return;
        }
        match String::from_utf8(std::mem::take(run)) {
            Ok(s) => out.push(s),
            Err(e) => out.extend(e.as_bytes().iter().map(|_| "\u{fffd}".to_string())),
        }
    };
    for tok in toks {
        match byte_token(&tok) {
            Some(b) => run.push(b),
            None => {
                flush(&mut run, &mut out);
                out.push(tok);
            }
        }
    }
    flush(&mut run, &mut out);
    out
}

/// The byte a `<0xNN>` token stands for, if the token is one.
fn byte_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() == 2 {
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

/// `Strip` decoder: remove at most `start` leading and `stop` trailing occurrences of `content`.
fn strip_token(tok: &str, content: char, start: usize, stop: usize) -> String {
    let chars: Vec<char> = tok.chars().collect();
    let mut lo = 0;
    while lo < start && lo < chars.len() && chars[lo] == content {
        lo += 1;
    }
    let mut hi = chars.len();
    while hi > lo && chars.len() - hi < stop && chars[hi - 1] == content {
        hi -= 1;
    }
    chars[lo..hi].iter().collect()
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_digit(c: char) -> bool {
    c.is_numeric()
}
fn is_punct(c: char) -> bool {
    !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric()
}

/// GPT-2 split (SmolLM/Llama): contractions, ` ?\p{L}+`, ` ?\p{N}+` (single if `individual_digits`),
/// ` ?[^\s\p{L}\p{N}]+`, and whitespace runs — leaving the last space to attach to the next word (the
/// `\s+(?!\S)` rule). Pieces are returned in raw-char space (byte-encoded by the caller).
fn pretokenize_gpt2(text: &str, individual_digits: bool) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];

        // contractions: 's 't 're 've 'm 'll 'd (lowercase, no leading space)
        if c == '\'' && i + 1 < n {
            let two: String = chars[i + 1..(i + 3).min(n)].iter().collect();
            let one = chars[i + 1];
            if ["re", "ve", "ll"].contains(&two.as_str()) {
                out.push(format!("'{two}"));
                i += 3;
                continue;
            } else if ['s', 't', 'm', 'd'].contains(&one) {
                out.push(format!("'{one}"));
                i += 2;
                continue;
            }
        }

        // ` ?X+`: optional single literal-space lead, then a run of one category.
        let lead_space = c == ' '
            && i + 1 < n
            && (is_letter(chars[i + 1]) || is_digit(chars[i + 1]) || is_punct(chars[i + 1]));
        let start = if lead_space { i + 1 } else { i };
        let head = chars[start];
        if is_letter(head) {
            let mut j = start;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        if is_digit(head) {
            let mut j = start + 1;
            if !individual_digits {
                while j < n && is_digit(chars[j]) {
                    j += 1;
                }
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        if is_punct(head) {
            let mut j = start;
            while j < n && is_punct(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        i = whitespace_run(&chars, i, &mut out);
    }
    out
}

/// cl100k/GPT-4-family split (Qwen, LFM2/Llama-3): contractions (case-insensitive),
/// `[^\r\n\p{L}\p{N}]?\p{L}+` (a letter run may absorb one preceding non-letter/non-digit/non-newline
/// char), `\p{N}{1,digit_cap}` (a digit run, capped — 1 for Qwen's bare `\p{N}`, 3 for LFM2/Llama-3's
/// `\p{N}{1,3}`, read off the checkpoint's own regex by [`digit_quantifier_cap`]),
/// ` ?[^\s\p{L}\p{N}]+[\r\n]*`, `\s*[\r\n]+`, and the trailing-whitespace rules.
fn pretokenize_cl100k(text: &str, digit_cap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let nl = |c: char| c == '\r' || c == '\n';
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];

        // contractions (?i): 's 't 're 've 'm 'll 'd
        if c == '\'' && i + 1 < n {
            let two: String = chars[i + 1..(i + 3).min(n)]
                .iter()
                .flat_map(|c| c.to_lowercase())
                .collect();
            let one = chars[i + 1].to_ascii_lowercase();
            if ["re", "ve", "ll"].contains(&two.as_str()) {
                out.push(chars[i..i + 3].iter().collect());
                i += 3;
                continue;
            } else if ['s', 't', 'm', 'd'].contains(&one) {
                out.push(chars[i..i + 2].iter().collect());
                i += 2;
                continue;
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+ : letters, optionally absorbing one preceding non-LN/non-alnum char.
        let absorb =
            !nl(c) && !is_letter(c) && !is_digit(c) && i + 1 < n && is_letter(chars[i + 1]);
        if is_letter(c) || absorb {
            let ls = if is_letter(c) { i } else { i + 1 };
            let mut j = ls;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        // \p{N}{1,digit_cap} : a digit run, greedy up to the cap, then the next chunk starts fresh
        // (a 7-digit run at cap 3 splits "123456"+"7" -> "123","456","7", left to right). Stops at
        // BOTH the cap and the first non-digit -- a lone "5" before a letter must not swallow it.
        if is_digit(c) {
            let cap = digit_cap.max(1);
            let mut j = i;
            while j < n && j - i < cap && is_digit(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let lead_space = c == ' ' && i + 1 < n && is_punct(chars[i + 1]);
        let pstart = if lead_space { i + 1 } else { i };
        if pstart < n && is_punct(chars[pstart]) {
            let mut j = pstart;
            while j < n && is_punct(chars[j]) {
                j += 1;
            }
            while j < n && nl(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        // `\s*[\r\n]+` : a whitespace run that contains a newline → consume through the trailing newlines.
        let mut j = i;
        while j < n && chars[j].is_whitespace() {
            j += 1;
        }
        if (i..j).any(|k| nl(chars[k])) {
            // shrink back to end just after the last newline in the run
            let mut e = j;
            while e > i && !nl(chars[e - 1]) {
                e -= 1;
            }
            out.push(chars[i..e].iter().collect());
            i = e;
            continue;
        }
        i = whitespace_run(&chars, i, &mut out);
    }
    out
}

/// Shared trailing-whitespace handling: a run of whitespace emits as one piece, except a non-final
/// space run leaves its last char to attach to the next word (`\s+(?!\S)`). Returns the new cursor.
fn whitespace_run(chars: &[char], i: usize, out: &mut Vec<String>) -> usize {
    let n = chars.len();
    let mut j = i;
    while j < n && chars[j].is_whitespace() {
        j += 1;
    }
    if j == n {
        out.push(chars[i..j].iter().collect()); // trailing run: all of it
        j
    } else if j - i >= 2 {
        out.push(chars[i..j - 1].iter().collect()); // leave the last ws for the next word's ` ?`
        j - 1
    } else {
        out.push(chars[i..j].iter().collect()); // lone tab/newline before a word
        j
    }
}

#[cfg(test)]
mod tests {
    use super::{
        byte_fallback_decode, byte_token, bytes_to_unicode, pretokenize_cl100k, pretokenize_gpt2,
        strip_token, undo_latin1, Dec, MergeQueue, PreTok, Tokenizer,
    };

    /// [`undo_latin1`] still does what it says for input that really is one Latin-1 char per byte.
    /// It is no longer on the `tokenizer.json` load path — see the second half, which is why.   CITATION-EXEMPT: HuggingFace `tokenizers` artifact, a data file, not a repo path
    #[test]
    fn undo_latin1_recovers_byte_level_keys_but_double_decodes_real_scalars() {
        // genuinely byte-per-char input: "Ġ" (U+0120 = bytes C4 A0) spelled as two Latin-1 chars.
        let expanded: String = [0xC4u8 as char, 0xA0u8 as char].iter().collect();
        assert_eq!(undo_latin1(&expanded), "Ġ");
        assert_eq!(undo_latin1("hello"), "hello"); // ASCII unchanged

        // and the trap: `Ã©` is a REAL Qwen2.5 vocab key (the `bytes_to_unicode` spelling of the two
        // UTF-8 bytes of `é`). The JSON parser hands it over correctly as two scalars; a second
        // decode collapses it to one char, and the vocab entry for `é` is then unreachable.
        assert_eq!(undo_latin1("\u{c3}\u{a9}"), "é");
        assert_ne!(undo_latin1("\u{c3}\u{a9}"), "\u{c3}\u{a9}");
    }

    /// The load path must take `tokenizer.json` strings **as parsed** — no second UTF-8 decode.   CITATION-EXEMPT: HuggingFace `tokenizers` artifact, a data file, not a repo path
    ///
    /// Checkpoint-free, so it guards on any machine (unlike the round-trip tests below, which skip
    /// themselves when the model is not on this box). The vocabulary is the smallest one that can
    /// show the bug: a key over the `bytes_to_unicode` alphabet whose chars all sit under U+0100 and
    /// whose Latin-1 bytes happen to be valid UTF-8, so a re-decode silently rewrites it.
    #[test]
    fn byte_level_vocab_keys_are_not_decoded_twice() {
        // `é` is C3 A9; bytes_to_unicode maps both to themselves, so the piece is spelled "Ã©".
        let json = r#"{
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
            "model": {"type": "BPE", "vocab": {"Ã": 0, "©": 1, "Ã©": 2}, "merges": ["Ã ©"]}
        }"#;
        let path = std::env::temp_dir().join("modelfold_bytelevel_double_decode.json");
        std::fs::write(&path, json).expect("write fixture");
        let tok = Tokenizer::from_file(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);

        // the merge fires, so the multi-char key is the one that gets looked up.
        assert_eq!(tok.encode("é"), vec![2], "the merged piece must resolve");
        assert_eq!(tok.decode(&[2]), "é");
        assert_eq!(tok.decode(&tok.encode("é")), "é");
    }

    #[test]
    fn bytes_to_unicode_is_a_bijection_with_space_as_g() {
        let m = bytes_to_unicode();
        // occupancy as the fold: `insert_with(c, (), keep-either)` — the set face of the one mutator.
        let mut uniq: apeiron::map::hashtrinity::Table<char, ()> =
            apeiron::map::hashtrinity::Table::with_capacity(256);
        for &c in m.iter() {
            uniq.insert_with(c, (), |new, _old| new);
        }
        assert_eq!(uniq.len(), 256); // every byte → a distinct char
        assert_eq!(m[b' ' as usize], 'Ġ'); // the space marker
        assert_eq!(m[b'a' as usize], 'a'); // printable ASCII maps to itself
    }

    #[test]
    fn gpt2_leaves_last_space_for_the_word() {
        // "  hi" → ["Ġ"(one space piece), " hi"] in raw space: [" ", " hi"]
        assert_eq!(
            pretokenize_gpt2("  hi", true),
            vec![" ".to_string(), " hi".to_string()]
        );
    }

    #[test]
    fn cl100k_absorbs_one_preceding_nonletter() {
        // "_case" is one piece (letter run absorbs the '_'); GPT-2 would split it.
        assert_eq!(pretokenize_cl100k("_case", 0), vec!["_case".to_string()]);
        assert_eq!(
            pretokenize_gpt2("_case", true),
            vec!["_".to_string(), "case".to_string()]
        );
    }

    #[test]
    fn merge_queue_pops_lowest_rank_then_leftmost() {
        let mut q = MergeQueue::with_capacity(4);
        q.push(5, 1);
        q.push(2, 9);
        q.push(2, 3);
        q.push(0, 7);
        assert_eq!(q.pop(), Some((0, 7)));
        assert_eq!(q.pop(), Some((2, 3))); // same rank → leftmost first
        assert_eq!(q.pop(), Some((2, 9)));
        assert_eq!(q.pop(), Some((5, 1)));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn byte_tokens_decode_back_to_raw_bytes() {
        assert_eq!(byte_token("<0x0A>"), Some(10));
        assert_eq!(byte_token("<0xFF>"), Some(255));
        assert_eq!(byte_token("<0x00>"), Some(0));
        assert_eq!(byte_token("<0xZZ>"), None);
        assert_eq!(byte_token("hello"), None);
        assert_eq!(byte_token("<0x1>"), None);
        // a run of byte tokens is ONE utf-8 string; "→" is E2 86 92.
        let toks = ["a", "<0xE2>", "<0x86>", "<0x92>", "b"]
            .map(String::from)
            .to_vec();
        assert_eq!(
            byte_fallback_decode(toks),
            vec!["a".to_string(), "\u{2192}".to_string(), "b".to_string()]
        );
        // an invalid run degrades to one replacement char per byte, never a panic.
        assert_eq!(
            byte_fallback_decode(vec!["<0xE2>".to_string()]),
            vec!["\u{fffd}".to_string()]
        );
    }

    #[test]
    fn strip_removes_at_most_the_declared_count() {
        assert_eq!(strip_token("  hi ", ' ', 1, 0), " hi ");
        assert_eq!(strip_token("  hi ", ' ', 2, 1), "hi");
        assert_eq!(strip_token("hi", ' ', 1, 1), "hi");
        assert_eq!(strip_token("   ", ' ', 1, 0), "  ");
    }

    /// The SentencePiece regime end to end, against ids taken from HuggingFace `tokenizers` 0.22.2 on
    /// the same checkpoint. Skipped when the checkpoint is not on this box.
    #[test]
    fn bonsai_sentencepiece_matches_huggingface_and_round_trips() {
        let dir = "/home/socol/models/bonsai";
        if !std::path::Path::new(&format!("{dir}/tokenizer.json")).exists() {
            return;
        }
        let tok = Tokenizer::from_file(&format!("{dir}/tokenizer.json"));
        assert_eq!(tok.pre, PreTok::Metaspace);
        assert_eq!(tok.eos_id(), Some(2));

        // pinned HF ids. "b"/"f" are the regression guard for the `"\b"`/`"\f"` vocab keys, which a
        // JSON reader that ignores those escapes silently collapses onto the letters.
        for (text, want) in [
            ("Hello world", &[22557u32, 1526][..]),
            (" Hello", &[28705, 22557][..]),
            ("b", &[287][..]),
            ("f", &[285][..]),
            ("abc def", &[18641, 801][..]),
            ("a\tb", &[264, 12, 28726][..]), // tab has no piece → <0x09> byte fallback
            // U+0001 *is* a piece, spelled with a \u escape in the json — the guard for that lift.
            ("a\u{1}b", &[264, 29534, 28726][..]),
            ("\u{1f642}", &[28705, 29340][..]),
            (
                "fn main() {\n    println!(\"hi\");\n}\n",
                &[
                    4611, 2191, 470, 371, 13, 2287, 2682, 4778, 19228, 5365, 1041, 13, 28752, 13,
                ][..],
            ),
        ] {
            let got: Vec<u32> = tok.encode(text).iter().map(|&i| i as u32).collect();
            assert_eq!(got, want, "encode {text:?}");
            assert_eq!(tok.decode(&tok.encode(text)), text, "round-trip {text:?}");
        }

        // every piece resolves — the silent drop is gone, so the id stream cannot lose text.
        for text in [
            "",
            " ",
            "  x",
            "x  ",
            "héllo 😀",
            "line1\nline2",
            "中文测试",
            "👨‍👩‍👧‍👦",
            "a\r\nb",
        ] {
            assert_eq!(tok.decode(&tok.encode(text)), text, "round-trip {text:?}");
        }
    }

    /// The byte-level regimes must be untouched: same regime, and the pieces still resolve.
    #[test]
    fn byte_level_checkpoints_keep_their_regime() {
        for (dir, want) in [
            (
                "/home/socol/models/qwen2.5-0.5b-instruct",
                PreTok::Cl100k { digit_cap: 1 },
            ),
            (
                "/home/socol/models/smollm2-135m",
                PreTok::Gpt2 {
                    individual_digits: true,
                },
            ),
        ] {
            if !std::path::Path::new(&format!("{dir}/tokenizer.json")).exists() {
                continue;
            }
            let tok = Tokenizer::from_file(&format!("{dir}/tokenizer.json"));
            assert_eq!(tok.pre, want, "{dir}");
            assert!(
                !tok.byte_fallback && tok.unk_id.is_none(),
                "{dir}: byte-level keeps the historical drop"
            );
            assert_eq!(tok.decs, vec![Dec::ByteLevel], "{dir}");
            for text in [
                "Hello world",
                " leading",
                "héllo 😀",
                "def f(x):\n    return x + 1\n",
            ] {
                assert_eq!(
                    tok.decode(&tok.encode(text)),
                    text,
                    "{dir} round-trip {text:?}"
                );
            }
        }
    }
}

impl Tokenizer {
    /// **Build a byte-level BPE tokenizer from the pieces a GGUF carries in its metadata** — the
    /// same object [`Tokenizer::from_file`] builds, assembled from arrays instead of from a
    /// `tokenizer.json` that a `.gguf` has no sibling for.
    ///
    /// A GGUF is self-contained by design: `tokenizer.ggml.tokens` is the vocab indexed BY ID,
    /// `tokenizer.ggml.merges` the rank-ordered `"a b"` pairs, and `tokenizer.ggml.token_type` marks
    /// which ids are control/user-defined rather than ordinary pieces. Serving a `.gguf` used to
    /// panic reading `<model>.gguf/tokenizer.json`, a path that cannot exist.
    ///
    /// `pre` is the `tokenizer.ggml.pre` string. It selects the SPLITTING regime, and getting it
    /// wrong is silent: the ids still decode, the text still reads, and the boundaries differ from
    /// the reference's — so the mapping is explicit and anything unrecognised falls back to GPT-2
    /// byte-level, which is what the family without a declared pre-tokenizer means.
    ///
    /// Scope: BYTE-LEVEL families only (`gpt2`-model GGUFs — Qwen, Llama-3, GPT-NeoX …). A
    /// SentencePiece GGUF (`tokenizer.ggml.model == "llama"`) carries scores and a metaspace
    /// normalizer this does not read, so it is REFUSED rather than approximated.
    pub fn from_gguf_parts(
        model: &str,
        pre: &str,
        tokens: Vec<String>,
        merges: Vec<String>,
        token_type: &[i32],
    ) -> Tokenizer {
        assert!(
            model == "gpt2",
            "from_gguf_parts: tokenizer.ggml.model is {model:?}, not \"gpt2\" — only byte-level BPE \
             is built here; a SentencePiece GGUF carries scores and a metaspace normalizer this \
             does not read, and approximating it would tokenize plausibly and differently"
        );
        let mut vocab: Table<String, u32> = Table::with_capacity(tokens.len());
        for (id, t) in tokens.iter().enumerate() {
            vocab.insert_with(t.clone(), id as u32, apeiron::map::hashtrinity::set);
        }

        let mut ranks: Table<String, u32> = Table::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let mut it = m.splitn(2, ' ');
            let (a, b) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
            assert!(
                !a.contains(RANK_SEP) && !b.contains(RANK_SEP),
                "merge {rank} piece contains U+0000, which the rank key uses as its separator"
            );
            let mut key = String::with_capacity(a.len() + 1 + b.len());
            key.push_str(a);
            key.push(RANK_SEP);
            key.push_str(b);
            ranks.insert_with(key, rank as u32, apeiron::map::hashtrinity::set);
        }

        // CONTROL (3) and USER_DEFINED (4) are ggml's marks for pieces matched literally in the raw
        // text rather than reached through BPE — `<|im_start|>` and friends. Longest-match first,
        // as `from_file` sorts its `added_tokens`.
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| matches!(token_type.get(*i), Some(3) | Some(4)))
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // The families that declare a regex split are cl100k-shaped; everything else is plain GPT-2
        // byte-level. Named rather than guessed — a wrong choice here changes token BOUNDARIES while
        // leaving every id decodable.
        let pre = match pre {
            // `qwen35` is its OWN family in llama.cpp (LLAMA_VOCAB_PRE_TYPE_QWEN35), and it differs
            // from `qwen2` in exactly one way: `\p{M}` joins the letter class, so combining marks
            // group with the letters they modify instead of splitting off. On ASCII the two regexes
            // are identical, which is why code and English tokenize the same; text carrying
            // combining marks (Devanagari, Arabic diacritics, decomposed accents) does not, and
            // this arm is an approximation there rather than a match.
            // `digit_cap` is READ from the family's own regex where the regex is known by name:
            // Qwen's cl100k pattern is bare `\p{N}` (cap 1) — see the `PreTok::Cl100k` doc for the
            // 2026-08-13 byte-diff against a live tokenizer — while Llama-3's is `\p{N}{1,3}`
            // (cap 3). GGUF names only the family, so the cap rides the name, not a guess.
            "llama3" | "llama-bpe" => PreTok::Cl100k { digit_cap: 3 },
            "qwen2" | "qwen35" | "deepseek-llm" | "deepseek-coder" | "cl100k" | "gpt-4o"
            | "smaug-bpe" => PreTok::Cl100k { digit_cap: 1 },
            // `gpt2` is the name of the ACTUAL GPT-2 pre-tokenizer, not a synonym for "byte-level".
            "gpt2" | "olmoe" | "dbrx" | "jais" => PreTok::Gpt2 {
                individual_digits: false,
            },
            // ── AND ANY OTHER NAME IS A REFUSAL ──────────────────────────────────────────────────
            // This arm used to be `_ => Gpt2`, and the comment above it already knew what that
            // costs: "a wrong choice here changes token BOUNDARIES while leaving every id
            // decodable." That is the worst shape a bug can have — the text round-trips perfectly,
            // so nothing downstream can tell, and only the MODEL notices that it is being fed a
            // segmentation it never saw in training.
            //
            // MEASURED, 2026-08-17: Qwen3.5/3.8 declare `qwen35`, which was in no arm, so both
            // checkpoints silently got GPT-2 boundaries. 22 774 tokens against llama.cpp's 21 623 on
            // the same 64 KB — 5.3 % more, diverging by the fifth token — and the resulting
            // perplexity was measuring the tokenizer, not the quantiser it was written to grade.
            other => panic!(
                "tokenizer.ggml.pre = {other:?} is not a pre-tokenizer this build knows. Falling \
back to GPT-2 boundaries would tokenize DECODABLY and DIFFERENTLY, which no downstream check can \
see. Add the family (llama.cpp's `llama-vocab.cpp` names it) rather than guessing."
            ),
        };

        let b2u = bytes_to_unicode();
        let span = b2u.iter().map(|&c| c as usize).max().unwrap_or(0) + 1;
        let mut u2b = vec![None; span].into_boxed_slice();
        for (b, &c) in b2u.iter().enumerate() {
            u2b[c as usize] = Some(b as u8);
        }
        Tokenizer {
            vocab,
            id_to_tok: tokens,
            ranks,
            specials,
            b2u,
            u2b,
            pre,
            norms: Vec::new(),
            decs: vec![Dec::ByteLevel],
            byte_fallback: false,
            byte_tok: [None; 256],
            unk_id: None,
        }
    }
}
