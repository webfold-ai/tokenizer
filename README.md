# tokenizer

**The native B/U accumulator n-gram counter: the tokenizer IS the fold.**

The tokenizer replaces `tokenizers::Tokenizer` (HuggingFace). It is not a BPE over a
`tokenizer.json`; it is the n-gram counter (`context → next-token count`) that the fold proves beats
a 1.71M-parameter torch GPT. One fold, O(1) per token, no Python, no oniguruma regex.

## Where it came from

Recovered **verbatim** from bedrock's git history (`src/model/tokenizer.rs` at `ecd714f^`). Bedrock's
commit `ecd714f` (2026-09-21) extracted `model/{seed.rs,tokenizer.rs}` to standalone repos —
*"verified building standalone before push"* — and **never pushed them**. The `Seed` half survived as
[`webfold-ai/seed-format`](https://github.com/webfold-ai/seed-format); this half did not, and every
workspace that referenced it was left pointing at a crate that could not resolve.

The only change from the recovered source is the table's home: `crate::sangdb::Store` — the store
face as it was named in September — is now `apeiron::map::hashtrinity::Table`, the in-tree port of
the same primitive that the rest of the stack moved to. The API is the same
(`new` / `get` / `put` / `insert_with` / `iter`), so the body is unchanged.

## The faces

- **NB/Set** = the vocabulary (which tokens exist)
- **B/Map** = the token ↔ id map
- **B/U** = the n-gram counter (the accumulator)
- **self** = the encode/decode (the fold turned on itself — byte stream ↔ token stream)

## Use

```rust
use tokenizer::Tokenizer;

let mut tok = Tokenizer::new(3);
tok.train("the quick brown fox");
let ids = tok.encode("the quick brown fox");
assert_eq!(tok.decode(&ids), "the quick brown fox");
```

## Licence

Apache-2.0. Zero dependencies beyond the table; no derive macro; no `unsafe`.
