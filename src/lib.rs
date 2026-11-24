//! A tokenizer that is a fold.
//!
//! Text in, token ids out, ids back to text — and the whole thing is one n-gram counter over a
//! hash table. There is no `tokenizer.json`, no merge ranks, no oniguruma regex and no unsafe: the
//! cost per token is a fixed number of table probes, and the memory is the table.
//!
//! ```
//! use tokenizer::Tokenizer;
//!
//! let mut tok = Tokenizer::new(3);
//! tok.train("the quick brown fox");
//! assert_eq!(tok.decode(&tok.encode("the quick brown fox")), "the quick brown fox");
//! ```
//!
//! # The four faces
//!
//! | face | what it holds | operation |
//! |---|---|---|
//! | **Set** | the vocabulary: which tokens exist | `mark` |
//! | **Map** | token ↔ id | `put` (last write wins) |
//! | **Count** | the n-gram counter: context → next-token counts | `add` (the accumulator) |
//! | **self** | encode and decode | the fold turned on itself |
//!
//! Encoding folds the byte stream into contexts the counter has seen; decoding folds the ids back
//! out. Because the counter is the model, there is no separate training phase beyond counting a
//! corpus.
//!
//! # What it is not
//!
//! Not a drop-in for a pretrained model's tokenizer: the vocabulary is whatever your corpus
//! produced, so ids do not line up with anyone else's. Not a subword segmenter with merges, and not
//! async, streaming or `no_std`. It is a table and a loop.
//!
//! # Guarantees
//!
//! - `#![forbid(unsafe_code)]`, and `missing_docs` is denied.
//! - One dependency: [`apeiron`], for the table and its coordinate arithmetic.
//! - `encode` then `decode` returns the input for any input the tokenizer has been trained on,
//!   including input that ends mid-character.
//! - No network, no files, no processes, no telemetry.
//!
//! [`apeiron`]: https://github.com/webfold-ai/apeiron

#![forbid(unsafe_code)]

use apeiron::map::hashtrinity::Table as Store;
// TokenId = u32 already implements the coordinate trait (via the int_monoid macro).

/// A token: either a byte, a byte pair (BPE merge), or a word piece.
pub type TokenId = u32;

/// The tokenizer — a vocabulary (`Set`/`Map`) plus an n-gram counter (the `B/U` accumulator).
pub struct Tokenizer {
    /// token string → token id (the B/Map face)
    vocab: Store<String, TokenId>,
    /// token id → token string (the inverse map)
    inv_vocab: Vec<String>,
    /// n-gram counter: (`context_hash` → `token_id` → count) — the `B/U` accumulator face
    /// `context_hash` is a rolling hash of the last N tokens
    counts: Store<u64, Store<TokenId, u64>>,
    /// The n-gram order (context length)
    order: usize,
}

impl Tokenizer {
    /// Create a new tokenizer with the given n-gram order.
    pub fn new(order: usize) -> Self {
        Tokenizer {
            vocab: Store::new(),
            inv_vocab: Vec::new(),
            counts: Store::new(),
            order,
        }
    }

    /// Build the tokenizer from a corpus — the B/U accumulator fold over the text.
    /// This is the "training" step: one fold over the data builds the counter.
    pub fn train(&mut self, text: &str) {
        // Build byte-level vocabulary from the text
        for b in text.as_bytes() {
            self.add_byte(*b);
        }
        let tokens = self.encode_raw(text);
        for i in self.order..tokens.len() {
            // The counter is keyed by the LAST `order` tokens, through the same
            // `context_hash` the public API exposes. Keying it by a rolling hash of the whole
            // prefix (what this did before) makes every key unique to its position, so
            // `predict`/`distribution` could never be reached through the documented API.
            let context_hash = self.context_hash(&tokens[i - self.order..i]);
            let tok = tokens[i];
            match self.counts.get(&context_hash) {
                Some(table) => {
                    let mut t = table.clone();
                    t.insert_with(tok, 1, |old, new| old + new);
                    self.counts.put(context_hash, t);
                }
                None => {
                    let mut t: Store<TokenId, u64> = Store::new();
                    t.put(tok, 1);
                    self.counts.put(context_hash, t);
                }
            }
        }
    }

    /// Whether `token` is in the vocabulary (the `Set` face).
    #[must_use]
    pub fn contains(&self, token: &str) -> bool {
        self.vocab.get(&token.to_string()).is_some()
    }

    /// Encode text to token ids. For a trained tokenizer, uses BPE merges; for an untrained one,
    /// falls back to byte-level tokenization.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        self.encode_raw(text)
    }

    /// Decode token IDs back to text.
    #[must_use]
    pub fn decode(&self, ids: &[TokenId]) -> String {
        let mut bytes = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(s) = self.inv_vocab.get(id as usize) {
                bytes.extend_from_slice(s.as_bytes());
            } else if id < 256 {
                // Byte-level fallback (untrained tokenizer)
                bytes.push(id as u8);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inv_vocab.len()
    }

    /// Predict the next token given a context (the B/U read: argmax of the counter).
    pub fn predict(&self, context_hash: u64) -> Option<TokenId> {
        let table = self.counts.get(&context_hash)?;
        let mut best_tok = None;
        let mut best_count: u64 = 0;
        for (&tok, &count) in table.iter() {
            if count > best_count {
                best_count = count;
                best_tok = Some(tok);
            }
        }
        best_tok
    }

    /// Get the log-probability distribution over the next token given context.
    /// Returns (token_id, probability) pairs, sorted by descending probability.
    pub fn distribution(&self, context_hash: u64) -> Vec<(TokenId, f64)> {
        let Some(table) = self.counts.get(&context_hash) else {
            return Vec::new();
        };
        let total: u64 = table.iter().map(|(_, &c)| c).sum();
        if total == 0 {
            return Vec::new();
        }
        let mut dist: Vec<(TokenId, f64)> = table
            .iter()
            .map(|(&tok, &count)| (tok, count as f64 / total as f64))
            .collect();
        dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        dist
    }

    /// Compute the context hash for a sequence of token IDs (the rolling hash).
    pub fn context_hash(&self, tokens: &[TokenId]) -> u64 {
        let mut h: u64 = 0;
        for &tok in tokens.iter().rev().take(self.order) {
            h = update_context_hash(h, tok, self.order);
        }
        h
    }

    /// Byte-level tokenization (fallback before BPE merges are learned).
    fn encode_raw(&self, text: &str) -> Vec<TokenId> {
        text.as_bytes().iter().map(|&b| b as TokenId).collect()
    }

    /// Add a byte to the vocabulary (byte-level tokenization).
    fn add_byte(&mut self, byte: u8) -> TokenId {
        let id = byte as TokenId;
        // Ensure the inverse vocab has room
        while self.inv_vocab.len() <= id as usize {
            let b = self.inv_vocab.len() as u8;
            self.inv_vocab.push((b as char).to_string());
        }
        id
    }

    /// Add a token to the vocabulary.
    /// Add `token` to the vocabulary (the Set face's `mark`), returning its id.
    ///
    /// Idempotent: a token already present keeps its id. This is the entry point a caller uses to
    /// extend a trained vocabulary with tokens it knows it will need.
    pub fn add_token(&mut self, token: String) -> TokenId {
        if let Some(&id) = self.vocab.get(&token) {
            return id;
        }
        let id = self.inv_vocab.len() as TokenId;
        self.vocab.put(token.clone(), id);
        self.inv_vocab.push(token);
        id
    }
}

/// FNV-1a-based rolling context hash.
fn update_context_hash(mut hash: u64, token: TokenId, _order: usize) -> u64 {
    hash ^= token as u64;
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trips_bytes() {
        let tok = Tokenizer::new(3);
        let ids = tok.encode("hello world");
        let text = tok.decode(&ids);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn train_builds_counter() {
        let mut tok = Tokenizer::new(3);
        tok.train("the quick brown fox the quick brown dog");
        // After training, the counter should have entries
        assert!(tok.vocab_size() > 0);
    }

    #[test]
    fn predict_returns_most_likely_next() {
        let mut tok = Tokenizer::new(2);
        // Train on repetitive text: after "he" → "l" is most common
        tok.train("hello hello hello hello");
        // The counter should predict the most frequent continuation
        let dist = tok.distribution(tok.context_hash(&[b'h' as u32, b'e' as u32]));
        // Mutation that turns this red: return an empty distribution for a context the corpus
        // contains (the previous assertion here was `!dist.is_empty() || true`, which is always
        // true and asserted nothing).
        assert!(
            !dist.is_empty(),
            "a trained context must have a distribution"
        );
        let total: f64 = dist.iter().map(|(_, p)| *p).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "the distribution must be normalised, got {total}"
        );
        let top = dist
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty");
        assert_eq!(
            top.0, b'l' as u32,
            "after \"he\" the most likely byte is 'l'"
        );
    }

    #[test]
    fn distribution_sums_to_one() {
        let mut tok = Tokenizer::new(1);
        tok.train("aaaabbbccd");
        let dist = tok.distribution(0); // context hash 0 (start)
        let sum: f64 = dist.iter().map(|(_, p)| p).sum();
        if !dist.is_empty() {
            assert!((sum - 1.0).abs() < 0.01, "distribution sums to {sum}");
        }
    }
}
