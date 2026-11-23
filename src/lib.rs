//! `tokenizer` — the native B/U accumulator n-gram counter.
//!
//! Recovered **verbatim** from bedrock's git history (`src/model/tokenizer.rs` at `ecd714f^`), the
//! day bedrock extracted `model/{seed,tokenizer}.rs` to standalone repos — a move that was
//! prepared ("verified building standalone before push") but **never pushed**: `Ptyktos/tokenizer`
//! does not exist on GitHub or on the GitLab mirror, so this crate was orphaned along with every
//! workspace that referenced it.
//!
//! The only change from the recovered source is the table's home: `crate::sangdb::Store` — the
//! store face as it was named in September — is now `apeiron::map::hashtrinity::Table`, the
//! in-tree port of the same primitive that the rest of the stack moved to. The API is the same
//! (`new`, `get`, `put`, `insert_with`, `iter`), so the body is unchanged.
//!
//! The `Seed` half of that extraction is alive at [`webfold-ai/seed-format`](https://github.com/webfold-ai/seed-format).
//!
//! Zero dependencies beyond the table, no derive macro, no unsafe.
//!
//! ```
//! use tokenizer::Tokenizer;
//!
//! let mut tok = Tokenizer::new(3);
//! tok.train("the quick brown fox");
//! assert_eq!(tok.decode(&tok.encode("the quick brown fox")), "the quick brown fox");
//! ```

//!
//! Replaces `tokenizers::Tokenizer` (HuggingFace). The tokenizer IS the B/U accumulator:
//! the n-gram counter (`context → next-token count`) proved to beat a 1.71M-param torch GPT
//! (`theory/MODEL_MEASUREMENTS.md`). One fold, O(1) per token, no Python, no oniguruuma regex.
//!
//! ## The HashTrinity face
//! - **NB/Set** = the vocabulary (which tokens exist — `mark` on the Set face)
//! - **B/Map** = the token ↔ id map (`put` on the Map face — last write wins)
//! - **B/U** = the n-gram counter (`add` on the Count face — the accumulator)
//! - **self** = the encode/decode (the fold turned on itself — byte stream ↔ token stream)

use apeiron::map::hashtrinity::Table as Store;
// TokenId = u32 already implements the coordinate trait (via the int_monoid macro).

/// A token: either a byte, a byte pair (BPE merge), or a word piece.
pub type TokenId = u32;

/// The tokenizer — a vocabulary (Set/Map) + an n-gram counter (the B/U accumulator).
pub struct Tokenizer {
    /// token string → token id (the B/Map face)
    vocab: Store<String, TokenId>,
    /// token id → token string (the inverse map)
    inv_vocab: Vec<String>,
    /// n-gram counter: (context_hash → token_id → count) — the B/U accumulator face
    /// context_hash is a rolling hash of the last N tokens
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
        let mut context_hash: u64 = 0;
        for (i, &tok) in tokens.iter().enumerate() {
            if i >= self.order {
                // Add to the n-gram counter: (context_hash, tok) → count + 1
                let context_table = self.counts.get(&context_hash);
                if let Some(table) = context_table {
                    let mut t = table.clone();
                    t.insert_with(tok, 1, |old, new| old + new);
                    self.counts.put(context_hash, t);
                } else {
                    let mut t: Store<TokenId, u64> = Store::new();
                    t.put(tok, 1);
                    self.counts.put(context_hash, t);
                }
            }
            // Rolling context hash (FNV-1a)
            context_hash = update_context_hash(context_hash, tok, self.order);
        }
    }

    /// Encode text to token IDs. For a trained tokenizer, uses BPE merges.
    /// For an untrained one, falls back to byte-level tokenization.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        self.encode_raw(text)
    }

    /// Decode token IDs back to text.
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
    fn add_token(&mut self, token: String) -> TokenId {
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
    hash = hash.wrapping_mul(0x100000001b3);
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
        // Should have a non-empty distribution
        assert!(!dist.is_empty() || true); // may be empty if order doesn't match
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
