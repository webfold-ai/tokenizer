# tokenizer

**A tokenizer that is a fold.** Text in, token ids out, ids back to text — and the whole thing is
one n-gram counter over a hash table, so the cost per token is a fixed number of operations and the
memory is the table, not a model.

It replaces `tokenizers::Tokenizer` for workloads that do not need a `tokenizer.json`: no BPE
training step, no merge ranks, no oniguruma regex, no Python, no unsafe. One dependency
(`apeiron`), one file of logic, and an API small enough to read in a sitting.

```rust
use tokenizer::Tokenizer;

let mut tok = Tokenizer::new(3);
tok.train("the quick brown fox jumps over the lazy dog");

let ids = tok.encode("the quick brown fox");
assert_eq!(tok.decode(&ids), "the quick brown fox");
```

## Why a fold is a tokenizer

A tokenizer is two tables and a count:

| face | what it holds | operation |
|---|---|---|
| **Set** | the vocabulary: which tokens exist | `mark` |
| **Map** | token ↔ id | `put` (last write wins) |
| **Count** | the n-gram counter: context → next-token counts | `add` (the accumulator) |
| **self** | encode and decode | the fold turned on itself |

Encoding folds the byte stream into contexts the counter has seen; decoding folds the ids back out.
Because the counter is the model, there is no separate training phase beyond counting the corpus, and
the per-token cost is a fixed number of probes — not a merge loop over a ranked list.

In our own measurements the counter outperforms a 1.71M-parameter torch GPT trained on the same
corpus. That harness is not in this repository; treat the sentence as a claim about our workloads
rather than a benchmark you can reproduce here, and measure your own.

## Use

```sh
cargo add --git https://github.com/webfold-ai/tokenizer
```

```rust
let mut tok = Tokenizer::new(4);        // context order: 4 bytes of context
tok.train(&corpus);                     // count once
let ids = tok.encode("hello world");    // Vec<TokenId>
let text = tok.decode(&ids);            // String

tok.vocab_size();                       // how many tokens the corpus produced
tok.predict(context_hash);              // the single most likely next token
tok.distribution(context_hash);         // the ranked distribution behind it
```

`TokenId` is `u32`. `context_hash` is what `Tokenizer::context_hash(&tokens)` returns for a token
window, so a caller can look up a distribution without re-encoding the prefix.

## What it is not

- Not a drop-in for a pretrained model's tokenizer. The vocabulary is whatever your corpus produced;
  ids do not line up with anyone else's.
- Not a subword segmenter with merges. There is no `tokenizer.json`, no special-token table, no
  `added_tokens`.
- Not async, not streaming, not `no_std`. It is a table and a loop.

## Guarantees

- `#![forbid(unsafe_code)]`.
- One dependency: `apeiron`, for the table and its coordinate arithmetic.
- `encode` then `decode` returns the input bytes for any input the tokenizer has been trained on,
  including inputs that end mid-character.
- Every public item is documented; `missing_docs` is denied, not warned.

## Building and testing

```sh
cargo build --lib
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

MSRV is stated in `Cargo.toml`; the crate tracks the stable toolchain the workspace uses.

## Security

See [`SECURITY.md`](SECURITY.md). Report privately; do not open a public issue.

## Licence

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
