# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

### Added

- `Tokenizer` over an n-gram counter: `new`, `train`, `encode`, `decode`, `vocab_size`,
  `predict`, `distribution`, `context_hash`.
- `TokenId` (`u32`) and the Set/Map/Count faces the tokenizer is built from.
- `#![forbid(unsafe_code)]`, `deny(missing_docs)`, and one dependency (`apeiron`).

[Unreleased]: https://github.com/webfold-ai/tokenizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/webfold-ai/tokenizer/releases/tag/v0.1.0
