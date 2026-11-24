# Security policy

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting:
<https://github.com/webfold-ai/tokenizer/security/advisories/new>, or by email to
security@webfold.ai. Do not open a public issue for a suspected vulnerability.

Include the affected revision, the input that triggers it (or how to build one), and what you
observed. A report is acknowledged within 3 days. A confirmed issue gets a fix or a published
advisory within 90 days of the report, sooner when it is being exploited. Reporters are credited in
the advisory unless they ask otherwise.

The project signs no non-disclosure agreement. A vulnerability found with AI tools is treated as
already public: anyone running the same tools can find it, so its fix is not held back for a
coordinated date.

## Supported versions

| version | supported |
|---|---|
| `main` | yes |
| `0.1.x` | yes, latest patch only |
| older | no |

Before 1.0, fixes land on `main` and in the next `0.1.x` patch release; there are no backports.

## Scope

In scope:

- Memory safety and panic-freedom on hostile input: any byte sequence passed to `train`, `encode`,
  `decode`, `context_hash`, `predict` or `distribution`. The crate is `#![forbid(unsafe_code)]`, so a
  memory-safety defect is either a compiler or a dependency bug — say which you believe it is.
- Correctness of the round trip: `decode(encode(x))` returning anything other than `x` for input the
  tokenizer was trained on.
- Denial of service that is *unbounded*: input whose cost grows super-linearly in the input size.
  Cost that grows with the corpus (a large vocabulary) is by design and is not a vulnerability.
- Dependency vulnerabilities reachable through this crate's single dependency.

Out of scope:

- Tokenization quality: a vocabulary that segments your text badly is a corpus and order question,
  not a security report.
- The `apeiron` crate's own internals. Report those upstream, and tell us so we can pin a fix.
- Anything requiring a modified build of Rust, or an attacker who can already write the crate's
  source.

## What this crate does with your data

It reads the bytes you hand it and keeps counts in memory. It makes no network calls, opens no files,
spawns no processes, and writes nothing to disk. There is no telemetry.

## Hardening in the build

`cargo deny check` (advisories, licences, bans, sources) and `cargo clippy -- -D warnings` run in CI
on every pull request. Dependencies are pinned by revision and updated by Dependabot, not silently.
