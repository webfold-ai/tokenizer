# Contributing

Small crate, short rules. Read them before opening a pull request.

## What this project is

One fold, one file of logic, one dependency. The value of the crate is that it stays that way: a
tokenizer you can read, audit and reason about in an afternoon. A change that makes it faster but
unreadable, or that adds a dependency to save twenty lines, will be declined.

## Open in perpetuity

Every release stays under the licence it shipped with. An open-source licence grant cannot be
withdrawn, so a released version stays open whatever happens to the project later.

Contributions are accepted under the repository's existing `LICENSE` (inbound = outbound). You keep
copyright in what you write. There is no contributor licence agreement and no copyright assignment.

## Before you open a pull request

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo doc --no-deps
```

All four must pass locally. CI runs the same four plus `cargo deny check`. A pull request that
weakens a lint, skips a test or adds an `#[allow]` without a reason in the attribute is not ready.

## What a good change looks like

- **A behaviour change comes with a test, and the test comes with a control.** State which mutation
  turns the test red. A test that cannot fail is not a test.
- **A performance claim comes with three numbers**: what it was, what it is, and the machine and
  revision you measured on. A single timing is not a measurement.
- **A public item gets documentation.** `missing_docs` is denied; the doc comment says what the item
  does and what it refuses, not just what it is called.
- **No `unsafe`.** The crate is `#![forbid(unsafe_code)]`. If you believe you need it, open an issue
  with the measurement that motivates it first.
- **No new dependency** without an issue that says what it replaces and what it costs. The single
  dependency is a feature of the crate.

## Commits

- One logical change per commit, with a message that says what changed and why.
- Sign off your commits (`git commit -s`): `Signed-off-by: Your Name <you@example.com>`.
- Do not rewrite history that has been reviewed. Add commits; the maintainer squashes on merge.
- Never commit generated output, editor state, or a `Cargo.lock` change you did not intend.

## Review

Every pull request gets a review before merge, from someone who did not write it. Review looks for
the four things above, not for style preferences. The reviewer may ask for the negative control, the
measurement or the doc comment; that is not a formality.

The default branch is protected: no force pushes, no deletions, and CI must pass.

## Reporting a problem

Bugs and feature requests: open an issue with the template. Security: **do not open an issue** - see
[`SECURITY.md`](SECURITY.md).
