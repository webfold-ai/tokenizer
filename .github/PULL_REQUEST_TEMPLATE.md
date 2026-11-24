## What changed

<!-- One paragraph. What behaviour is different after this, and for whom. -->

## Why

<!-- The problem, not the solution. Link the issue if there is one. -->

## Evidence

- [ ] A test that fails without this change, and the mutation that turns it red:
- [ ] If this is a performance claim: before, after, machine, revision
- [ ] If this touches a public item: the doc comment is updated

## Checks

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test`
- [ ] `cargo doc --no-deps`
- [ ] Commits are signed off
