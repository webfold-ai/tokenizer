#!/usr/bin/env bash
# The four gates CI runs, one per line with its exit code. No piping: the exit code is the result.
set -u
status=0
run() {
  name="$1"; shift
  "$@" >"/tmp/gate-$name.log" 2>&1
  code=$?
  printf 'GATE %-10s exit=%s\n' "$name" "$code"
  [ "$code" -eq 0 ] || { printf -- '--- %s (tail) ---\n' "$name"; tail -15 "/tmp/gate-$name.log"; status=1; }
}
run fmt cargo fmt --check
run clippy cargo clippy --all-targets -- -D warnings
run test cargo test
run doc env RUSTDOCFLAGS=-D\ warnings cargo doc --no-deps
printf 'GATES_TOTAL exit=%s\n' "$status"
exit "$status"
