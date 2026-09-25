#!/usr/bin/env bash
# Thin wrapper: the canonical script ships inside the `jigor` crate at
# crates/jigor/build-support/build-onnxruntime-macos.sh so
# `crates/jigor/build.rs` can run it from a packaged .crate (plain
# `cargo install jigor-cli` on an Intel Mac). Keep calling this path —
# the repo's CI (ci-macos-build.sh, build-npm-darwin.sh) captures its
# stdout (ORT_LIB_PATH) exactly as before.
exec bash "$(cd "$(dirname "$0")/.." && pwd)/crates/jigor/build-support/build-onnxruntime-macos.sh" "$@"
