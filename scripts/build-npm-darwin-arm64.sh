#!/usr/bin/env bash
# Build and publish the Apple Silicon jigor binary as the npm
# `jigor-darwin-arm64` platform package. Called by the npm-darwin-arm64 step
# of .woodpecker/npm.yml (darwin/arm64 agent).
#
# Unlike the x64 package this needs no source-compiled ONNX Runtime —
# ort-sys ships a prebuilt aarch64-apple-darwin runtime — so it is a plain
# native `cargo build`.
#
# Living in a script file (not inline YAML) matters: Woodpecker rewrites
# inline step commands with its envsubst (${...} gets interpolated and can
# even make the config fail to parse), while script files reach bash
# untouched.
set -euo pipefail

export MACOSX_DEPLOYMENT_TARGET=11.0
export CMAKE_OSX_DEPLOYMENT_TARGET=11.0

export RUSTUP_HOME="$(eval echo "~$(id -un)")/.rustup"
export CARGO_HOME="$(eval echo "~$(id -un)")/.cargo"
export PATH="$CARGO_HOME/bin:$PATH"
rustup default stable

cargo build -p jigor-cli --release
cp target/release/jigor npm/platforms/darwin-arm64/

node scripts/npm-version.js npm/platforms/darwin-arm64
cd npm/platforms/darwin-arm64
npm publish --access public