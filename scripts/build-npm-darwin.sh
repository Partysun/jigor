#!/usr/bin/env bash
# Build and publish the universal2 macOS jigor binary as the npm
# `jigor-darwin` platform package. Called by the npm-darwin step of
# .woodpecker/npm-darwin.yml (the single darwin/amd64 agent — there is no
# arm64 Mac; the universal2 fat binary covers Intel AND Apple Silicon).
#
# Living in a script file (not inline YAML) matters: Woodpecker rewrites
# inline step commands with its envsubst (${...} gets interpolated and can
# even make the config fail to parse), while script files reach bash
# untouched.
#
# ort-sys ships prebuilt binaries for aarch64-apple-darwin only, and
# Microsoft dropped official Intel macOS packages at ONNX Runtime 1.24 —
# so the x86_64 side needs a source-compiled ONNX Runtime
# (scripts/build-onnxruntime-macos.sh with ORT_OSX_ARCH='x86_64;arm64',
# same flow as the universal2 wheel in scripts/ci-macos-build.sh).
# ORT_LIB_PATH makes ort-sys skip download-binaries and statically link
# the merged universal archives; the `universal2-apple-darwin` rustup
# target then links one fat binary that runs on both Mac architectures.
# Persistent caches under ~/Library/Caches/jigor-ci keep the huge native
# builds incremental.
set -euo pipefail

# Match the app's minimum macOS: cc-built C/C++ (ONNX Runtime, ...) otherwise
# targets the SDK's default deployment version and risks availability gates.
# Hard 11.0 (a host-provided value must not leak in), same as govor's
# ci-macos-build.sh.
export MACOSX_DEPLOYMENT_TARGET=11.0
export CMAKE_OSX_DEPLOYMENT_TARGET=11.0
echo "deployment target: $MACOSX_DEPLOYMENT_TARGET / $CMAKE_OSX_DEPLOYMENT_TARGET"

export RUSTUP_HOME="$(eval echo "~$(id -un)")/.rustup"
export CARGO_HOME="$(eval echo "~$(id -un)")/.cargo"
export PATH="$CARGO_HOME/bin:$PATH"
rustup component add rustfmt clippy 2>/dev/null || true
rustup default stable
# no universal2-apple-darwin target exists on this toolchain; build both
# arch targets explicitly (like govor's ci-macos-build.sh) and merge with
# Xcode's lipo below
rustup target add aarch64-apple-darwin x86_64-apple-darwin

CACHE="$(eval echo "~$(id -un)")/Library/Caches/jigor-ci"
mkdir -p "$CACHE"

# Force ort-sys to rebuild against the merged ONNX Runtime (with merged
# absl + re2) so it picks up the correct absl namespace.
cargo clean -p ort-sys

# The universal build needs a source-compiled ONNX Runtime (ort-sys ships
# prebuilt binaries for arm64 macOS only): built once into the persistent
# cache root, ~30-60 min first run, idempotent afterwards. ORT_LIB_PATH
# makes ort-sys skip download-binaries and statically link the universal
# archives (lib*.a + _deps).
ONNX_RUNTIME_DIR="$CACHE/onnxruntime"
# stdout of the script may carry stray git submodule lines (clone
# noise), so take the LAST line only; the script prints the final
# path last. pipefail keeps the script's failure exit code.
ORT_LIB_PATH="$(ORT_OSX_ARCH='x86_64;arm64' ONNX_RUNTIME_DIR="$ONNX_RUNTIME_DIR" bash scripts/build-onnxruntime-macos.sh | tail -n 1)"
test -f "$ORT_LIB_PATH/libonnxruntime_session.a" || {
  echo "bad ORT_LIB_PATH: '$ORT_LIB_PATH'" >&2
  exit 1
}
export ORT_LIB_PATH
echo "ORT_LIB_PATH: $ORT_LIB_PATH"

cargo build -p jigor-cli --release --target x86_64-apple-darwin
cargo build -p jigor-cli --release --target aarch64-apple-darwin

X64_BIN="$CARGO_TARGET_DIR/x86_64-apple-darwin/release/jigor"
ARM_BIN="$CARGO_TARGET_DIR/aarch64-apple-darwin/release/jigor"
[ -f "$X64_BIN" ] || X64_BIN="target/x86_64-apple-darwin/release/jigor"
[ -f "$ARM_BIN" ] || ARM_BIN="target/aarch64-apple-darwin/release/jigor"
test -f "$X64_BIN" || { echo "missing x86_64 binary: $X64_BIN" >&2; exit 1; }
test -f "$ARM_BIN" || { echo "missing arm64 binary: $ARM_BIN" >&2; exit 1; }
lipo -create -output npm/platforms/darwin/jigor "$X64_BIN" "$ARM_BIN"
./npm/platforms/darwin/jigor models

echo "//registry.npmjs.org/:_authToken=$NPM_TOKEN" >> ~/.npmrc
node scripts/npm-version.js npm/platforms/darwin
cd npm/platforms/darwin
npm publish --access public