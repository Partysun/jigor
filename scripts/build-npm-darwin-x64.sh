#!/usr/bin/env bash
# Build and publish the x86_64 macOS jigor binary as the npm
# `jigor-darwin-x64` platform package. Called by the npm-darwin-x64 step of
# .woodpecker/npm.yml (darwin/amd64 agent).
#
# Living in a script file (not inline YAML) matters: Woodpecker rewrites
# inline step commands with its envsubst (${...} gets interpolated and can
# even make the config fail to parse), while script files reach bash
# untouched.
#
# ort-sys ships prebuilt binaries for aarch64-apple-darwin only, and
# Microsoft dropped official Intel macOS packages at ONNX Runtime 1.24 —
# so the x64 binary links a source-compiled ONNX Runtime
# (scripts/build-onnxruntime-macos.sh with ORT_OSX_ARCH='x86_64', same flow
# as the universal2 wheel in scripts/ci-macos-build.sh). ORT_LIB_PATH makes
# ort-sys skip download-binaries and statically link the source-built
# archives. Persistent caches under ~/Library/Caches/jigor-ci keep the huge
# native builds incremental.
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
rustup default stable

CACHE="$(eval echo "~$(id -un)")/Library/Caches/jigor-ci"
mkdir -p "$CACHE"
export CARGO_TARGET_DIR="$CACHE/target"

# Force ort-sys to rebuild against the source-compiled ONNX Runtime
# (with merged absl + re2) so it picks up the correct absl namespace.
cargo clean -p ort-sys

# x86_64 source build; ~30-60 min first run, idempotent afterwards.
ONNX_RUNTIME_DIR="$CACHE/onnxruntime"
ORT_LIB_PATH="$(ORT_OSX_ARCH='x86_64' ONNX_RUNTIME_DIR="$ONNX_RUNTIME_DIR" bash scripts/build-onnxruntime-macos.sh | tail -n 1)"
test -f "$ORT_LIB_PATH/libonnxruntime_session.a" || {
  echo "bad ORT_LIB_PATH: '$ORT_LIB_PATH'" >&2
  exit 1
}
export ORT_LIB_PATH
echo "ORT_LIB_PATH: $ORT_LIB_PATH"

cargo build -p jigor-cli --release

BIN="$CARGO_TARGET_DIR/release/jigor"
if [ ! -f "$BIN" ]; then
  BIN="target/release/jigor"
fi
cp "$BIN" npm/platforms/darwin-x64/
./npm/platforms/darwin-x64/jigor models

echo "//registry.npmjs.org/:_authToken=$NPM_TOKEN" >> ~/.npmrc
node scripts/npm-version.js npm/platforms/darwin-x64
cd npm/platforms/darwin-x64
npm publish --access public