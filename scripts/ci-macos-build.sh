#!/usr/bin/env bash
# Build and publish the universal macOS wheel. Called by the
# macos-publish step of .woodpecker/release-macos.yml (darwin agent).
#
# Living in a script file (not inline YAML) matters: Woodpecker rewrites
# inline step commands with its envsubst (${...} gets interpolated and can
# even make the config fail to parse), while script files reach bash
# untouched.
#
# ort-sys ships prebuilt binaries for aarch64-apple-darwin only, and
# Microsoft dropped official Intel macOS packages at ONNX Runtime 1.24 —
# so the universal2 wheel needs a source-compiled ONNX Runtime with the
# x86_64 + arm64 static archives lipo-merged (scripts/build-onnxruntime-macos.sh,
# same flow as govor/lubo). ORT_LIB_PATH makes ort-sys skip
# download-binaries and statically link these universal archives.
# Persistent caches under ~/Library/Caches/jigor-ci keep the huge native
# builds (ONNX Runtime + ort-sys) incremental.
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
# no universal2-apple-darwin target exists on this toolchain; like govor's
# ci-macos-build.sh, install both per-arch targets and let maturin's
# --universal2 merge them
rustup target add aarch64-apple-darwin x86_64-apple-darwin

CACHE="$(eval echo "~$(id -un)")/Library/Caches/jigor-ci"
mkdir -p "$CACHE"
export CARGO_TARGET_DIR="$CACHE/target"

# Force ort-sys to rebuild against the universal ONNX Runtime (with
# merged absl + re2) so it picks up the correct absl namespace.
cargo clean -p ort-sys

# The universal build needs a source-compiled ONNX Runtime (ort-sys ships
# prebuilt binaries for arm64 macOS only): built once into the persistent
# cache root. ~30-60 min first run, idempotent afterwards. ORT_LIB_PATH
# makes ort-sys skip download-binaries and statically link these universal
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

# ort-sys embeds the ORT static archives into its rlib when its build
# script runs; drop stale artifacts from before the last universal
# merge so a cached rlib (built against the old prebuilt tree) can
# never be reused (same as govor's ci-macos-build.sh).
stamp="$ORT_LIB_PATH/.merged-stamp"
if [ -f "$stamp" ]; then
  while IFS= read -r -d '' release; do
    find "$release" \
      \( -path '*/build/ort-sys-*' -o -path '*/deps/libort_sys-*' -o -path '*/.fingerprint/ort-sys-*' \) \
      ! -newer "$stamp" -exec rm -rf {} + 2>/dev/null || true
  done < <(find "$CARGO_TARGET_DIR" -maxdepth 3 -type d -name release -print0 2>/dev/null)
fi

echo "Building the universal2 wheel"
cargo install --quiet uv || true
uv tool install --quiet maturin || true
uv tool run maturin build --release --universal2 --out dist
uv publish dist/*