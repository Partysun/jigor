#!/usr/bin/env bash
# Build ONNX Runtime from source for Intel (x86_64) macOS and print the
# ORT_LIB_PATH to use. ort-sys ships prebuilt binaries for aarch64-apple-darwin
# only, and Microsoft dropped official Intel macOS packages at ONNX Runtime
# 1.24 — so on Intel Macs the library must be compiled once from source.
#
# Copied verbatim from lubo (same marker/versions/tree default) so releases on
# the shared macOS agent reuse the already-compiled onnxruntime checkout at
# $HOME/Dev/tmp/onnxruntime instead of rebuilding from scratch.
# Two govor-only deviations, both commented in place:
#   - --cmake_extra_defines is passed one define per argv (nargs="+");
#   - ensure_re2() rebuilds the re2 static libs when ORT's FetchContent
#     skipped them (FIND_PACKAGE_ARGS) and the ort-sys link would fail.
#
# Single arch (local dev, default): builds the host architecture into
#   build/MacOS/Release  (via build.sh, like the manual build)
# and prints that directory.
#
# Universal (CI: universal dmg): set ORT_OSX_ARCH=x86_64;arm64 (or "universal").
# ONNX Runtime's --osx_arch accepts one arch only, so each arch is built into
# its own tree (build/x86_64, build/arm64) and the static libraries are merged
# with lipo into build/universal (+ build/universal/_deps for ort-sys's
# full-static linking). Prints build/universal.
#
# IMPORTANT: stdout carries ONLY the final ORT_LIB_PATH (all progress goes to
# stderr) — callers capture it with ORT_LIB_PATH="$(bash script)".
#
# Usage:
#   bash scripts/build-onnxruntime-macos.sh            # host arch, prints ORT_LIB_PATH
#   ORT_OSX_ARCH=x86_64\;arm64 bash scripts/build-onnxruntime-macos.sh
#   ORT_LIB_PATH="$(bash scripts/build-onnxruntime-macos.sh)" cargo build
#
# Idempotent: skips the download and any already-built arch. The checkout dir
# is overridable with ONNX_RUNTIME_DIR.
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-v1.24.2}"
# Immutable commit for $ORT_VERSION (a tag can be force-moved; the commit
# cannot). Verified after clone/fetch so a tampered tag fails loudly:
#   git ls-remote https://github.com/microsoft/onnxruntime.git refs/tags/v1.24.2
ORT_COMMIT="${ORT_COMMIT:-058787ceead760166e3c50a0a4cba8a833a6f53f}"
SRC_DIR="${ONNX_RUNTIME_DIR:-$HOME/Dev/tmp/onnxruntime}"
ARCHS="${ORT_OSX_ARCH:-$(uname -m)}"

# KleidiAI must stay OFF: ORT 1.24 enables ARM SME kernels by default when
# building for aarch64, but cross-compiling the arm64 slice from an x86_64
# host leaves the kai_*_sme_mla symbols undefined at link time.
# CMAKE_OSX_DEPLOYMENT_TARGET matches the app's minimum (11.0) so the built
# objects stop claiming the host SDK version (e.g. 26.0).
# NOTE (deviation from lubo): ONNX Runtime's build.py passes
# --cmake_extra_defines through with nargs="+" — one define per argument.
# lubo's original single semicolon-separated string silently collapses all
# but the first define into CMAKE_IGNORE_PREFIX_PATH, leaving
# onnxruntime_BUILD_UNIT_TESTS on (tests get compiled into [all] and a later
# test target fails the link). Each define is therefore expanded separately
# at call time (unquoted "$CMAKE_DEFINES" still yields one argv per word).
CMAKE_DEFINES="CMAKE_IGNORE_PREFIX_PATH=$(brew --prefix) onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_USE_KLEIDIAI=OFF CMAKE_OSX_DEPLOYMENT_TARGET=11.0"

# A change in the defines below must rebuild the cached trees (they are
# otherwise idempotent and would keep the stale, KleidiAI-broken archives).
BUILD_ARGS_MARKER=".lubo-build-args"
BUILD_ARGS="$(printf '%s|%s|%s' "$CMAKE_DEFINES" "$ORT_VERSION" "$ORT_COMMIT")"

# Component libs that get lipo-merged into build/universal.
ORT_COMPONENTS=(
    onnxruntime_common
    onnxruntime_flatbuffers
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_lora
    onnxruntime_mlas
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_session
    onnxruntime_util
)

install_tools() {
    if ! command -v cmake >/dev/null; then
        brew install cmake
    fi
    if ! python3 -c "import numpy" >/dev/null 2>&1; then
        python3 -m pip install numpy
    fi
}

fetch_source() {
    if [ ! -f "$SRC_DIR/build.sh" ]; then
        # Shallow recursive clone (full clone often dies on flaky networks).
        git clone --recursive --depth 1 --shallow-submodules \
            --branch "$ORT_VERSION" \
            https://github.com/microsoft/onnxruntime.git "$SRC_DIR"
    fi
    # Verify the checkout is the pinned commit (protects against a force-pushed
    # or otherwise-replaced tag, e.g. a compromised/renamed release).
    actual="$(git -C "$SRC_DIR" rev-parse HEAD)"
    if [ "$actual" != "$ORT_COMMIT" ]; then
        echo "onnxruntime checkout is $actual, expected $ORT_COMMIT (tag $ORT_VERSION)" >&2
        echo "fix: ORT_COMMIT=<new-sha> bash scripts/build-onnxruntime-macos.sh" >&2
        exit 1
    fi
    echo "onnxruntime checkout verified: $ORT_COMMIT ($ORT_VERSION)" >&2
}

# build one architecture into $SRC_DIR/build/$1
build_arch() {
    local arch="$1"
    local build_dir="$SRC_DIR/build/$arch"
    local marker="$build_dir/$BUILD_ARGS_MARKER"
    if [ -f "$build_dir/Release/libonnxruntime_session.a" ]; then
        if [ "$(cat "$marker" 2>/dev/null)" = "$BUILD_ARGS" ]; then
            echo "ONNX Runtime already built for $arch: $build_dir/Release" >&2
            return
        fi
        echo "build arguments changed; rebuilding $arch" >&2
        rm -rf "$build_dir"
    fi
    (cd "$SRC_DIR" && python3 tools/ci_build/build.py \
        --build_dir "build/$arch" \
        --config Release \
        --osx_arch "$arch" \
        --skip_tests \
        --parallel \
        --cmake_extra_defines $CMAKE_DEFINES 1>&2)
    echo "$BUILD_ARGS" > "$marker"
}

# build the host architecture with build.sh (keeps the original build/MacOS layout)
build_host() {
    local build_dir="$SRC_DIR/build/MacOS"
    local marker="$build_dir/$BUILD_ARGS_MARKER"
    if [ -f "$build_dir/Release/libonnxruntime_session.a" ]; then
        if [ "$(cat "$marker" 2>/dev/null)" = "$BUILD_ARGS" ]; then
            echo "ONNX Runtime already built: $build_dir/Release" >&2
        else
            echo "build arguments changed; rebuilding" >&2
            rm -rf "$build_dir"
        fi
    fi
    if [ ! -f "$build_dir/Release/libonnxruntime_session.a" ]; then
        (cd "$SRC_DIR" && ./build.sh \
            --config Release \
            --osx_arch "$(uname -m)" \
            --skip_tests \
            --parallel \
            --cmake_extra_defines $CMAKE_DEFINES 1>&2)
    fi
    # ort-sys's full-static-linking layout expects `_deps` next to the lib dir
    # and looks for the flat `_deps/*-build` directories when no profile is set.
    [ -e "$build_dir/_deps" ] || ln -s Release/_deps "$build_dir/_deps"
    echo "$BUILD_ARGS" > "$marker"
    echo "$build_dir/Release"
}

# lipo-merge two per-arch builds into build/universal + build/universal/_deps
merge_universal() {
    local x86_dir="$SRC_DIR/build/x86_64/Release"
    local arm_dir="$SRC_DIR/build/arm64/Release"
    local out_dir="$SRC_DIR/build/universal"
    local deps_dir="$out_dir/_deps"
    local arch_a arch_b f rel target missing

    # Idempotency: skip only when every dep archive from BOTH trees is already
    # present. KleidiAI (aarch64-only) lives in the arm64 tree, so a cache
    # merged before this check existed must be completed, not skipped.
    missing=""
    if [ "$(cat "$out_dir/$BUILD_ARGS_MARKER" 2>/dev/null)" != "$BUILD_ARGS" ]; then
        echo "build arguments changed; re-merging universal tree" >&2
        rm -rf "$out_dir"
    fi
    if [ -f "$out_dir/libonnxruntime_session.a" ] && [ -f "$deps_dir/onnx-build/libonnx.a" ]; then
        for src_dir in "$x86_dir/_deps" "$arm_dir/_deps"; do
            while IFS= read -r -d '' f; do
                rel="${f#"$src_dir"/}"
                if [ ! -f "$deps_dir/$rel" ]; then
                    missing="yes"
                    break
                fi
            done < <(find "$src_dir" -type f -name '*.a' -print0 2>/dev/null)
            [ -z "$missing" ] || break
        done
        if [ -z "$missing" ]; then
            echo "Universal ONNX Runtime already merged: $out_dir" >&2
            echo "$out_dir"
            return
        fi
        echo "completing missing deps in $deps_dir (cache predates full merge)" >&2
    fi

    mkdir -p "$out_dir" "$deps_dir"
    for lib in "${ORT_COMPONENTS[@]}"; do
        arch_a="$x86_dir/lib${lib}.a"
        arch_b="$arm_dir/lib${lib}.a"
        [ -f "$arch_a" ] || { echo "missing $arch_a" >&2; exit 1; }
        [ -f "$arch_b" ] || { echo "missing $arch_b" >&2; exit 1; }
        if [ ! -f "$out_dir/lib${lib}.a" ]; then
            lipo -create "$arch_a" "$arch_b" -output "$out_dir/lib${lib}.a"
        fi
    done
    # ort-sys's full-static linking also needs every _deps archive (protobuf,
    # onnx, cpuinfo, re2, abseil subdirs, kleidiai, ...), with the same
    # relative paths as the per-arch builds. Some deps are arch-specific
    # (kleidiai exists only in the arm64 tree), so lipo what both arches have
    # and copy single-arch ones.
    while IFS= read -r -d '' f; do
        rel="${f#"$x86_dir/_deps"/}"
        target="$deps_dir/$rel"
        [ -f "$target" ] && continue
        arch_a="$x86_dir/_deps/$rel"
        arch_b="$arm_dir/_deps/$rel"
        if [ -f "$arch_a" ] && [ -f "$arch_b" ]; then
            mkdir -p "$(dirname "$target")"
            lipo -create "$arch_a" "$arch_b" -output "$target"
        elif [ -f "$arch_a" ]; then
            mkdir -p "$(dirname "$target")"
            cp "$arch_a" "$target"
            echo "single-arch dep copied: $rel (x86_64 only)" >&2
        fi
    done < <(find "$x86_dir/_deps" -type f -name '*.a' -print0 2>/dev/null)
    while IFS= read -r -d '' f; do
        rel="${f#"$arm_dir/_deps"/}"
        target="$deps_dir/$rel"
        [ -f "$target" ] && continue
        arch_a="$x86_dir/_deps/$rel"
        arch_b="$arm_dir/_deps/$rel"
        if [ -f "$arch_a" ] && [ -f "$arch_b" ]; then
            mkdir -p "$(dirname "$target")"
            lipo -create "$arch_a" "$arch_b" -output "$target"
        elif [ -f "$arch_b" ]; then
            mkdir -p "$(dirname "$target")"
            cp "$arch_b" "$target"
            echo "single-arch dep copied: $rel (arm64 only)" >&2
        fi
    done < <(find "$arm_dir/_deps" -type f -name '*.a' -print0 2>/dev/null)
    # Stamp merges so ci-macos-build.sh can drop ort-sys's cached rlib (which
    # embeds these archives at build-script time and never re-runs otherwise).
    touch "$out_dir/.merged-stamp"
    echo "$BUILD_ARGS" > "$out_dir/$BUILD_ARGS_MARKER"
    echo "$out_dir"
}

# ort-sys (build/static_link) unconditionally emits a link request for the
# re2 static library, but ONNX Runtime's FetchContent may not build re2 into
# the per-arch trees at all — its FetchContent_Declare has FIND_PACKAGE_ARGS
# (cmake/external/onnxruntime_external_deps.cmake), so a system re2 found by
# find_package makes FetchContent skip the build and the merged tree ends up
# without _deps/re2-build/libre2.a while the final app link still demands it.
# Rebuild re2 from source for both arches into the merged _deps so the link
# always succeeds. Idempotent; all progress on stderr (stdout carries only
# ORT_LIB_PATH).
ensure_re2() {
    local out_dir="$1"
    # ort-sys resolves external deps from <ORT_LIB_PATH>/_deps
    # (static_link/mod.rs: base_lib_dir/_deps), so the archives must land
    # inside the same _deps the merge script fills.
    local build_dir="$out_dir/_deps/re2-build"
    if [ -f "$build_dir/libre2.a" ] && \
       [ -f "$build_dir/.re2-absl-ok" ] && \
       lipo -info "$build_dir/libre2.a" | grep -qiE 'architectures in the fat file:.*x86_64.*arm64'; then
        return 0
    fi
    # Remove the wrongly-placed legacy dir (pre-_deps bug) and any previous
    # build that was compiled against the wrong (system) abseil namespace.
    [ -d "$out_dir/re2-build" ] && rm -rf "$out_dir/re2-build"
    rm -rf "$build_dir"
    local src=""
    # ORT's deps (FetchContent) live under build/<arch>/Release/_deps (the
    # merge script ships them into universal/_deps from exactly there).
    for d in "$SRC_DIR/build/x86_64/Release/_deps/re2-src" \
             "$SRC_DIR/build/arm64/Release/_deps/re2-src"; do
        [ -d "$d" ] && src="$d" && break
    done
    if [ -z "$src" ]; then
        src="$SRC_DIR/_re2-src"
        if [ ! -d "$src" ]; then
            echo "re2 source not in ORT trees; cloning google/re2 (pinned tag)" >&2
            git clone --quiet --depth 1 --branch 2024-07-02 \
                https://github.com/google/re2.git "$src"
        fi
    else
        echo "re2 source reused from ORT trees: $src" >&2
    fi
    # re2's CMake does find_package(absl REQUIRED) and links a set of absl
    # targets, but the undefined absl::lts_<ver> symbols at app link time must
    # resolve against the archives ort-sys ships from _deps/abseil_cpp-build
    # (the SAME abseil ORT was built with). A system (brew) abseil uses a
    # different LTS namespace and breaks the link. For COMPILATION re2 only
    # needs the abseil headers, so instead of resolving the abseil build tree
    # (which lacks a usable abslTargets.cmake) we stub every target re2 asks
    # for as an INTERFACE imported library pointing at the ORT trees' abseil
    # *_src checkout — headers from the exact revision ORT built with.
    local absl_src=""
    for d in "$SRC_DIR/build/x86_64/Release/_deps/abseil_cpp-src" \
             "$SRC_DIR/build/arm64/Release/_deps/abseil_cpp-src"; do
        [ -d "$d/absl" ] && absl_src="$d" && break
    done
    if [ -z "$absl_src" ]; then
        echo "abseil source not found in ORT trees; cannot build re2" >&2
        return 1
    fi
    echo "building re2 for x86_64 + arm64 into $build_dir" >&2
    mkdir -p "$build_dir"
    local pre="$build_dir/re2-absl-stub.cmake"
    {
        echo "# Auto-generated by build-onnxruntime-macos.sh (ensure_re2):"
        echo "# header-only stand-ins for the absl targets re2's CMake requires,"
        echo "# backed by the abseil checkout ORT itself was built with."
        echo "if(NOT TARGET absl::base)"
        echo "  set(_re2_absl_inc \"\${ABSEIL_SRC}\")"
        echo "  # Prepend ORT absl source to CMAKE_CXX_FLAGS so it's found before system absl"
        echo "  set(CMAKE_CXX_FLAGS \"-I\${_re2_absl_inc} \${CMAKE_CXX_FLAGS}\")"
        local pair flat ns
        for pair in "absl_absl_check;absl::absl_check" \
                    "absl_absl_log;absl::absl_log" \
                    "absl_base;absl::base" \
                    "absl_core_headers;absl::core_headers" \
                    "absl_fixed_array;absl::fixed_array" \
                    "absl_flags;absl::flags" \
                    "absl_flat_hash_map;absl::flat_hash_map" \
                    "absl_flat_hash_set;absl::flat_hash_set" \
                    "absl_hash;absl::hash" \
                    "absl_inlined_vector;absl::inlined_vector" \
                    "absl_optional;absl::optional" \
                    "absl_span;absl::span" \
                    "absl_str_format;absl::str_format" \
                    "absl_strings;absl::strings" \
                    "absl_synchronization;absl::synchronization"; do
            flat="${pair%%;*}"
            ns="${pair#*;}"
            echo "  add_library($flat INTERFACE IMPORTED)"
            echo "  set_property(TARGET $flat PROPERTY INTERFACE_INCLUDE_DIRECTORIES \"\${_re2_absl_inc}\")"
            echo "  add_library(${ns} ALIAS $flat)"
        done
        echo "endif()"
    } > "$pre"
    local arch build
    for arch in x86_64 arm64; do
        build="$build_dir/$arch"
        # Point CMake to ORT's absl build dir so find_package(absl) finds the right one
        local absl_build_dir="$SRC_DIR/build/$arch/Release/_deps/abseil_cpp-build"
        echo "re2($arch): absl headers from $absl_src, absl config from $absl_build_dir" >&2
        cmake -S "$src" -B "$build" \
            -DCMAKE_BUILD_TYPE=Release \
            -DRE2_BUILD_TESTING=OFF \
            -DCMAKE_PROJECT_INCLUDE_BEFORE="$pre" \
            -DABSEIL_SRC="$absl_src" \
            -DCMAKE_CXX_STANDARD=17 \
            -DCMAKE_CXX_STANDARD_REQUIRED=ON \
            -DCMAKE_CXX_FLAGS="-I${absl_src}" \
            -DCMAKE_IGNORE_PREFIX_PATH=/usr/local \
            -DCMAKE_PREFIX_PATH="$absl_build_dir" \
            -DCMAKE_CXX_STANDARD=17 \
            -DCMAKE_CXX_STANDARD_REQUIRED=ON \
            -DCMAKE_OSX_ARCHITECTURES="$arch" \
            -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 >/dev/null || return 1
        cmake --build "$build" --target re2 -j >/dev/null || return 1
    done
    lipo -create "$build_dir/x86_64/libre2.a" "$build_dir/arm64/libre2.a" \
        -output "$build_dir/libre2.a" || return 1
    touch "$build_dir/.re2-absl-ok"
    echo "re2 merged: $build_dir/libre2.a" >&2
}

install_tools
fetch_source

case "$ARCHS" in
    universal | *\;*)
        build_arch x86_64
        build_arch arm64
        merge_universal
        ensure_re2 "$SRC_DIR/build/universal"
        ;;
    *)
        build_host
        ;;
esac
