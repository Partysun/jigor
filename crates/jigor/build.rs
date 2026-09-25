//! Build script for the `jigor` library — ONNX Runtime linking on Intel Mac.
//!
//! `ort-sys` ships ONNX Runtime prebuilts for every tier-1 target except
//! `x86_64-apple-darwin` (Microsoft dropped Intel macOS builds at 1.24), so
//! on an Intel Mac `cargo install jigor-cli` dies inside ort-sys's own build
//! script with "no prebuilt binaries available for target" before this crate
//! is even compiled. For that one target we disable ort-sys's linking (see
//! the target-scoped `ort-sys` dependency with `disable-linking` in
//! Cargo.toml) and take over here:
//!
//!   1. `ORT_LIB_PATH`/`ORT_LIB_LOCATION` set → link those archives (the
//!      dev/CI flow, identical to what ort-sys would do with them);
//!   2. otherwise → build ONNX Runtime once from source via
//!      `build-support/build-onnxruntime-macos.sh` (shipped inside this
//!      crate, git clone + CMake; ~15–60 min on the first run, cached and
//!      idempotent afterwards) and link the result.
//!
//! The emitted directives mirror ort-sys's `build/static_link` logic — same
//! search dirs, same link order — so the resulting binary matches a build
//! that went through ort-sys's normal prebuilt flow. Every other target
//! returns immediately and keeps ort-sys's download path untouched.

use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// The single target without ort-sys prebuilts; anything else is ort-sys's
/// problem, not ours.
const TARGET: &str = "x86_64-apple-darwin";

/// ONNX Runtime component archives of a full-static build, in ort-sys's
/// link order (`build/static_link/mod.rs`).
const COMPONENTS: [&str; 10] = [
    "common",
    "flatbuffers",
    "framework",
    "graph",
    "lora",
    "mlas",
    "optimizer",
    "providers",
    "session",
    "util",
];

fn main() {
    if env::var("TARGET").unwrap_or_default() != TARGET {
        return;
    }

    // ort-sys registers these before deciding what to link; mirror them so
    // changes re-run this script.
    for var in [
        "ORT_LIB_PATH",
        "ORT_LIB_LOCATION",
        "ORT_LIB_PROFILE",
        "ORT_PREFER_DYNAMIC_LINK",
        "ORT_CXX_STDLIB",
        "CXXSTDLIB",
        "CC",
        "DEVELOPER_DIR",
        "ONNX_RUNTIME_DIR",
        "ORT_OSX_ARCH",
        "ORT_VERSION",
        "ORT_COMMIT",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build-support/build-onnxruntime-macos.sh");

    let base = resolve_lib_dir();

    // parity with ort-sys's dynamic-link escape hatch
    if prefer_dynamic_linking() {
        println!("cargo:rustc-link-lib=onnxruntime");
        println!("cargo:rustc-link-search=native={}", base.display());
        return;
    }

    static_link(&base);
    static_link_prerequisites();
}

fn resolve_lib_dir() -> PathBuf {
    for var in ["ORT_LIB_PATH", "ORT_LIB_LOCATION"] {
        if let Ok(dir) = env::var(var)
            && !dir.is_empty()
        {
            let dir = PathBuf::from(dir);
            verify(&dir);
            return dir;
        }
    }
    build_from_source()
}

/// Run the bundled script (git clone + CMake) and return the printed
/// ORT_LIB_PATH. Progress goes to the inherited stderr; stdout carries only
/// the final path (possibly preceded by stray git submodule lines).
fn build_from_source() -> PathBuf {
    let manifest = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    let script = manifest
        .join("build-support")
        .join("build-onnxruntime-macos.sh");

    println!(
        "cargo:warning=jigor: no ORT_LIB_PATH set — building ONNX Runtime for {TARGET} \
		 from source (first run takes 15-60 min, cached afterwards; set ORT_LIB_PATH to skip)"
    );

    let mut cmd = Command::new("bash");
    cmd.arg(&script)
        // deterministic arch: also lets an arm64 host cross-build the x86_64
        // target; a pre-set ORT_OSX_ARCH (universal CI flow) wins
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if env::var_os("ORT_OSX_ARCH").is_none_or(|v| v.is_empty()) {
        cmd.env("ORT_OSX_ARCH", "x86_64");
    }

    let mut child = cmd
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn `bash {}`: {e}", script.display()));

    let mut stdout = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        use std::io::Read;
        let _ = pipe.read_to_string(&mut stdout);
    }
    let status = child
        .wait()
        .expect("failed to wait for build-onnxruntime-macos.sh");

    let path = stdout.lines().map(str::trim).rfind(|l| !l.is_empty());

    if !status.success() || path.is_none() {
        panic!(
            "jigor: build-onnxruntime-macos.sh failed ({status}).\n\
			 ONNX Runtime could not be built; see the script output above.\n\
			 Requirements: git, cmake, python3 (with numpy), Xcode command line tools,\n\
			 ~10 GB free disk. Alternatives:\n\
			   * point ORT_LIB_PATH at an existing ONNX Runtime build, or\n\
			   * install jigor from the prebuilt packages (npm/pip) instead of cargo"
        );
    }

    let dir = PathBuf::from(path.expect("checked above"));
    verify(&dir);
    dir
}

/// The directory ort-sys's profile detection would end up linking from.
fn find_lib_dir(base: &Path) -> Option<PathBuf> {
    if base.join("libonnxruntime_session.a").exists() {
        return Some(base.to_path_buf());
    }
    ["Release", "RelWithDebInfo", "MinSizeRel", "Debug"]
        .iter()
        .map(|i| base.join(i))
        .find(|dir| dir.join("libonnxruntime_session.a").exists())
}

fn verify(base: &Path) {
    if find_lib_dir(base).is_some() {
        return;
    }
    panic!(
        "jigor: no ONNX Runtime libraries found in `{}`\n\
		 expected libonnxruntime_session.a directly in that directory or under a\n\
		 Release/ (RelWithDebInfo/, MinSizeRel/, Debug/) subdirectory.\n\
		 Fix one of:\n\
		   * point ORT_LIB_PATH at an existing ONNX Runtime build\n\
		     (scripts/build-onnxruntime-macos.sh prints one), or\n\
		   * unset ORT_LIB_PATH so jigor builds ONNX Runtime from source\n\
		     (needs git, cmake, python3 + Xcode CLT; ~15-60 min on the first run)",
        base.display()
    );
}

fn prefer_dynamic_linking() -> bool {
    match env::var("ORT_PREFER_DYNAMIC_LINK") {
        Ok(val) => val == "1" || val.to_lowercase() == "true",
        Err(_) => false,
    }
}

fn add_search_dir(dir: &Path) {
    if dir.join("Release").is_dir() {
        println!(
            "cargo:rustc-link-search=native={}",
            dir.join("Release").display()
        );
    } else if dir.join("Debug").is_dir() {
        println!(
            "cargo:rustc-link-search=native={}",
            dir.join("Debug").display()
        );
    } else {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
}

fn optional_link_lib(dir: &Path, lib: &str) -> bool {
    if dir.exists() && dir.join(format!("lib{lib}.a")).exists() {
        add_search_dir(dir);
        println!("cargo:rustc-link-lib=static={lib}");
        true
    } else {
        false
    }
}

/// Mirror of ort-sys's `static_link` (build/static_link/mod.rs) for the
/// macOS full-static case: identical search dirs and link order. Tries the
/// same candidate layouts and stops at the first one that holds all
/// component archives.
fn static_link(base: &Path) {
    let mut profile = env::var("ORT_LIB_PROFILE").unwrap_or_default();
    if profile.is_empty() {
        for i in ["Release", "RelWithDebInfo", "MinSizeRel", "Debug"] {
            if base.join(i).join("libonnxruntime_common.a").exists() {
                profile = i.to_owned();
                break;
            }
        }
    }
    let transform = |p: PathBuf| p.join(&profile);

    add_search_dir(base);

    if base.join("libonnxruntime.a").exists() {
        println!("cargo:rustc-link-lib=static=onnxruntime");
        return;
    }

    // ort-sys's static_configs, minus the vcpkg/windows flavours that can
    // never occur in a jigor build: (component libs, ortcustomops dir, _deps dir)
    let configs: [(PathBuf, PathBuf, PathBuf); 4] = [
        (
            transform(base.to_path_buf()),
            base.join("lib"),
            base.join("_deps"),
        ),
        (
            transform(base.to_path_buf()),
            base.join("lib"),
            transform(base.join("_deps")),
        ),
        (
            base.to_path_buf(),
            base.join("lib"),
            base.parent()
                .map_or_else(|| base.join("_deps"), |p| p.join("_deps")),
        ),
        (
            base.join("onnxruntime"),
            base.join("onnxruntime").join("lib"),
            base.join("_deps"),
        ),
    ];

    for (lib_dir, ext_dir, external) in configs {
        if !lib_dir.join("libonnxruntime_common.a").exists() {
            continue;
        }
        // check every component before emitting anything so a half-matching
        // layout can't leave stray -l flags behind
        if !COMPONENTS
            .iter()
            .all(|c| lib_dir.join(format!("libonnxruntime_{c}.a")).exists())
        {
            eprintln!(
                "jigor build.rs: {} is missing component archives; trying the next layout",
                lib_dir.display()
            );
            continue;
        }

        add_search_dir(&lib_dir);
        for c in COMPONENTS {
            println!("cargo:rustc-link-lib=static=onnxruntime_{c}");
        }

        if ext_dir.join("libortcustomops.a").exists() {
            add_search_dir(&ext_dir);
            println!("cargo:rustc-link-lib=static=ortcustomops");
            println!("cargo:rustc-link-lib=static=ocos_operators");
            println!("cargo:rustc-link-lib=static=noexcep_operators");
        }

        // protobuf
        let protobuf_build = transform(external.join("protobuf-build"));
        add_search_dir(&protobuf_build);
        for lib in ["protobuf-lited", "protobuf-lite", "protobuf"] {
            if protobuf_build.join(format!("lib{lib}.a")).exists() {
                println!("cargo:rustc-link-lib=static={lib}");
            }
        }

        // onnx
        add_search_dir(&transform(external.join("onnx-build")));
        println!("cargo:rustc-link-lib=static=onnx");
        println!("cargo:rustc-link-lib=static=onnx_proto");

        // nsync (not every ONNX Runtime configuration builds it)
        optional_link_lib(&transform(external.join("google_nsync-build")), "nsync_cpp");

        // cpuinfo (+ clog when built)
        add_search_dir(&transform(external.join("pytorch_cpuinfo-build")));
        if !optional_link_lib(
            &transform(
                external
                    .join("pytorch_cpuinfo-build")
                    .join("deps")
                    .join("clog"),
            ),
            "clog",
        ) {
            optional_link_lib(&transform(external.join("pytorch_clog-build")), "clog");
        }
        println!("cargo:rustc-link-lib=static=cpuinfo");

        // re2 — guaranteed present by ensure_re2() in the build script
        add_search_dir(&transform(external.join("re2-build")));
        println!("cargo:rustc-link-lib=static=re2");

        emit_absl(&external, &profile);

        // optional EPs — linked only if the ONNX Runtime build enabled them
        optional_link_lib(&lib_dir, "onnxruntime_providers_acl");
        optional_link_lib(&lib_dir, "onnxruntime_providers_armnn");
        optional_link_lib(&lib_dir, "onnxruntime_providers_azure");
        if optional_link_lib(&lib_dir, "onnxruntime_providers_coreml") {
            println!("cargo:rustc-link-lib=framework=CoreML");
            println!("cargo:rustc-link-lib=coreml_proto");
        }
        optional_link_lib(&lib_dir, "onnxruntime_providers_nnapi");
        optional_link_lib(&lib_dir, "onnxruntime_providers_qnn");
        optional_link_lib(&lib_dir, "onnxruntime_providers_rknpu");
        optional_link_lib(&lib_dir, "onnxruntime_providers_tvm");
        if optional_link_lib(&lib_dir, "onnxruntime_providers_xnnpack") {
            add_search_dir(&transform(external.join("googlexnnpack-build")));
            println!("cargo:rustc-link-lib=static=XNNPACK");
            optional_link_lib(
                &transform(external.join("googlexnnpack-build")),
                "microkernels-prod",
            );
            add_search_dir(&transform(external.join("pthreadpool-build")));
            println!("cargo:rustc-link-lib=static=pthreadpool");
        }
        // kleidiai is aarch64-only — impossible for this target

        return;
    }

    panic!(
        "jigor: no usable ONNX Runtime layout found in `{}`\n\
		 (looked for libonnxruntime_common.a directly and under Release/, plus a\n\
		 complete set of component archives). Check ORT_LIB_PATH / the output of\n\
		 build-support/build-onnxruntime-macos.sh",
        base.display()
    );
}

/// abseil archives, in ort-sys's exact link order.
fn emit_absl(external: &Path, profile: &str) {
    let t = |p: PathBuf| p.join(profile);

    add_search_dir(&t(external
        .join("abseil_cpp-build")
        .join("absl")
        .join("debugging")));
    println!("cargo:rustc-link-lib=static=absl_examine_stack");
    println!("cargo:rustc-link-lib=static=absl_debugging_internal");
    println!("cargo:rustc-link-lib=static=absl_demangle_internal");
    println!("cargo:rustc-link-lib=static=absl_demangle_rust");
    println!("cargo:rustc-link-lib=static=absl_decode_rust_punycode");
    println!("cargo:rustc-link-lib=static=absl_utf8_for_code_point");

    add_search_dir(&t(external
        .join("abseil_cpp-build")
        .join("absl")
        .join("base")));
    println!("cargo:rustc-link-lib=static=absl_base");
    println!("cargo:rustc-link-lib=static=absl_spinlock_wait");
    println!("cargo:rustc-link-lib=static=absl_malloc_internal");
    println!("cargo:rustc-link-lib=static=absl_strerror");
    println!("cargo:rustc-link-lib=static=absl_raw_logging_internal");
    println!("cargo:rustc-link-lib=static=absl_throw_delegate");

    add_search_dir(&t(external
        .join("abseil_cpp-build")
        .join("absl")
        .join("hash")));
    println!("cargo:rustc-link-lib=static=absl_hash");
    println!("cargo:rustc-link-lib=static=absl_city");
    optional_link_lib(
        &t(external.join("abseil_cpp-build").join("absl").join("hash")),
        "absl_low_level_hash",
    );

    add_search_dir(&t(external
        .join("abseil_cpp-build")
        .join("absl")
        .join("container")));
    println!("cargo:rustc-link-lib=static=absl_hashtablez_sampler");
    println!("cargo:rustc-link-lib=static=absl_raw_hash_set");

    add_search_dir(&t(external
        .join("abseil_cpp-build")
        .join("absl")
        .join("synchronization")));
    println!("cargo:rustc-link-lib=static=absl_kernel_timeout_internal");
    println!("cargo:rustc-link-lib=static=absl_graphcycles_internal");
    println!("cargo:rustc-link-lib=static=absl_synchronization");

    add_search_dir(&t(external
        .join("abseil_cpp-build")
        .join("absl")
        .join("time")));
    println!("cargo:rustc-link-lib=static=absl_time_zone");
    println!("cargo:rustc-link-lib=static=absl_time");

    add_search_dir(&t(external
        .join("abseil_cpp-build")
        .join("absl")
        .join("numeric")));
    println!("cargo:rustc-link-lib=static=absl_int128");

    add_search_dir(&t(external
        .join("abseil_cpp-build")
        .join("absl")
        .join("strings")));
    println!("cargo:rustc-link-lib=static=absl_str_format_internal");
    println!("cargo:rustc-link-lib=static=absl_strings");
    println!("cargo:rustc-link-lib=static=absl_string_view");
    println!("cargo:rustc-link-lib=static=absl_strings_internal");

    add_search_dir(&t(external
        .join("abseil_cpp-build")
        .join("absl")
        .join("debugging")));
    println!("cargo:rustc-link-lib=static=absl_symbolize");
    println!("cargo:rustc-link-lib=static=absl_stacktrace");

    let log_dir = t(external.join("abseil_cpp-build").join("absl").join("log"));
    add_search_dir(&log_dir);
    println!("cargo:rustc-link-lib=static=absl_log_globals");
    println!("cargo:rustc-link-lib=static=absl_log_internal_format");
    println!("cargo:rustc-link-lib=static=absl_log_internal_proto");
    println!("cargo:rustc-link-lib=static=absl_log_internal_globals");
    optional_link_lib(&log_dir, "absl_log_internal_check_op");
    optional_link_lib(&log_dir, "absl_log_internal_structured_proto");
    optional_link_lib(&log_dir, "absl_log_internal_nullguard");
    println!("cargo:rustc-link-lib=static=absl_log_internal_log_sink_set");
    println!("cargo:rustc-link-lib=static=absl_log_sink");
    println!("cargo:rustc-link-lib=static=absl_log_internal_message");
}

/// ort-sys's `static_link_prerequisites` for apple targets.
fn static_link_prerequisites() {
    let target = env::var("TARGET").unwrap_or_default();

    let cpp_link_stdlib = env::var("ORT_CXX_STDLIB")
        .or_else(|_| env::var("CXXSTDLIB"))
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            if target.contains("apple") {
                Some("c++".to_owned())
            } else {
                None
            }
        });
    if let Some(cpp_link_stdlib) = cpp_link_stdlib {
        println!("cargo:rustc-link-lib={cpp_link_stdlib}");
    }

    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreML");

    if target.contains("apple-darwin")
        && let Some(dir) = macos_rtlib_search_dir()
    {
        println!("cargo:rustc-link-search={dir}");
        println!("cargo:rustc-link-lib=clang_rt.osx");
    }
}

/// `<clang resource>/lib/darwin`, where libclang_rt.osx.a lives — same
/// discovery as ort-sys's `static_link::apple::macos_rtlib_search_dir`.
fn macos_rtlib_search_dir() -> Option<String> {
    let cc = env::var("CC").unwrap_or_else(|_| "clang".to_owned());
    let output = Command::new(&cc).arg("--print-search-dirs").output().ok()?;
    if !output.status.success() {
        println!(
            "cargo:warning=jigor: `{cc} --print-search-dirs` failed; not adding the clang_rt search dir"
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;
            if !path.is_empty() && Path::new(path).is_dir() {
                return Some(format!("{path}/lib/darwin"));
            }
        }
    }
    None
}
