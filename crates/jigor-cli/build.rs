//! Align the linked `jigor` binary with the deployment target of the
//! ONNX Runtime statics (built with `CMAKE_OSX_DEPLOYMENT_TARGET=11.0`,
//! same as `scripts/ci-macos-build.sh`) — rustc's x86_64-apple-darwin
//! default (10.12) would advertise support the bundled ORT does not have.
//! An explicit user `MACOSX_DEPLOYMENT_TARGET` always wins.

use std::env;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    println!("cargo::rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    if env::var_os("MACOSX_DEPLOYMENT_TARGET").is_none() {
        println!("cargo::rustc-env=MACOSX_DEPLOYMENT_TARGET=11.0");
    }
}
