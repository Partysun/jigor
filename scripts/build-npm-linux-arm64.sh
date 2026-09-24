#!/usr/bin/env bash
# Build and publish the arm64 linux jigor binary as the npm
# `jigor-linux-arm64-gnu` platform package. Called by the build-arm64
# step of .woodpecker/npm-linux.yml.
#
# Living in a script file (not inline YAML) matters: Woodpecker rewrites
# inline step commands with its envsubst and runs them under /bin/sh
# (dash on Debian), which chokes on `-` in export variable names like
# CXX_aarch64-unknown-linux-gnu; script files run under bash untouched.
set -euo pipefail

apt-get update && apt-get install -y nodejs npm gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
rustup target add aarch64-unknown-linux-gnu
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
# `-` in a variable name is rejected by both dash and bash `export`; only
# the env utility can set CXX_aarch64-unknown-linux-gnu (the dashless
# spelling just above covers the underscored lookup cargo also accepts).
env "CXX_aarch64-unknown-linux-gnu=aarch64-linux-gnu-g++" cargo build -p jigor-cli --release --target aarch64-unknown-linux-gnu
cp target/release/jigor npm/platforms/linux-arm64-gnu/
node scripts/npm-version.js npm/platforms/linux-arm64-gnu
echo "//registry.npmjs.org/:_authToken=$NPM_TOKEN" >> ~/.npmrc
node scripts/npm-publish.js npm/platforms/linux-arm64-gnu