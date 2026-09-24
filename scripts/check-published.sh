#!/usr/bin/env bash
# Check that a version is live on every registry: crates.io (jigor,
# jigor-cli), PyPI (jigor sdist + wheels), and npm (the meta package and
# the four platform packages). Exits non-zero when anything is missing
# or a registry cannot be reached.
#
# Usage: scripts/check-published.sh [version]
# (defaults to the workspace version in Cargo.toml)
set -euo pipefail

cd "$(dirname "$0")/.."

version="${1:-}"
if [ -z "$version" ]; then
  version=$(sed -n 's/^version = "\([^"]*\)".*/\1/p' Cargo.toml | head -n1)
fi
if [ -z "$version" ]; then
  echo "cannot read version from Cargo.toml" >&2
  exit 2
fi

checked=0
missing=0

check() { # <label> <url>
  checked=$((checked + 1))
  code=$(curl -sS -o /dev/null -w '%{http_code}' -L --max-time 30 \
    -A 'jigor-check-published (https://github.com/Partysun/jigor)' "$2") || code=000
  if [ "$code" = 200 ]; then
    printf '  ok        %s\n' "$1"
  elif [ "$code" = 404 ]; then
    printf '  MISSING   %s\n' "$1"
    missing=$((missing + 1))
  else
    printf '  ERROR(%s) %s\n' "$code" "$1"
    missing=$((missing + 1))
  fi
}

echo "checking version $version"

check "jigor $version (crates.io)" "https://crates.io/api/v1/crates/jigor/$version"
check "jigor-cli $version (crates.io)" "https://crates.io/api/v1/crates/jigor-cli/$version"
check "jigor $version (PyPI)" "https://pypi.org/pypi/jigor/$version/json"
check "@zatsepin/jigor $version (npm)" "https://registry.npmjs.org/@zatsepin%2fjigor/$version"
check "@zatsepin/jigor-darwin $version (npm)" "https://registry.npmjs.org/@zatsepin%2fjigor-darwin/$version"
check "@zatsepin/jigor-linux-x64-gnu $version (npm)" "https://registry.npmjs.org/@zatsepin%2fjigor-linux-x64-gnu/$version"
check "@zatsepin/jigor-linux-arm64-gnu $version (npm)" "https://registry.npmjs.org/@zatsepin%2fjigor-linux-arm64-gnu/$version"
check "@zatsepin/jigor-win32-x64-msvc $version (npm)" "https://registry.npmjs.org/@zatsepin%2fjigor-win32-x64-msvc/$version"

echo
if [ "$missing" -eq 0 ]; then
  echo "published: all $checked checks passed for $version"
else
  echo "published: $missing of $checked checks failed for $version" >&2
  exit 1
fi
