#!/usr/bin/env bash
# Publish dist/* artifacts to PyPI, skipping files that already exist —
# PyPI rejects re-uploads with "File already exists", so re-runs of the
# same version must be a no-op (mirrors scripts/npm-publish.js for npm).
#
# Usage: bash scripts/pypi-publish.sh   (run from the workspace root)
set -euo pipefail

PROJECT="jigor" # PyPI project name (static publish target)
VERSION="$(sed -n 's/^[[:space:]]*version = "\([^"]*\)"/\1/p' Cargo.toml | head -n 1)"
[ -n "$VERSION" ] || { echo "cannot read version from Cargo.toml" >&2; exit 1; }

shopt -s nullglob
artifacts=(dist/*)
shopt -u nullglob
[ "${#artifacts[@]}" -gt 0 ] || { echo "no artifacts in dist/" >&2; exit 1; }

missing=()
for artifact in "${artifacts[@]}"; do
  f="$(basename "$artifact")"
  # exact quoted-name match: longer similar wheel names must not mask a
  # missing file (e.g. ..._universal2.whl vs ..._x86_64._arm64__universal2)
  if curl -fsS "https://pypi.org/pypi/$PROJECT/$VERSION/json" 2>/dev/null | grep -qF "\"$f\""; then
    echo "skip: $f already on PyPI"
  else
    missing+=("$artifact")
  fi
done

if [ "${#missing[@]}" -eq 0 ]; then
  echo "nothing new to publish"
  exit 0
fi
echo "publishing ${#missing[@]} file(s): ${missing[*]}"
uv publish "${missing[@]}"