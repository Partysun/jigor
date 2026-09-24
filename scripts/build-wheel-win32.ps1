# Build and publish the Windows jigor wheel. Called by the publish-windows
# step of .woodpecker/wheel-windows.yml (native powershell on the local
# Windows agent; step environments DO reach powershell steps).
#
# Living in a script file (not inline YAML) matters: Woodpecker rewrites
# inline step commands with its envsubst (${...} gets interpolated and can
# even make the config fail to parse — e.g. `$env:` in a one-liner),
# while script files reach the shell untouched.
$ErrorActionPreference = "Stop"

# ort-sys panics when it can't resolve its prebuilt-binaries cache dir on
# Windows (SHGetKnownFolderPath returns nothing inside the agent session —
# same fix as govor's ci-windows-build.sh); point it at a writable dir.
if (-not $env:ORT_CACHE_DIR) {
    if ($env:LOCALAPPDATA) {
        $env:ORT_CACHE_DIR = Join-Path $env:LOCALAPPDATA "jigor-ci\ort-pyke"
    } else {
        $env:ORT_CACHE_DIR = Join-Path $env:TEMP "jigor-ci\ort-pyke"
    }
}

# guard against an agent rustup with no default toolchain configured
rustup default stable
if (-not $?) { throw "rustup default stable failed" }

cargo install --quiet uv
if (-not $?) { throw "cargo install uv failed" }

uv tool install --quiet maturin
if (-not $?) { throw "uv tool install maturin failed" }

uv tool run maturin build --release --out dist
if (-not $?) { throw "maturin build failed" }

uv publish dist/*
exit $LASTEXITCODE