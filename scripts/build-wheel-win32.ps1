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

# PyPI rejects re-uploads ("File already exists"): skip files already on
# the registry so re-runs of the same version are a no-op (mirrors
# scripts/pypi-publish.sh).
$version = $null
foreach ($line in (Get-Content Cargo.toml)) {
    if ($line -match '^\s*version = "([^"]+)"') { $version = $matches[1]; break }
}
$artifacts = @(Get-ChildItem dist\* -File 2>$null)
if ($artifacts.Count -eq 0) { throw "no artifacts in dist\" }
$missing = @()
foreach ($a in $artifacts) {
    $found = $false
    try {
        $json = Invoke-RestMethod -Uri "https://pypi.org/pypi/jigor/$version/json" -TimeoutSec 20
        foreach ($u in $json.urls) {
            if ($u.filename -eq $a.Name) { $found = $true; break }
        }
    } catch { }
    if ($found) { echo "skip: $($a.Name) already on PyPI" } else { $missing += $a.Path }
}
if ($missing.Count -eq 0) {
    echo "nothing new to publish"
    exit 0
}
echo "publishing $($missing.Count) file(s)"
uv publish @missing
exit $LASTEXITCODE