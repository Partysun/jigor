# Build and publish the Windows jigor binary as the npm `jigor-win32-x64-msvc`
# platform package. Called by the npm-win32-x64 step of .woodpecker/npm.yml
# (native powershell on the local Windows agent; step environments DO reach
# powershell steps).
#
# Living in a script file (not inline YAML) matters: Woodpecker rewrites
# inline step commands with its envsubst (${...} gets interpolated and can
# even make the config fail to parse — e.g. `$LASTEXITCODE` in a one-liner),
# while script files reach the shell untouched.
$ErrorActionPreference = "Stop"

cargo build -p jigor-cli --release
if (-not $?) { throw "cargo build failed" }

Copy-Item target\release\jigor.exe npm\platforms\win32-x64-msvc\jigor.exe
$smoke = & ".\npm\platforms\win32-x64-msvc\jigor.exe" models
if (-not $?) { throw "jigor.exe smoke failed" }

npm config set "//registry.npmjs.org/:_authToken" $env:NPM_TOKEN
if (-not $?) { throw "npm config set failed" }

node scripts\npm-version.js npm/platforms/win32-x64-msvc
if (-not $?) { throw "npm-version.js failed" }

Set-Location npm\platforms\win32-x64-msvc
npm publish --access public
exit $LASTEXITCODE