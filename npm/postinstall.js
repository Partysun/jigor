// Port of ast-grep's @ast-grep/cli postinstall: locate the native `jigor`
// binary for the current platform (a platform package installed as an
// optional dependency, or a local `target/release|debug/jigor` dev build)
// and hard-link it over the JS shims.
const fs = require("fs");
const path = require("path");

const binaryName = process.platform === "win32" ? "jigor.exe" : "jigor";

function detectPackageName() {
  const { platform, arch } = process;
  switch (platform) {
    case "darwin":
      if (arch === "arm64") return "jigor-darwin-arm64";
      if (arch === "x64") return "jigor-darwin-x64";
      break;
    case "linux": {
      const { MUSL, familySync } = require("detect-libc");
      if (familySync() === MUSL) return null;
      if (arch === "arm64") return "jigor-linux-arm64-gnu";
      if (arch === "x64") return "jigor-linux-x64-gnu";
      break;
    }
    case "win32":
      if (arch === "x64") return "jigor-win32-x64-msvc";
      break;
  }
  return null;
}

function resolveBinaryDir() {
  const pkgName = detectPackageName();
  if (pkgName) {
    try {
      const dir = path.dirname(
        require.resolve(`${pkgName}/package.json`, { paths: [__dirname] }),
      );
      if (fs.existsSync(path.join(dir, binaryName))) return dir;
    } catch (_) {
      // fall through to local dev paths
    }
  }
  for (const profile of ["release", "debug"]) {
    const dir = path.join(__dirname, "..", "target", profile);
    if (fs.existsSync(path.join(dir, binaryName))) return dir;
  }
  return null;
}

function resolveBinaryPath() {
  const dir = resolveBinaryDir();
  return dir ? path.join(dir, binaryName) : null;
}

function installBinary(src, dest) {
  try {
    fs.linkSync(src, dest);
  } catch (_) {
    fs.copyFileSync(src, dest);
  }
}

function main() {
  const sourceDir = resolveBinaryDir();
  if (!sourceDir) {
    console.error(
      "[jigor] No native binary found for " +
        `${process.platform ?? "?"}/${process.arch ?? "?"}. ` +
        "Supported: linux x64/arm64 (glibc), macos x64/arm64, windows x64. " +
        "Musl and other platforms are not packaged yet.",
    );
    process.exit(1);
  }

  const src = path.join(sourceDir, binaryName);
  const destBin = path.join(__dirname, binaryName);
  try {
    installBinary(src, destBin);
  } catch (_) {
    console.error("[jigor] Failed to move the native binary into place.");
    process.exit(1);
  }
}

module.exports = {
  binaryName,
  detectPackageName,
  resolveBinaryDir,
  resolveBinaryPath,
};

if (require.main === module) {
  main();
}