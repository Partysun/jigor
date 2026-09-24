// Pin npm package versions to the release tag (CI_COMMIT_TAG=v0.1.0 ->
// 0.1.0), falling back to the workspace version in Cargo.toml (deployment
// task runs without a tag). When a meta package is in `dir`, its
// optionalDependencies are synced to the same version.
//
// Usage: node scripts/npm-version.js <dir>...
const fs = require("fs");
const path = require("path");

function workspaceVersion() {
  const cargo = fs.readFileSync(path.join(__dirname, "..", "Cargo.toml"), "utf8");
  const m = cargo.match(/^\s*version = "([^"]+)"/m);
  return m ? m[1] : null;
}

let version = process.env.CI_COMMIT_TAG;
if (version && version.startsWith("v")) version = version.slice(1);
if (!version) version = workspaceVersion();
if (!version) {
  console.error("cannot determine version (CI_COMMIT_TAG or Cargo.toml)");
  process.exit(1);
}

for (const dir of process.argv.slice(2)) {
  const pkgPath = path.join(dir, "package.json");
  const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
  pkg.version = version;
  if (pkg.optionalDependencies) {
    for (const name of Object.keys(pkg.optionalDependencies)) {
      pkg.optionalDependencies[name] = version;
    }
  }
  fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n");
  console.log(`${dir} -> ${version}`);
}