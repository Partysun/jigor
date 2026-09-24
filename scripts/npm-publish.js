// Publish an npm package dir idempotently: skip when <name>@<version>
// already exists on the registry (tag/deployment re-runs must not fail
// with npm's 403 "cannot publish over previously published versions"),
// publish otherwise. Runs after scripts/npm-version.js pinned the version
// in <dir>/package.json; invoked from the workspace root by every
// platform pipeline (bash on linux/macOS, PowerShell on Windows).
//
// Usage: node scripts/npm-publish.js <dir>
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

async function main() {
  const dir = process.argv[2];
  if (!dir) {
    console.error("usage: node scripts/npm-publish.js <dir>");
    process.exit(1);
  }
  const pkg = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8"));
  const tag = `${pkg.name}@${pkg.version}`;

  let res;
  try {
    res = await fetch(`https://registry.npmjs.org/${pkg.name}/${pkg.version}`, { redirect: "manual" });
  } catch (e) {
    console.error(`cannot reach registry for ${tag}: ${e.message}`);
    process.exit(1);
  }
  if (res.status === 200) {
    console.log(`skip: ${tag} already published`);
    process.exit(0);
  }
  if (res.status !== 404) {
    console.error(`unexpected registry response ${res.status} for ${tag}`);
    process.exit(1);
  }

  console.log(`publishing ${tag}`);
  execSync("npm publish --access public", { cwd: dir, stdio: "inherit" });
}

main();