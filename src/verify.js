#!/usr/bin/env node
/**
 * Turing installation verifier.
 *
 * Checks that all expected files are in place and reports status.
 *
 * Usage:
 *   node src/verify.js [--scope global|project]
 */

import { access, readdir } from "fs/promises";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { getConfigFiles, getExpectedCommandPaths } from "./command-registry.js";
import { getTargetPaths } from "./paths.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = join(__dirname, "..");

const EXPECTED_AGENTS = ["ml-researcher.md", "ml-evaluator.md"];

async function templateFiles(root, relativeDir = "templates") {
  const dir = join(root, relativeDir);
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    if (entry.name === "__pycache__" || entry.name === ".pytest_cache") {
      continue;
    }

    const relativePath = `${relativeDir}/${entry.name}`;
    if (entry.isDirectory()) {
      files.push(...await templateFiles(root, relativePath));
    } else if (!entry.name.endsWith(".pyc")) {
      files.push(relativePath);
    }
  }

  return files;
}

async function fileExists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

export async function verify(opts = {}) {
  const expectedCommands = await getExpectedCommandPaths();
  const expectedConfig = await getConfigFiles();
  const scopes = opts.scope ? [opts.scope] : ["global", "project"];
  const expectedTemplates = await templateFiles(PLUGIN_ROOT);
  let found = false;
  let totalMissing = 0;

  for (const scope of scopes) {
    const paths = getTargetPaths(scope);
    const exists = await fileExists(join(paths.commands, "SKILL.md"));
    if (!exists) continue;
    found = true;

    console.log(`\n✓ turing found (${scope}): ${paths.commands}\n`);

    let missing = 0;

    console.log("Commands:");
    for (const cmd of expectedCommands) {
      const ok = await fileExists(join(paths.commands, cmd));
      console.log(`  ${ok ? "✓" : "✗"} commands/${cmd}`);
      if (!ok) missing++;
    }

    console.log("\nAgents:");
    for (const agent of EXPECTED_AGENTS) {
      const ok = await fileExists(join(paths.agents, agent));
      console.log(`  ${ok ? "✓" : "✗"} agents/${agent}`);
      if (!ok) missing++;
    }

    console.log("\nConfig:");
    for (const cfg of expectedConfig) {
      const ok = await fileExists(join(paths.config, cfg));
      console.log(`  ${ok ? "✓" : "✗"} config/${cfg}`);
      if (!ok) missing++;
    }

    console.log("\nTemplates:");
    for (const template of expectedTemplates) {
      const ok = await fileExists(join(paths.commands, template));
      console.log(`  ${ok ? "✓" : "✗"} commands/${template}`);
      if (!ok) missing++;
    }

    // Check CLAUDE.md
    const claudeOk = await fileExists(paths.claudeMd);
    console.log(`\n  ${claudeOk ? "✓" : "✗"} CLAUDE.md`);

    totalMissing += missing;
    console.log(
      `\n  ${missing === 0 ? "✓ Installation complete" : `✗ ${missing} files missing — run claude-turing install`}\n`,
    );
  }

  if (!found) {
    console.log("\n✗ turing not found. Run: claude-turing install\n");
    totalMissing++;
  }

  if (totalMissing > 0) {
    process.exitCode = 1;
  }
}

// Direct execution
const isDirectRun =
  process.argv[1] &&
  import.meta.url.endsWith(process.argv[1].replace(/^.*\//, ""));
if (isDirectRun) {
  const scopeIdx = process.argv.indexOf("--scope");
  verify({
    scope: scopeIdx !== -1 ? process.argv[scopeIdx + 1] : undefined,
  });
}
