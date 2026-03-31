#!/usr/bin/env node
/**
 * Turing installation verifier.
 *
 * Checks that all expected files are in place and reports status.
 *
 * Usage:
 *   node src/verify.js [--scope global|project]
 */

import { access } from "fs/promises";
import { join } from "path";
import { getTargetPaths } from "./paths.js";

const EXPECTED_COMMANDS = [
  "SKILL.md",
  "init/SKILL.md",
  "train/SKILL.md",
  "status/SKILL.md",
  "compare/SKILL.md",
  "sweep/SKILL.md",
  "validate/SKILL.md",
  "try/SKILL.md",
  "brief/SKILL.md",
  "suggest/SKILL.md",
  "design/SKILL.md",
  "logbook/SKILL.md",
  "poster/SKILL.md",
  "report/SKILL.md",
  "mode/SKILL.md",
  "preflight/SKILL.md",
  "card/SKILL.md",
];

const EXPECTED_AGENTS = ["ml-researcher.md", "ml-evaluator.md"];

const EXPECTED_CONFIG = [
  "defaults.yaml", "lifecycle.toml", "taxonomy.toml",
  "experiment_archetypes.yaml", "novelty_aliases.yaml",
  "relationships.toml", "state.toml", "task_taxonomy.yaml",
];

async function fileExists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

export async function verify(opts = {}) {
  const scopes = opts.scope ? [opts.scope] : ["global", "project"];
  let found = false;

  for (const scope of scopes) {
    const paths = getTargetPaths(scope);
    const exists = await fileExists(join(paths.commands, "SKILL.md"));
    if (!exists) continue;
    found = true;

    console.log(`\n✓ turing found (${scope}): ${paths.commands}\n`);

    let missing = 0;

    console.log("Commands:");
    for (const cmd of EXPECTED_COMMANDS) {
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
    for (const cfg of EXPECTED_CONFIG) {
      const ok = await fileExists(join(paths.config, cfg));
      console.log(`  ${ok ? "✓" : "✗"} config/${cfg}`);
      if (!ok) missing++;
    }

    // Check CLAUDE.md
    const claudeOk = await fileExists(paths.claudeMd);
    console.log(`\n  ${claudeOk ? "✓" : "✗"} CLAUDE.md`);

    console.log(
      `\n  ${missing === 0 ? "✓ Installation complete" : `✗ ${missing} files missing — run claude-turing install`}\n`,
    );
  }

  if (!found) {
    console.log("\n✗ turing not found. Run: claude-turing install\n");
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
