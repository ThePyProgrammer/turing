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
import { getTargetPaths } from "./paths.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = join(__dirname, "..");

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
  "explore/SKILL.md",
  "design/SKILL.md",
  "logbook/SKILL.md",
  "poster/SKILL.md",
  "report/SKILL.md",
  "mode/SKILL.md",
  "preflight/SKILL.md",
  "card/SKILL.md",
  "seed/SKILL.md",
  "reproduce/SKILL.md",
  "diagnose/SKILL.md",
  "ablate/SKILL.md",
  "frontier/SKILL.md",
  "profile/SKILL.md",
  "checkpoint/SKILL.md",
  "export/SKILL.md",
  "lit/SKILL.md",
  "paper/SKILL.md",
  "queue/SKILL.md",
  "retry/SKILL.md",
  "fork/SKILL.md",
  "diff/SKILL.md",
  "watch/SKILL.md",
  "regress/SKILL.md",
  "ensemble/SKILL.md",
  "stitch/SKILL.md",
  "warm/SKILL.md",
  "scale/SKILL.md",
  "budget/SKILL.md",
  "distill/SKILL.md",
  "transfer/SKILL.md",
  "audit/SKILL.md",
  "sanity/SKILL.md",
  "baseline/SKILL.md",
  "leak/SKILL.md",
  "xray/SKILL.md",
  "sensitivity/SKILL.md",
  "calibrate/SKILL.md",
  "feature/SKILL.md",
  "curriculum/SKILL.md",
  "prune/SKILL.md",
  "quantize/SKILL.md",
  "merge/SKILL.md",
  "surgery/SKILL.md",
  "trend/SKILL.md",
  "flashback/SKILL.md",
  "archive/SKILL.md",
  "annotate/SKILL.md",
  "search/SKILL.md",
  "template/SKILL.md",
  "replay/SKILL.md",
  "cite/SKILL.md",
  "present/SKILL.md",
  "changelog/SKILL.md",
  "onboard/SKILL.md",
  "share/SKILL.md",
  "review/SKILL.md",
  "whatif/SKILL.md",
  "counterfactual/SKILL.md",
  "simulate/SKILL.md",
  "update/SKILL.md",
  "registry/SKILL.md",
  "postmortem/SKILL.md",
  "doctor/SKILL.md",
  "plan/SKILL.md",
];

const EXPECTED_AGENTS = ["ml-researcher.md", "ml-evaluator.md"];

const EXPECTED_CONFIG = [
  "defaults.yaml", "lifecycle.toml", "taxonomy.toml",
  "experiment_archetypes.yaml", "novelty_aliases.yaml",
  "relationships.toml", "state.toml", "task_taxonomy.yaml",
  "failure_modes.yaml",
  "watch_alerts.yaml",
];

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
