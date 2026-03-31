#!/usr/bin/env node
/**
 * Turing installer.
 *
 * Deploys commands, agents, and config to the Claude Code plugin directory.
 * Optionally inserts a managed section into the project's CLAUDE.md.
 *
 * Usage:
 *   node src/install.js [--global] [--project]
 */

import { readdir, copyFile, mkdir } from "fs/promises";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { getTargetPaths } from "./paths.js";
import { updateClaudeMd } from "./claude-md.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = join(__dirname, "..");

// Single source of truth for sub-commands (DRY — used for dirs and file copy)
const SUB_COMMANDS = [
  "init", "train", "status", "compare", "sweep", "validate",
  "try", "brief", "suggest", "design", "logbook", "poster",
  "report", "mode", "preflight",
];

export async function install(opts = {}) {
  const scope = opts.global ? "global" : opts.project ? "project" : "global";
  const paths = getTargetPaths(scope);

  console.log("Turing ML Research Harness — Installer");
  console.log(`Target: ${paths.commands} (${scope})`);
  console.log("");

  // Create directories for each sub-command + agents + config
  for (const subDir of ["", "agents", "config", "rules", ...SUB_COMMANDS]) {
    await mkdir(join(paths.commands, subDir), { recursive: true });
  }

  // Copy root command (router) as SKILL.md
  await copyFile(
    join(PLUGIN_ROOT, "commands", "turing.md"),
    join(paths.commands, "SKILL.md"),
  );
  console.log("  Router -> SKILL.md");

  // Copy sub-commands as <name>/SKILL.md
  for (const cmd of SUB_COMMANDS) {
    await copyFile(
      join(PLUGIN_ROOT, "commands", `${cmd}.md`),
      join(paths.commands, cmd, "SKILL.md"),
    );
  }
  console.log(`  ${SUB_COMMANDS.length} commands installed`);

  // Copy rules
  await copyFile(
    join(PLUGIN_ROOT, "commands", "rules", "loop-protocol.md"),
    join(paths.commands, "rules", "loop-protocol.md"),
  );
  console.log("  Rules installed");

  // Copy agents
  const agentFiles = await readdir(join(PLUGIN_ROOT, "agents"));
  for (const file of agentFiles) {
    await copyFile(
      join(PLUGIN_ROOT, "agents", file),
      join(paths.agents, file),
    );
  }
  console.log(`  ${agentFiles.length} agents installed`);

  // Copy config (static schema files only)
  const CONFIG_FILES = [
    "defaults.yaml", "lifecycle.toml", "taxonomy.toml",
    "experiment_archetypes.yaml", "novelty_aliases.yaml",
    "relationships.toml", "state.toml", "task_taxonomy.yaml",
  ];
  for (const file of CONFIG_FILES) {
    await copyFile(
      join(PLUGIN_ROOT, "config", file),
      join(paths.config, file),
    );
  }
  console.log(`  ${CONFIG_FILES.length} config files installed`);

  // Update CLAUDE.md
  await updateClaudeMd(paths.claudeMd);
  console.log("  CLAUDE.md updated");

  console.log("");
  console.log(
    `Installation complete. Run /turing:init to scaffold an ML project.`,
  );
}

// Direct execution
const isDirectRun =
  process.argv[1] &&
  fileURLToPath(import.meta.url).endsWith(process.argv[1].replace(/^.*\//, ""));
if (isDirectRun) {
  install({
    global: process.argv.includes("--global"),
    project: process.argv.includes("--project"),
  });
}
