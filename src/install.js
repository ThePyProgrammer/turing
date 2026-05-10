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

import { readdir, copyFile, mkdir, cp } from "fs/promises";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { getTargetPaths } from "./paths.js";
import { updateClaudeMd } from "./claude-md.js";
import { getCommandNames, getConfigFiles } from "./command-registry.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = join(__dirname, "..");


export async function install(opts = {}) {
  const scope = opts.global ? "global" : opts.project ? "project" : "global";
  const paths = getTargetPaths(scope);
  const subCommands = await getCommandNames();
  const configFiles = await getConfigFiles();

  console.log("Turing ML Research Harness — Installer");
  console.log(`Target: ${paths.commands} (${scope})`);
  console.log("");

  // Create directories for each sub-command + agents + config
  for (const subDir of ["", "agents", "config", "rules", "templates", ...subCommands]) {
    await mkdir(join(paths.commands, subDir), { recursive: true });
  }

  // Copy root command (router) as SKILL.md
  await copyFile(
    join(PLUGIN_ROOT, "commands", "turing.md"),
    join(paths.commands, "SKILL.md"),
  );
  console.log("  Router -> SKILL.md");

  // Copy sub-commands as <name>/SKILL.md
  for (const cmd of subCommands) {
    await copyFile(
      join(PLUGIN_ROOT, "commands", `${cmd}.md`),
      join(paths.commands, cmd, "SKILL.md"),
    );
  }
  console.log(`  ${subCommands.length} commands installed`);

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
  for (const file of configFiles) {
    await copyFile(
      join(PLUGIN_ROOT, "config", file),
      join(paths.config, file),
    );
  }
  console.log(`  ${configFiles.length} config files installed`);

  // Copy templates used by /turing:init
  await cp(
    join(PLUGIN_ROOT, "templates"),
    join(paths.commands, "templates"),
    {
      recursive: true,
      force: true,
      filter: (src) =>
        !src.includes("__pycache__") &&
        !src.includes(".pytest_cache") &&
        !src.endsWith(".pyc"),
    },
  );
  console.log("  Templates installed");

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
