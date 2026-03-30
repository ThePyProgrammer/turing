#!/usr/bin/env node
/**
 * Turing installer.
 *
 * Deploys commands, agents, and config to the Claude Code plugin directory.
 * Optionally inserts a managed section into the project's CLAUDE.md.
 *
 * Usage:
 *   node src/install.js [--global]
 *
 * --global: Install to ~/.claude/ (user-wide) instead of ./.claude/ (project-local)
 */

import { existsSync, mkdirSync, cpSync, readFileSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PLUGIN_ROOT = join(__dirname, "..");

const isGlobal = process.argv.includes("--global");
const homeDir = process.env.HOME || process.env.USERPROFILE;
const targetBase = isGlobal ? join(homeDir, ".claude") : ".claude";

const MANAGED_HEADER = "<!-- turing:managed-start -->";
const MANAGED_FOOTER = "<!-- turing:managed-end -->";

function copyDir(src, dest) {
  if (!existsSync(src)) return;
  mkdirSync(dest, { recursive: true });
  cpSync(src, dest, { recursive: true });
}

function generateClaudeMdSection() {
  return `${MANAGED_HEADER}
## Turing ML Research Harness

| Command | Purpose |
|---------|---------|
| \`/turing\` | Router — detects ML intent and routes to sub-commands |
| \`/turing:init\` | Scaffold a new ML project with autoresearch harness |
| \`/turing:train [N]\` | Run autonomous experiment loop (optional max iterations) |
| \`/turing:status\` | Show experiment status, best model, convergence state |
| \`/turing:compare <a> <b>\` | Side-by-side experiment comparison |
| \`/turing:sweep\` | Generate and run hyperparameter sweep |

| Agent | Purpose |
|-------|---------|
| \`@ml-researcher\` | Autonomous training agent (Read/Write/Edit/Bash) |
| \`@ml-evaluator\` | Read-only analysis agent (Read/Bash only) |
${MANAGED_FOOTER}`;
}

function updateClaudeMd(claudeMdPath) {
  const section = generateClaudeMdSection();

  if (!existsSync(claudeMdPath)) {
    writeFileSync(claudeMdPath, section + "\n");
    console.log(`  Created ${claudeMdPath}`);
    return;
  }

  let content = readFileSync(claudeMdPath, "utf-8");
  const startIdx = content.indexOf(MANAGED_HEADER);
  const endIdx = content.indexOf(MANAGED_FOOTER);

  if (startIdx !== -1 && endIdx !== -1) {
    content =
      content.slice(0, startIdx) +
      section +
      content.slice(endIdx + MANAGED_FOOTER.length);
  } else {
    content = content.trimEnd() + "\n\n" + section + "\n";
  }

  writeFileSync(claudeMdPath, content);
  console.log(`  Updated ${claudeMdPath}`);
}

function install() {
  console.log("Turing ML Research Harness — Installer");
  console.log(`Target: ${targetBase} (${isGlobal ? "global" : "local"})`);
  console.log("");

  // Copy commands
  const commandsSrc = join(PLUGIN_ROOT, "commands");
  const commandsDest = join(targetBase, "commands", "turing");
  copyDir(commandsSrc, commandsDest);
  console.log(`  Commands -> ${commandsDest}`);

  // Copy agents
  const agentsSrc = join(PLUGIN_ROOT, "agents");
  const agentsDest = join(targetBase, "agents", "turing");
  copyDir(agentsSrc, agentsDest);
  console.log(`  Agents   -> ${agentsDest}`);

  // Copy config
  const configSrc = join(PLUGIN_ROOT, "config");
  const configDest = join(targetBase, "config", "turing");
  copyDir(configSrc, configDest);
  console.log(`  Config   -> ${configDest}`);

  // Update CLAUDE.md
  const claudeMdPath = isGlobal
    ? join(homeDir, ".claude", "CLAUDE.md")
    : "CLAUDE.md";
  updateClaudeMd(claudeMdPath);

  console.log("");
  console.log("Installation complete. Run /turing:init to scaffold an ML project.");
}

install();
