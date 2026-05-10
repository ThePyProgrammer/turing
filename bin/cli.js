#!/usr/bin/env node
import { createRequire } from "module";
import { realpathSync } from "fs";
import { fileURLToPath } from "url";
const require = createRequire(import.meta.url);
const { Command } = require("commander");
const pkg = require("../package.json");

export function buildInitArgs(name, dir) {
  return [name, dir].filter(Boolean);
}

function isDirectRun() {
  if (!process.argv[1]) return false;

  try {
    return realpathSync(fileURLToPath(import.meta.url)) === realpathSync(process.argv[1]);
  } catch {
    return false;
  }
}

const program = new Command();

program
  .name("claude-turing")
  .description(pkg.description)
  .version(pkg.version);

program
  .command("install")
  .description("Install turing commands and agents to Claude Code")
  .option("--global", "Install globally (~/.claude/)")
  .option("--project", "Install for current project (.claude/)")
  .action(async (opts) => {
    const { install } = await import("../src/install.js");
    await install(opts);
  });

program
  .command("verify")
  .description("Verify turing installation is complete")
  .option("--scope <scope>", "Check a specific scope (global|project)")
  .action(async (opts) => {
    const { verify } = await import("../src/verify.js");
    await verify(opts);
  });

program
  .command("init [name] [dir]")
  .description("Scaffold ML project (CLI mode, non-Claude-Code usage)")
  .action(async (name, dir) => {
    const { spawnSync } = await import("child_process");
    const { dirname, join } = await import("path");
    const { fileURLToPath } = await import("url");
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const script = join(__dirname, "turing-init.sh");
    const args = buildInitArgs(name, dir);
    const result = spawnSync("bash", [script, ...args], { stdio: "inherit" });
    process.exit(result.status ?? 1);
  });

if (isDirectRun()) {
  program.parse();
}
