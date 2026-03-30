#!/usr/bin/env node
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { Command } = require("commander");
const pkg = require("../package.json");

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
    const { execSync } = await import("child_process");
    const { dirname, join } = await import("path");
    const { fileURLToPath } = await import("url");
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const script = join(__dirname, "turing-init.sh");
    const args = [name, dir].filter(Boolean).join(" ");
    execSync(`bash "${script}" ${args}`, { stdio: "inherit" });
  });

program.parse();
