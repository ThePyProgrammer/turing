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

program.parse();
