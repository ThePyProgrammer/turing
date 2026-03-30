#!/usr/bin/env node
/**
 * Turing installation verifier.
 *
 * Checks that all expected files are in place and reports status.
 *
 * Usage:
 *   node src/verify.js [--global]
 */

import { existsSync } from "fs";
import { join } from "path";

const isGlobal = process.argv.includes("--global");
const homeDir = process.env.HOME || process.env.USERPROFILE;
const targetBase = isGlobal ? join(homeDir, ".claude") : ".claude";

const EXPECTED_COMMANDS = [
  "turing.md",
  "init.md",
  "train.md",
  "status.md",
  "compare.md",
  "sweep.md",
];

const EXPECTED_AGENTS = ["ml-researcher.md", "ml-evaluator.md"];

const EXPECTED_CONFIG = ["defaults.yaml"];

let passed = 0;
let failed = 0;

function check(label, path) {
  const exists = existsSync(path);
  const status = exists ? "\x1b[32m✓\x1b[0m" : "\x1b[31m✗\x1b[0m";
  console.log(`  ${status} ${label}`);
  if (exists) passed++;
  else failed++;
}

console.log("Turing Installation Verification");
console.log(`Checking: ${targetBase}`);
console.log("");

console.log("Commands:");
for (const cmd of EXPECTED_COMMANDS) {
  check(cmd, join(targetBase, "commands", "turing", cmd));
}

console.log("\nAgents:");
for (const agent of EXPECTED_AGENTS) {
  check(agent, join(targetBase, "agents", "turing", agent));
}

console.log("\nConfig:");
for (const cfg of EXPECTED_CONFIG) {
  check(cfg, join(targetBase, "config", "turing", cfg));
}

console.log("");
console.log(`${passed} passed, ${failed} failed`);

if (failed > 0) {
  console.log("\nRun 'node src/install.js' to fix missing files.");
  process.exit(1);
}
