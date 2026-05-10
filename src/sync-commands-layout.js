#!/usr/bin/env node
/**
 * Synchronize the legacy commands/ compatibility tree from skills/turing/.
 *
 * Usage:
 *   node src/sync-commands-layout.js [--check]
 */

import { mkdir, readdir, readFile, rm, writeFile } from "fs/promises";
import { dirname, join, relative } from "path";
import { fileURLToPath } from "url";
import { getCommandNames } from "./command-registry.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = join(__dirname, "..");
const SKILLS_DIR = join(PLUGIN_ROOT, "skills", "turing");
const COMMANDS_DIR = join(PLUGIN_ROOT, "commands");

async function readUtf8(path) {
  return readFile(path, "utf8");
}

async function copyTextFile(source, target) {
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, await readUtf8(source));
}

async function compatibilityEntries() {
  const names = await getCommandNames();
  return [
    {
      source: join(SKILLS_DIR, "SKILL.md"),
      target: join(COMMANDS_DIR, "turing.md"),
    },
    ...names.map((name) => ({
      source: join(SKILLS_DIR, name, "SKILL.md"),
      target: join(COMMANDS_DIR, `${name}.md`),
    })),
    {
      source: join(SKILLS_DIR, "rules", "loop-protocol.md"),
      target: join(COMMANDS_DIR, "rules", "loop-protocol.md"),
    },
  ];
}

async function existingCompatibilityEntries(dir = COMMANDS_DIR) {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") {
      return [];
    }
    throw error;
  }

  const paths = [];
  for (const entry of entries) {
    const path = join(dir, entry.name);
    paths.push(path);
    if (entry.isDirectory()) {
      paths.push(...await existingCompatibilityEntries(path));
    }
  }
  return paths;
}

async function findDrift() {
  const entries = await compatibilityEntries();
  const expectedTargets = new Set(entries.map(({ target }) => target));
  const expectedPaths = new Set([COMMANDS_DIR]);
  for (const target of expectedTargets) {
    let current = target;
    while (current.startsWith(COMMANDS_DIR)) {
      expectedPaths.add(current);
      if (current === COMMANDS_DIR) {
        break;
      }
      current = dirname(current);
    }
  }
  const issues = [];

  for (const { source, target } of entries) {
    let sourceText;
    try {
      sourceText = await readUtf8(source);
    } catch (error) {
      issues.push(`missing source ${relative(PLUGIN_ROOT, source)}: ${error.message}`);
      continue;
    }

    let targetText;
    try {
      targetText = await readUtf8(target);
    } catch (error) {
      if (error.code === "ENOENT") {
        issues.push(`missing compatibility file ${relative(PLUGIN_ROOT, target)}`);
      } else {
        issues.push(`cannot read compatibility file ${relative(PLUGIN_ROOT, target)}: ${error.message}`);
      }
      continue;
    }

    if (targetText !== sourceText) {
      issues.push(`diverged compatibility file ${relative(PLUGIN_ROOT, target)}`);
    }
  }

  for (const path of await existingCompatibilityEntries()) {
    if (!expectedPaths.has(path)) {
      issues.push(`stale compatibility path ${relative(PLUGIN_ROOT, path)}`);
    }
  }

  return issues;
}

export async function syncCommandsLayout({ check = false } = {}) {
  if (check) {
    const issues = await findDrift();
    if (issues.length > 0) {
      for (const issue of issues) {
        console.error(issue);
      }
      process.exitCode = 1;
      return;
    }
    console.log("commands compatibility tree is in sync");
    return;
  }

  await rm(COMMANDS_DIR, { recursive: true, force: true });
  for (const { source, target } of await compatibilityEntries()) {
    await copyTextFile(source, target);
  }
  console.log("commands compatibility tree synchronized");
}

const isDirectRun =
  process.argv[1] &&
  fileURLToPath(import.meta.url).endsWith(process.argv[1].replace(/^.*\//, ""));

if (isDirectRun) {
  syncCommandsLayout({ check: process.argv.includes("--check") }).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
