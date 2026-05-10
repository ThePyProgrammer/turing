#!/usr/bin/env node
/**
 * Synchronize the modern skills/turing package mirror from commands/.
 *
 * Usage:
 *   node src/sync-skills-layout.js [--check]
 */

import { mkdir, readdir, readFile, rm, writeFile } from "fs/promises";
import { dirname, join, relative } from "path";
import { fileURLToPath } from "url";
import { getCommandNames } from "./command-registry.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = join(__dirname, "..");
const COMMANDS_DIR = join(PLUGIN_ROOT, "commands");
const SKILLS_DIR = join(PLUGIN_ROOT, "skills", "turing");

async function readUtf8(path) {
  return readFile(path, "utf8");
}

async function copyTextFile(source, target) {
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, await readUtf8(source));
}

async function mirrorEntries() {
  const names = await getCommandNames();
  return [
    {
      source: join(COMMANDS_DIR, "turing.md"),
      target: join(SKILLS_DIR, "SKILL.md"),
    },
    ...names.map((name) => ({
      source: join(COMMANDS_DIR, `${name}.md`),
      target: join(SKILLS_DIR, name, "SKILL.md"),
    })),
    {
      source: join(COMMANDS_DIR, "rules", "loop-protocol.md"),
      target: join(SKILLS_DIR, "rules", "loop-protocol.md"),
    },
  ];
}

async function existingMirrorEntries(dir = SKILLS_DIR) {
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
      paths.push(...await existingMirrorEntries(path));
    }
  }
  return paths;
}

async function findDrift() {
  const entries = await mirrorEntries();
  const expectedTargets = new Set(entries.map(({ target }) => target));
  const expectedPaths = new Set([SKILLS_DIR]);
  for (const target of expectedTargets) {
    let current = target;
    while (current.startsWith(SKILLS_DIR)) {
      expectedPaths.add(current);
      if (current === SKILLS_DIR) {
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
        issues.push(`missing mirror ${relative(PLUGIN_ROOT, target)}`);
      } else {
        issues.push(`cannot read mirror ${relative(PLUGIN_ROOT, target)}: ${error.message}`);
      }
      continue;
    }

    if (targetText !== sourceText) {
      issues.push(`diverged mirror ${relative(PLUGIN_ROOT, target)}`);
    }
  }

  for (const path of await existingMirrorEntries()) {
    if (!expectedPaths.has(path)) {
      issues.push(`stale mirror ${relative(PLUGIN_ROOT, path)}`);
    }
  }

  return issues;
}

export async function syncSkillsLayout({ check = false } = {}) {
  if (check) {
    const issues = await findDrift();
    if (issues.length > 0) {
      for (const issue of issues) {
        console.error(issue);
      }
      process.exitCode = 1;
      return;
    }
    console.log("skills/turing mirror is in sync");
    return;
  }

  await rm(SKILLS_DIR, { recursive: true, force: true });
  for (const { source, target } of await mirrorEntries()) {
    await copyTextFile(source, target);
  }
  console.log("skills/turing mirror synchronized");
}

const isDirectRun =
  process.argv[1] &&
  fileURLToPath(import.meta.url).endsWith(process.argv[1].replace(/^.*\//, ""));

if (isDirectRun) {
  syncSkillsLayout({ check: process.argv.includes("--check") }).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
