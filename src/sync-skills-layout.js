#!/usr/bin/env node
/**
 * Backward-compatible wrapper for the flipped source layout.
 *
 * The editable source is now skills/turing/, and sync generates commands/.
 */

import { fileURLToPath } from "url";
import { syncCommandsLayout } from "./sync-commands-layout.js";

const isDirectRun =
  process.argv[1] &&
  fileURLToPath(import.meta.url).endsWith(process.argv[1].replace(/^.*\//, ""));

if (isDirectRun) {
  syncCommandsLayout({ check: process.argv.includes("--check") }).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
