#!/usr/bin/env node
/**
 * npm postinstall hook.
 *
 * Prints setup instructions after `npm install claude-turing`.
 * Does NOT auto-install — the user must explicitly run the installer.
 */

console.log("");
console.log("╔══════════════════════════════════════════════════════╗");
console.log("║           Turing ML Research Harness                ║");
console.log("║                                                     ║");
console.log("║  To complete setup, run:                            ║");
console.log("║    npx claude-turing install --global               ║");
console.log("║                                                     ║");
console.log("║  Or within a project:                               ║");
console.log("║    npx claude-turing install                        ║");
console.log("║                                                     ║");
console.log("║  Then in Claude Code:                               ║");
console.log("║    /turing:init                                     ║");
console.log("╚══════════════════════════════════════════════════════╝");
console.log("");
