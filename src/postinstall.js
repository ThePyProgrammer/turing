#!/usr/bin/env node
/**
 * npm postinstall hook.
 *
 * Prints setup instructions after `npm install claude-helios`.
 * Does NOT auto-install — the user must explicitly run the installer.
 */

console.log("");
console.log("╔══════════════════════════════════════════════════════╗");
console.log("║           Helios ML Research Harness                ║");
console.log("║                                                     ║");
console.log("║  To complete setup, run:                            ║");
console.log("║    npx claude-helios install --global               ║");
console.log("║                                                     ║");
console.log("║  Or within a project:                               ║");
console.log("║    npx claude-helios install                        ║");
console.log("║                                                     ║");
console.log("║  Then in Claude Code:                               ║");
console.log("║    /helios:init                                     ║");
console.log("╚══════════════════════════════════════════════════════╝");
console.log("");
