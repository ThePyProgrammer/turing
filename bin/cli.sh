#!/usr/bin/env bash
# Helios CLI — entry point for `npx claude-helios`.
#
# Subcommands:
#   install [--global]  Deploy to Claude Code plugin directory
#   verify  [--global]  Check installation completeness
#   init    [name] [dir] Scaffold ML project (non-Claude Code usage)
#
# For Claude Code usage, use /helios:* slash commands instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"

COMMAND="${1:-help}"
shift 2>/dev/null || true

case "$COMMAND" in
    install)
        node "${PLUGIN_DIR}/src/install.js" "$@"
        ;;
    verify)
        node "${PLUGIN_DIR}/src/verify.js" "$@"
        ;;
    init)
        bash "${PLUGIN_DIR}/bin/helios-init.sh" "$@"
        ;;
    help|--help|-h)
        echo "Helios ML Research Harness"
        echo ""
        echo "Usage: claude-helios <command> [options]"
        echo ""
        echo "Commands:"
        echo "  install [--global]    Deploy commands/agents to Claude Code"
        echo "  verify  [--global]    Check installation completeness"
        echo "  init [name] [dir]     Scaffold ML project (CLI mode)"
        echo "  help                  Show this help"
        echo ""
        echo "For Claude Code: use /helios:init, /helios:train, etc."
        ;;
    *)
        echo "Unknown command: $COMMAND" >&2
        echo "Run 'claude-helios help' for usage." >&2
        exit 1
        ;;
esac
