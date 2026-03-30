#!/usr/bin/env bash
# Turing CLI — entry point for `npx claude-turing`.
#
# Subcommands:
#   install [--global]  Deploy to Claude Code plugin directory
#   verify  [--global]  Check installation completeness
#   init    [name] [dir] Scaffold ML project (non-Claude Code usage)
#
# For Claude Code usage, use /turing:* slash commands instead.

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
        bash "${PLUGIN_DIR}/bin/turing-init.sh" "$@"
        ;;
    help|--help|-h)
        echo "Turing ML Research Harness"
        echo ""
        echo "Usage: claude-turing <command> [options]"
        echo ""
        echo "Commands:"
        echo "  install [--global]    Deploy commands/agents to Claude Code"
        echo "  verify  [--global]    Check installation completeness"
        echo "  init [name] [dir]     Scaffold ML project (CLI mode)"
        echo "  help                  Show this help"
        echo ""
        echo "For Claude Code: use /turing:init, /turing:train, etc."
        ;;
    *)
        echo "Unknown command: $COMMAND" >&2
        echo "Run 'claude-turing help' for usage." >&2
        exit 1
        ;;
esac
