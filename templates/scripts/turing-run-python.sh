#!/usr/bin/env bash

run_python() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "turing: uv is required. Install uv or run legacy environment setup manually." >&2
        return 127
    fi
    uv run python "$@"
}
