#!/bin/bash
set -eo pipefail
cd "${1:?worktree required}"
expected=${2:?source SHA required}
if [[ "$(git rev-parse HEAD)" != "${expected}" ]]; then
    echo "Source SHA differs from the pinned revision" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "Source worktree has tracked or untracked changes" >&2
    git status --short >&2
    exit 2
fi
