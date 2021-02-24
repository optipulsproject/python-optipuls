#!/usr/bin/env bash

# Usage:
# ./calculate.sh <revision> <repo-dir> <scratch-dir>


log() { echo ":: $*" >&2; }
err() { echo "E: $*" >&2; }
die() { err "$@"; exit 1; }

# parsing command line arguments

ARG_REV="$1"
if ! [[ "$ARG_REV" ]]; then die "Bad usage: expected $0 <revision>"; fi

if ! REV="$(cd "$REPO_DIR" && git rev-parse --short --quiet "$ARG_REV")"; then die "Bad revision: $ARG_REV"; fi

REPO_DIR="$2"
SCRATCH_DIR="$3"

WORK_TREE_DIR="$SCRATCH_DIR/$REV/code"
OUTPUT_DIR="$SCRATCH_DIR/$REV"


if [[ -d "$OUTPUT_DIR" ]]; then die "Output for revision $REV already exists: $OUTPUT_DIR"; fi
mkdir -p "$OUTPUT_DIR"

rm -rf "$WORK_TREE_DIR"
mkdir -p "$WORK_TREE_DIR"
(cd "$REPO_DIR" && GIT_WORK_TREE="$WORK_TREE_DIR" git checkout -f "$REV" -- .)

PYTHONPATH=${REPO_DIR}:${PYTHONPATH}
(cd "$WORK_TREE_DIR" && python run.py --scratch="$OUTPUT_DIR")
