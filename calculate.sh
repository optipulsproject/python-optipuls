#!/usr/bin/env bash

# Usage:
# calculate.sh experiments/85f11fc

# Setting up the conda environment
__conda_setup="$('/usr/scratch4/dmst0651/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/scratch4/dmst0651/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/scratch4/dmst0651/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/scratch4/dmst0651/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate FEniCS-conda-env

# For TU Ilmenau HPC:
REPO_DIR=/usr/wrk/people9/dmst0651/repos/code-FEniCS.git
SCRATCH_DIR=/usr/scratch4/dmst0651

log() { echo ":: $*" >&2; }
err() { echo "E: $*" >&2; }
die() { err "$@"; exit 1; }

ARG_REV="$1"
if ! [[ "$ARG_REV" ]]; then die "Bad usage: expected $0 <revision>"; fi

if ! REV="$(cd "$REPO_DIR" && git rev-parse --short --quiet "$ARG_REV")"; then die "Bad revision: $ARG_REV"; fi

TARGET_DIR="$SCRATCH_DIR/tmp/$REV"
OUTPUT_DIR="$SCRATCH_DIR/output/$REV"

if [[ -d "$OUTPUT_DIR" ]]; then die "Output for revision $REV already exists: $OUTPUT_DIR"; fi
mkdir -p "$OUTPUT_DIR"

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
(cd "$REPO_DIR" && GIT_WORK_TREE="$TARGET_DIR" git checkout -f "$REV" -- .)

(cd "$TARGET_DIR" && python fenics_simulation.py --output="$OUTPUT_DIR")
