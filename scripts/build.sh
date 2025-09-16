#!/bin/bash
cd "$(dirname "$0")"
cd ../models

export SINGULARITY_TMPDIR="$HOME/.singularity/tmp/"
export SINGULARITY_CACHEDIR="$HOME/.singularity/cache/"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"


srun singularity build --fakeroot lama.sif lama.def
