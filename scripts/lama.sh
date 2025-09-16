#!/bin/bash
cd "$(dirname "$0")"
cd ../models

export SINGULARITY_TMPDIR="$HOME/.singularity/tmp/"
export SINGULARITY_CACHEDIR="$HOME/.singularity/cache/"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"


srun --gres=gpu:4 --pty singularity exec --nv lama.sif bash -c 'ifconfig && unset ROCR_VISIBLE_DEVICES && OLLAMA_DEBUG=1 OLLAMA_LOAD_TIMEOUT=10m OLLAMA_HOST=0.0.0.0:${PORT} OLLAMA_MAX_LOADED_MODELS=1 ollama serve'
