#!/bin/bash

srun singularity exec --nv models/lama.sif bash scripts/clone.sh


ollama serve  &
sleep 5


ollama pull qwen2.5:32b

