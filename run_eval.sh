#!/bin/sh

source /data/$USER/conda/etc/profile.d/conda.sh && source /data/$USER/conda/etc/profile.d/mamba.sh && mamba activate py38

python run_eval.py