#!/bin/bash

# Redirect all cache/temp directories away from home
export TMPDIR=/bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/tmp
export TF_CACHE_DIR=/bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/tmp/tf_cache
export XDG_CACHE_HOME=/bigdata/MIMIC-IV/MIMIC-IV/AFA-Benchmark/tmp/cache
export HYDRA_FULL_ERROR=1

mkdir -p "$TMPDIR" "$TF_CACHE_DIR" "$XDG_CACHE_HOME"


WANDB_PROJECT=afabench uv run snakemake -s extra/workflow/snakefiles/orchestration/pipeline.smk all --configfile extra/workflow/conf/eval_hard_budgets.yaml extra/workflow/conf/methods.yaml extra/workflow/conf/method_sets.yaml extra/workflow/conf/method_options.yaml extra/workflow/conf/pretrain_mapping.yaml extra/workflow/conf/soft_budget_params.yaml extra/workflow/conf/unmaskers.yaml extra/workflow/conf/classifier_names.yaml extra/workflow/conf/datasets_mimic.yaml --config eval_dataset_split=test "dataset_instance_indices=[0]" smoke_test=false use_wandb=false device=cpu --cores 2
