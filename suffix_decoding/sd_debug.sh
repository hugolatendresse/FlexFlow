#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

reset
make -j
source set_python_envs.sh

matching_strategy="dynamic_token_tree"
max_tree_depth=16
max_spec_factor=1.0

# export CUDA_VISIBLE_DEVICES=1,2,3,5
export CUDA_VISIBLE_DEVICES=2,3,4,5
export LEGION_BACKTRACE=1

########### LLAMA 70B on 8 GPUs ###########
model_name="meta-llama/Meta-Llama-3-70B-Instruct"
partition_name="SQL_FANOUT1"
NGPUS=8
FSIZE=32000
ZSIZE=200000
CSIZE=200000

########### LLAMA 70B on 4 GPUs ###########
model_name="meta-llama/Meta-Llama-3-70B-Instruct"
partition_name="FEATURE_EXTRACTION"
NGPUS=4
FSIZE=38000
ZSIZE=200000
CSIZE=200000

########### LLAMA 8B on 8 GPUs ###########
# model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# partition_name="FEATURE_EXTRACTION"
# NGPUS=1
# FSIZE=30000
# ZSIZE=30000
# CSIZE=60000

echo "Running partition ${partition_name} with model ${model_name} and "
python ../inference/utils/download_hf_model.py --half-precision-only $model_name

rm /usr/FlexFlow/inference/output/cortex_${partition_name}_sd.out || true

./inference/suffix_decoding/suffix_decoding \
    -ll:gpu $NGPUS -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree $NGPUS \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
    --fusion \
    --max-sequence-length 1200 \
    --max-requests-per-batch 8 \
    --max-tokens-per-batch 1024 \
    --max-output-length 900 \
    --matching-strategy $matching_strategy \
    --max-tree-depth $max_tree_depth \
    --max-spec-factor $max_spec_factor \
    -llm-model $model_name \
    -trace /usr/suffix-tree-decoding/trace/cortex.json \
    -trace-output-path /usr/FlexFlow/inference/output/cortex_ff_${partition_name}.json \
    -output-file /usr/FlexFlow/inference/output/cortex_${partition_name}.out \
    -csv-output-path /usr/FlexFlow/inference/output/cortex_${partition_name}.csv \
    -target-partition ${partition_name}