#!/bin/bash
set -e

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 papers_file output_dir model_type (finetuned/multiset/nq) [model_path (finetuned only)] [batch_size]"
    echo "Example: $0 AQA/pid_to_title_abs_new.json encoded_AQA finetuned AQA/finetuned_models 32"
    exit 1
fi

PAPERS_FILE=$1
OUTPUT_DIR=$2
MODEL_TYPE=$3
MODEL_PATH=$4
BATCH_SIZE=${5:-32}

ARGS="--papers_file $PAPERS_FILE --output_dir $OUTPUT_DIR --model_type $MODEL_TYPE"

if [ "$MODEL_TYPE" = "finetuned" ]; then
    ARGS="$ARGS --model_path $MODEL_PATH"
fi

if [ -n "$BATCH_SIZE" ]; then
    ARGS="$ARGS --batch_size $BATCH_SIZE"
fi

python dpr_embedder.py $ARGS