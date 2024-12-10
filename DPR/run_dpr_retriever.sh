#!/bin/bash
set -e

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 encoded_dataset_dir questions_file output_dir model_type (finetuned/multiset/nq) [model_path (finetuned only)] [top_k]"
    echo "Example: $0 encoded_AQA AQA/qa_test_wo_ans.txt results finetuned AQA/finetuned_models 100"
    exit 1
fi

ENCODED_DATASET=$1
QUESTIONS_FILE=$2
OUTPUT_DIR=$3
MODEL_TYPE=$4
MODEL_PATH=$5
TOP_K=${6:-100}

ARGS="--encoded_dataset $ENCODED_DATASET --questions_file $QUESTIONS_FILE --output_dir $OUTPUT_DIR --model_type $MODEL_TYPE"

if [ "$MODEL_TYPE" = "finetuned" ]; then
    ARGS="$ARGS --model_path $MODEL_PATH"
fi

if [ -n "$TOP_K" ]; then
    ARGS="$ARGS --top_k $TOP_K"
fi

python dpr_retriever.py $ARGS