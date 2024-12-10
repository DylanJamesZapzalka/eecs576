#!/bin/bash
set -e

# Ensure proper usage
if [ "$#" -lt 6 ]; then
    echo "Usage: $0 papers_file train_file test_file output_dir model_type (finetuned/multiset/nq) [model_path (finetuned only)] [batch_size] [top_k]"
    echo "Example: $0 AQA/pid_to_title_abs_new.json AQA/qa_train.txt AQA/qa_test_wo_ans.txt results finetuned AQA/finetuned_models 32 100"
    exit 1
fi

PAPERS_FILE=$1
TRAIN_FILE=$2
TEST_FILE=$3
OUTPUT_DIR=$4
MODEL_TYPE=$5
MODEL_PATH=${6:-}
BATCH_SIZE=${7:-32}
TOP_K=${8:-100}

ENCODED_DATASET_DIR="${OUTPUT_DIR}/encoded_AQA"
echo "Step 1: Running embedder..."
EMBEDDER_ARGS="sh run_dpr_embedder.sh $PAPERS_FILE $ENCODED_DATASET_DIR $MODEL_TYPE"
if [ "$MODEL_TYPE" = "finetuned" ]; then
    EMBEDDER_ARGS="$EMBEDDER_ARGS $MODEL_PATH"
fi
if [ -n "$BATCH_SIZE" ]; then
    EMBEDDER_ARGS="$EMBEDDER_ARGS $BATCH_SIZE"
fi
eval $EMBEDDER_ARGS

RESULTS_DIR="${OUTPUT_DIR}/results"
echo "Step 2: Running retriever..."
RETRIEVER_ARGS="sh run_dpr_retriever.sh $ENCODED_DATASET_DIR $TEST_FILE $RESULTS_DIR $MODEL_TYPE"
if [ "$MODEL_TYPE" = "finetuned" ]; then
    RETRIEVER_ARGS="$RETRIEVER_ARGS $MODEL_PATH"
fi
if [ -n "$TOP_K" ]; then
    RETRIEVER_ARGS="$RETRIEVER_ARGS $TOP_K"
fi
eval $RETRIEVER_ARGS

echo "DPR pipeline finished!"