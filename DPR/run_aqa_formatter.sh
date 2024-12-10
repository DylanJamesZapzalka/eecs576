#!/bin/bash
set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 papers_file train_file output_path [num_negatives]"
    echo "Example: $0 AQA/pid_to_title_abs_new.json AQA/qa_train.txt qa_train_formatted.json 16"
    exit 1
fi

PAPERS_FILE=$1
TRAIN_FILE=$2
OUTPUT_FILE=$3
NUM_NEGATIVES=${4:-16}

ARGS="--papers_file $PAPERS_FILE --questions_file $TRAIN_FILE --output_path $OUTPUT_FILE --num_negatives $NUM_NEGATIVES"

python aqa_formatter.py $ARGS