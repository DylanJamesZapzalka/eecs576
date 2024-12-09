#!/bin/bash

# Define the arrays of parameters
gnn_types=("gat" "sage" "gcn")
loss_functions=("pairwise" "cross_entropy")

# Iterate through all combinations
for gnn_type in "${gnn_types[@]}"; do
  for loss_function in "${loss_functions[@]}"; do
    # Create an output filename based on the current combination
    output_file="${gnn_type}_${loss_function}_amr_kg_output.txt"

    # Run the command with the current combination and redirect output to the file
    python aqa_amr_main_experiments.py --model_name amr+kg \
      --train_num_samples 8757 \
      --test_num_samples 2919 \
      --gnn_type "$gnn_type" \
      --amr_number_of_links 20 \
      --num_epochs 10 \
      --loss_function "$loss_function" > "$output_file" 2>&1
  done
done
