import argparse
from datasets import load_from_disk
import json
from tqdm import tqdm
import os

def extract_pid_embeddings(dataset_path, output_file):
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    print("Extracting PIDs and embeddings...")
    pid_embeddings = {}
    for item in tqdm(dataset, desc='Processing entries'):
    	pid = item['pid']
    	embeddings = item['embeddings']
    	pid_embeddings[pid] = embeddings

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(pid_embeddings, f)
    
    print("\nExtraction complete:")
    print(f"Number of PIDs processed: {len(pid_embeddings)}")
    
    print("\nSample entry:")
    first_pid = next(iter(pid_embeddings))
    print(f"PID: {first_pid}")
    print(f"Embedding length: {len(pid_embeddings[first_pid])}")
    
def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings from encoded dataset')
    parser.add_argument('--dataset_path', required=True,
                       help='Path to encoded dataset')
    parser.add_argument('--output_file', required=True,
                       help='Output JSON file for pid-embedding pairs')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extract_pid_embeddings(args.dataset_path, args.output_file)