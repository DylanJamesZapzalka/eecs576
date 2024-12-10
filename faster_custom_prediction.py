import json
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from amrlib import load_stog_model
from tqdm import tqdm
import os
from datetime import datetime
import csv
import math
import glob

def get_last_processed_index(chunk_dir: str) -> int:
    """
    Find the last processed index in a chunk directory
    """
    if not os.path.exists(chunk_dir):
        return -1
        
    # Get all batch files
    batch_files = glob.glob(os.path.join(chunk_dir, "batch_*.csv"))
    if not batch_files:
        return -1
        
    last_index = -1
    for file in batch_files:
        try:
            # Read the last line of each file to get the last index
            with open(file, 'r', encoding='utf-8') as f:
                # Skip header
                next(f)
                # Get last line by reading entire file (files should be small)
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    # Last column is the index
                    index = int(last_line.strip().split(',')[-1])
                    last_index = max(last_index, index)
        except Exception as e:
            print(f"Warning: Error reading file {file}: {e}")
            continue
            
    return last_index

def process_chunk(data_items, chunk_id, total_chunks, output_dir, batch_size=32, 
                 num_workers=4, device='cuda', reverse=False):
    """
    Process a chunk with resume capability
    """
    # Calculate chunk boundaries
    chunk_size = math.ceil(len(data_items) / total_chunks)
    start_idx = chunk_id * chunk_size
    end_idx = min((chunk_id + 1) * chunk_size, len(data_items))
    
    # Create chunk directory
    chunk_dir = os.path.join(output_dir, f'chunk_{chunk_id}')
    os.makedirs(chunk_dir, exist_ok=True)
    
    # Find last processed index
    last_processed = get_last_processed_index(chunk_dir)
    if last_processed >= end_idx - 1:
        print(f"Chunk {chunk_id} is already complete. Skipping.")
        return
    
    # Adjust start index for resuming
    if last_processed >= start_idx:
        start_idx = last_processed + 1
        print(f"Resuming chunk {chunk_id} from index {start_idx}")
    
    # Get chunk data
    chunk_data = data_items[start_idx:end_idx]
    if reverse:
        chunk_data = list(reversed(chunk_data))
    
    print(f"\nProcessing chunk {chunk_id}:")
    print(f"Total indices: {start_idx} to {end_idx-1}")
    print(f"Items to process: {len(chunk_data)}")
    
    # Load model
    stog = load_stog_model(device=device)
    
    # Create batches
    for batch_start in tqdm(range(0, len(chunk_data), batch_size), 
                           desc=f"Chunk {chunk_id} batches"):
        batch_end = min(batch_start + batch_size, len(chunk_data))
        batch = chunk_data[batch_start:batch_end]
        
        # Extract sentences and indices
        sentences = [item[2] for item in batch]  # item = (index, paper_id, sentence)
        indices = [item[0] for item in batch]
        paper_ids = [item[1] for item in batch]
        
        try:
            # Process batch
            amr_graphs = stog.parse_sents(sentences)
            
            # Save batch results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_file = os.path.join(
                chunk_dir, 
                f"batch_{min(indices)}_{max(indices)}_{timestamp}.csv"
            )
            
            with open(batch_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['paper_id', 'sentence', 'amr', 'index'])
                for paper_id, sent, amr, idx in zip(paper_ids, sentences, amr_graphs, indices):
                    writer.writerow([paper_id, sent, amr, idx])
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\nHandling OOM error...")
                torch.cuda.empty_cache()
                # Process in smaller batches
                for i in range(0, len(sentences), batch_size // 2):
                    sub_batch_end = min(i + batch_size // 2, len(sentences))
                    sub_sentences = sentences[i:sub_batch_end]
                    sub_indices = indices[i:sub_batch_end]
                    sub_paper_ids = paper_ids[i:sub_batch_end]
                    
                    sub_amr_graphs = stog.parse_sents(sub_sentences)
                    
                    # Save sub-batch results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    sub_batch_file = os.path.join(
                        chunk_dir,
                        f"batch_{min(sub_indices)}_{max(sub_indices)}_{timestamp}.csv"
                    )
                    
                    with open(sub_batch_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['paper_id', 'sentence', 'amr', 'index'])
                        for p_id, sent, amr, idx in zip(sub_paper_ids, sub_sentences, 
                                                      sub_amr_graphs, sub_indices):
                            writer.writerow([p_id, sent, amr, idx])
            else:
                raise e
        
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--chunk-id', type=int, required=True)
    parser.add_argument('--total-chunks', type=int, required=True)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    with open('pid_to_title_abs_new.json', 'r') as f:
        data = json.load(f)
    
    # Prepare data items
    data_items = []
    index = 0
    for paper_id, paper_info in data.items():
        sentences = nltk.sent_tokenize(paper_info['abstract'])
        for sent in sentences:
            data_items.append((index, paper_id, sent))
            index += 1
    
    # Process chunk
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    process_chunk(
        data_items=data_items,
        chunk_id=args.chunk_id,
        total_chunks=args.total_chunks,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        reverse=args.reverse
    )