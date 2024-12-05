from datasets import load_from_disk
import json
import pickle
from tqdm import tqdm

def extract_pid_embeddings(dataset_path='/scratch/chaijy_root/chaijy2/josuetf/eecs576_datasets/encoded_AQA', output_file='pid_embeddings'):
    """Extract PIDs and embeddings from the dataset and save to JSON"""
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Create simplified format
    print("Extracting PIDs and embeddings...")
    pid_embeddings = {}
    for item in tqdm(dataset):
        pid = item['pid']
        embeddings = item['embeddings']
        pid_embeddings[pid] = embeddings
    
    # Save to JSON
    print(f"Saving to {output_file}...")
    with open(f'{output_file}.pickle', 'wb') as handle:
        pickle.dump(pid_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(f"Saving to {output_file}...")
    # with open(output_file, 'w') as f:
    #     json.dump(pid_embeddings, f)
    
    # Print sample to verify
    print("\nFirst PID and embedding shape:")
    first_pid = next(iter(pid_embeddings))
    print(f"PID: {first_pid}")
    print(f"Embedding length: {len(pid_embeddings[first_pid])}")
    
    return pid_embeddings

if __name__ == "__main__":
    # pid_embeddings = extract_pid_embeddings()
    with open('pid_embeddings.pickle', 'rb') as handle:
        pid_embeddings = pickle.load(handle)
    pid = "53e99a4eb7602d97022b40fa"
    print(len(pid_embeddings[pid]))
    print(pid_embeddings[pid])