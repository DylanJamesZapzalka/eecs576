import argparse
import torch
from datasets import Dataset
import json
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import os

def create_aqa_dataset(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'finetuned':
        print(f"Loading finetuned model from {args.model_path}")
        ctx_encoder = DPRContextEncoder.from_pretrained(args.model_path + "/ctx_encoder")
    elif args.model_type == 'multiset':
        print("Loading multiset base model")
        ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    elif args.model_type == 'nq':
        print("Loading NQ base model")
        ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    ctx_encoder = ctx_encoder.to(device).eval()
    
    print(f"Loading papers from {args.papers_file}")
    with open(args.papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    dataset_dict = {
        'pid': [],
        'title': [],
        'text': [],
        'embeddings': []
    }
    
    papers_list = list(papers.items())
    print(f"Encoding {len(papers_list)} papers in batches of {args.batch_size}")
    
    for i in tqdm(range(0, len(papers_list), args.batch_size)):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        batch = papers_list[i:i + args.batch_size]
        
        texts = [paper[1]['abstract'] for paper in batch]
        
        with torch.no_grad():
            inputs = ctx_tokenizer(
                texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            embeddings = ctx_encoder(**inputs).pooler_output.cpu().numpy()
        
        for idx, (pid, paper) in enumerate(batch):
            dataset_dict['pid'].append(pid)
            dataset_dict['title'].append(paper['title'])
            dataset_dict['text'].append(paper['abstract'])
            dataset_dict['embeddings'].append(embeddings[idx])
    
    print("Creating dataset...")
    dataset = Dataset.from_dict(dataset_dict)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving dataset to {args.output_dir}")
    dataset.save_to_disk(args.output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Encode papers using DPR')
    parser.add_argument('--papers_file', required=True,
                       help='Input JSON file containing papers')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for encoded dataset')
    parser.add_argument('--model_type', choices=['finetuned', 'multiset', 'nq'],
                       required=True, help='Type of DPR model to use')
    parser.add_argument('--model_path', 
                       help='Path to finetuned model (required if model_type is finetuned)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for encoding')
    args = parser.parse_args()
    
    if args.model_type == 'finetuned' and not args.model_path:
        parser.error("--model_path required when model_type is finetuned")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    create_aqa_dataset(args)