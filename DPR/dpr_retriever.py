import argparse
import json
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import os

def load_questions(file_path):
    questions = []
    print(f"Loading questions from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading questions"):
            data = json.loads(line)
            question = {
                'question': data['question'],
                'body': data.get('body', ''),
            }
            if 'pids' in data:
                question['pids'] = data['pids']
            questions.append(question)
    return questions

def retrieve_papers(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Dataset.load_from_disk(args.encoded_dataset)
    questions = load_questions(args.questions_file)
    
    if args.model_type == 'finetuned':
        print(f"Loading finetuned model from {args.model_path}")
        q_encoder = DPRQuestionEncoder.from_pretrained(args.model_path + "/question_encoder")
    elif args.model_type == 'multiset':
        print("Loading multiset base model")
        q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    elif args.model_type == 'nq':
        print("Loading NQ base model")
        q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    q_encoder = q_encoder.to(device).eval()
    
    dataset.add_faiss_index(column='embeddings')
    
    results = []
    for q in tqdm(questions, desc="Processing questions"):
        with torch.no_grad():
            question_tokens = q_tokenizer(
                q['question'],
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            question_embedding = q_encoder(**question_tokens).pooler_output.cpu().numpy()[0]
        
        scores, retrieved = dataset.get_nearest_examples('embeddings', question_embedding, k=args.top_k)
        
        retrieved_papers = []
        for score, pid, embedding in zip(scores, 
                                       retrieved['pid'],
                                       retrieved['embeddings']):
            retrieved_papers.append({
                "pid": pid,
                "embeddings": embedding,
                "score": float(score)
            })
        
        result = {"question": q['question']}
        if 'pids' in q:
            result['pids'] = q['pids']
        result["retrieved_papers"] = retrieved_papers
        results.append(result)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'{os.path.basename(args.questions_file)}_results.json')
    
    print(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve papers using DPR')
    parser.add_argument('--encoded_dataset', required=True,
                       help='Path to encoded dataset')
    parser.add_argument('--questions_file', required=True,
                       help='Path to questions file')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--model_type', choices=['finetuned', 'multiset', 'nq'],
                       required=True, help='Type of DPR model to use')
    parser.add_argument('--model_path',
                       help='Path to finetuned model (required if model_type is finetuned)')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of papers to retrieve')
    args = parser.parse_args()
    
    if args.model_type == 'finetuned' and not args.model_path:
        parser.error("--model_path required when model_type is finetuned")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    retrieve_papers(args)