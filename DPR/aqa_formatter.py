import argparse
import json
import random
from tqdm import tqdm
import os

def create_dpr_training_data(papers_path, questions_path, output_path, num_negatives=3):
    print(f"Loading papers from {papers_path}")
    with open(papers_path, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)
    
    pid_to_paper = {
        pid: {
            'title': paper_info['title'],
            'abstract': paper_info['abstract']
        }
        for pid, paper_info in papers_data.items()
    }
    
    all_pids = list(papers_data.keys())
    dpr_format_data = []
    
    print(f"Processing questions from {questions_path}")
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Converting to DPR format"):
            question_data = json.loads(line.strip())
            
            if 'pids' in question_data:
                all_positive_pids = set(question_data['pids'])
                positive_ctxs = []
                
                for pid in question_data['pids']:
                    if pid in pid_to_paper:
                        paper = pid_to_paper[pid]
                        positive_ctxs.append({
                            'title': paper['title'],
                            'text': paper['abstract'],
                            'pids': [pid]
                        })
                
                # Random sample both negative and hard negative contexts
                negative_pids = [pid for pid in all_pids if pid not in all_positive_pids]
                
                negative_ctxs = []
                hard_negative_ctxs = []
                
                if negative_pids:
                    sampled_neg_pids = random.sample(negative_pids, min(num_negatives, len(negative_pids)))
                    negative_ctxs = [{
                        'title': pid_to_paper[pid]['title'],
                        'text': pid_to_paper[pid]['abstract'],
                        'pids': [pid]
                    } for pid in sampled_neg_pids]
                    
                    remaining_neg_pids = [pid for pid in negative_pids if pid not in sampled_neg_pids]
                    if remaining_neg_pids:
                        sampled_hard_neg_pids = random.sample(remaining_neg_pids, min(num_negatives, len(remaining_neg_pids)))
                        hard_negative_ctxs = [{
                            'title': pid_to_paper[pid]['title'],
                            'text': pid_to_paper[pid]['abstract'],
                            'pids': [pid]
                        } for pid in sampled_hard_neg_pids]
                
                dpr_entry = {
                    "question": question_data['question'],
                    "answers": question_data['body'],
                    "positive_ctxs": positive_ctxs,
                    "negative_ctxs": negative_ctxs,
                    "hard_negative_ctxs": hard_negative_ctxs
                }
                dpr_format_data.append(dpr_entry)
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving formatted data to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dpr_format_data, f, indent=2)
    
    print(f"Created {len(dpr_format_data)} training examples")

def parse_args():
    parser = argparse.ArgumentParser(description='Format AQA data for DPR training')
    parser.add_argument('--papers_file', required=True,
                       help='Input JSON file containing papers')
    parser.add_argument('--questions_file', required=True,
                       help='Input file containing questions')
    parser.add_argument('--output_path', required=True,
                       help='Output path for formatted data')
    parser.add_argument('--num_negatives', type=int, default=16,
                       help='Number of negative examples per question')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_dpr_training_data(args.papers_file, args.questions_file, args.output_path, args.num_negatives)