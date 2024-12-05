from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import datasets as Dataset
import torch
import json
from tqdm import tqdm


def load_questions(file_path):
    """Load questions from validation/test set"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = {
                'question': data['question'],
                'body': data.get('body', ''),
            }
            if 'pids' in data:
                question['pids'] = data['pids']
            questions.append(question)
    return questions

def retrieve_papers(dataset_path, question_file, output_file, k):
    # Load dataset
    dataset = Dataset.load_from_disk(dataset_path)
    
    # Load questions
    questions = load_questions(question_file)
    
    # Setup encoders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_encoder = DPRQuestionEncoder.from_pretrained("/scratch/chaijy_root/chaijy2/josuetf/eecs576_datasets/converted_models/question_encoder").to(device)
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    
    # Add FAISS index for retrieval
    dataset.add_faiss_index(column='embeddings')
    
    # Process each question
    results = []
    count = 0
    for q in tqdm(questions, desc="Processing questions"):
        if count == 100:
            break
        # Encode question
        with torch.no_grad():
            question_tokens = q_tokenizer(
                q['question'],
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            question_embedding = q_encoder(**question_tokens).pooler_output.cpu().numpy()[0]
        
        # Get nearest examples
        scores, retrieved_examples = dataset.get_nearest_examples('embeddings', question_embedding, k=k)
        
        # Format results
        retrieved_papers = []
        for score, pid, title, text, embedding in zip(scores, 
                                                    retrieved_examples['pid'],
                                                    retrieved_examples['title'],
                                                    retrieved_examples['text'],
                                                    retrieved_examples['embeddings']):
            retrieved_papers.append({
                "pid": pid,
                "title": title,
                "text": text,
                "embeddings": embedding,
                "score": float(score)
            })

        result = {
            "question": q['question'],
            "body": q['body'],
        }
        if 'pids' in q:
            result['pids'] = q['pids']
        result["retrieved_papers"] = retrieved_papers
        results.append(result)
        count += 1
    
    # Save results
    output_file = f"retrieval_results_{output_file}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    dataset_path = "/scratch/chaijy_root/chaijy2/josuetf/eecs576_datasets/encoded_AQA"
    question_file = "/scratch/chaijy_root/chaijy2/josuetf/AQA/qa_train.txt"

    results = retrieve_papers(dataset_path, question_file, "qa_train", 100)
    print(results)