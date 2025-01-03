import argparse
import os
import pickle
import pprint

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

import constants
import wandb
from amr_bart_utils import load_data_aqa, load_data_aqa_val
from models import GCN, GAT, GraphSAGE
from utils import get_data_amr, get_data_amg_plus_kg

# Make sure cuda is being used
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'The device I am using is: {device}')

# Get the necessary constants
NQ_TEST_FILE_NAME = constants.NQ_TEST_FILE_NAME
TRIVIA_TEST_FILE_NAME = constants.TRIVIA_TEST_FILE_NAME
AMR_NQ_TEST_FILE_NAME = constants.AMR_NQ_TEST_FILE_NAME
AMR_NQ_TRAIN_FILE_NAME = constants.AMR_NQ_TRAIN_FILE_NAME
DATASETS_DIR = constants.DATASETS_DIR
AQA_TRAIN_FILE_NAME = "/scratch/chaijy_root/chaijy2/josuetf/eecs576_datasets/retrieval_results_qa_train.json"
AQA_TEST_FILE_NAME = "/scratch/chaijy_root/chaijy2/josuetf/eecs576_datasets/retrieval_results_qa_test_wo_ans.json"
AQA_VAL_FILE_NAME = "/scratch/chaijy_root/chaijy2/josuetf/eecs576_datasets/retrieval_results_qa_valid_wo_ans.json"
AQA_VAL_ANS = "/scratch/chaijy_root/chaijy2/josuetf/eecs576_datasets/qa_valid_flag.txt"


# Get arguments for experiments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, type=str, help='Model used in the experiments. Can be "kg", "amr" or "amr+kg"')
parser.add_argument("--train_num_samples", required=True, type=int, help='Number of samples used to train the model.')
parser.add_argument("--test_num_samples", required=True, type=int, help='Number of samples used to test the model.')
parser.add_argument("--kg_link_type", type=str, help='Method that will be used to creat the graph. Can be "ssr" or "se".')
parser.add_argument("--kg_number_of_links", type=int, help='Number of connections needed to create an edge for the reranking graph.')
parser.add_argument("--amr_number_of_links", type=int, help='Number of connections needed to create an edge for the reranking graph.')
parser.add_argument("--gnn_type", required=True, type=str, help='Type of GNN for the reranker. Can be "gcn", "gat", or "sage".')
parser.add_argument("--num_epochs", required=True, type=int, help='Number of epochs the GNN model will be trained over.')
parser.add_argument("--batch_size", default=8, type=int, help='Batch size for training the gnn.')
args = parser.parse_args()


# Load in the pretrained context encoders over the multiset datasets
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(device).eval()
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
# Load in the pretrained question encoders over the multiset datasets
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base").to(device).eval()
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")

#Retreive Questions and Answers for Train and Test

aqa_data_train = load_data_aqa(AQA_TRAIN_FILE_NAME, args.train_num_samples)
questions_array_train = [example['question'] for example in aqa_data_train]
answers_array_train = [example['pids'] for example in aqa_data_train]

aqa_data_test = load_data_aqa_val(AQA_VAL_FILE_NAME, AQA_VAL_ANS, args.test_num_samples)
questions_array_test = [example['question'] for example in aqa_data_test]
answers_array_test = [example['pids'] for example in aqa_data_test]

#Create or load in questiom_embeddings

if os.path.exists("question_embeddings_train.pickle"):
    with open('question_embeddings_train.pickle', 'rb') as handle:
        question_embeddings_train = pickle.load(handle)
else:
    question_embeddings_train = []
    for question in tqdm(questions_array_train, desc='Embedding questions'):
        question_tokens = q_tokenizer(question, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        question_embedding = q_encoder(**question_tokens)
        question_embedding = question_embedding.pooler_output
        question_embedding = question_embedding.cpu().detach().numpy()
        question_embeddings_train.append(question_embedding)

    with open('question_embeddings_train.pickle', 'wb') as handle:
        pickle.dump(question_embeddings_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

if os.path.exists("question_embeddings_test.pickle"):
    with open('question_embeddings_test.pickle', 'rb') as handle:
        question_embeddings_test = pickle.load(handle)
else:
    question_embeddings_test = []
    for question in tqdm(questions_array_test, desc='Embedding questions'):
        question_tokens = q_tokenizer(question, max_length=512, truncation=True, padding='max_length', return_tensors='pt').to(device)
        question_embedding = q_encoder(**question_tokens)
        question_embedding = question_embedding.pooler_output
        question_embedding = question_embedding.cpu().detach().numpy()
        question_embeddings_test.append(question_embedding)

    with open('question_embeddings_test.pickle', 'wb') as handle:
        pickle.dump(question_embeddings_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

#loading AMR graphs
print("loading amr_graphs...")
data_list = []
with open('/scratch/chaijy_root/chaijy2/josuetf/eecs576_datasets/amr_graphs.pickle', 'rb') as handle:
    amr_data = pickle.load(handle)

#loading dictionary of pid to abstract embeddings
print("loading pid_embeddings...")
emb_path = "/scratch/chaijy_root/chaijy2/josuetf/eecs576/pid_embeddings.pickle"
with open(emb_path, 'rb') as file:
    embeddings_dict = pickle.load(file)

#Load exiting dataset or create dataset for training and testing
if args.model_name == 'amr' and os.path.exists("data_loader_train_amr.pth"):
    data_loader_train = torch.load("data_loader_train_amr.pth")
elif args.model_name == 'amr+kg' and os.path.exists("data_loader_train_amr_kg.pth"):
    data_loader_train = torch.load("data_loader_train_amr_kg.pth")
else:
    if args.model_name == 'amr':
        for i in tqdm(range(0, len(question_embeddings_train)), total=args.train_num_samples, desc='Creating dataset'):
            # Get question and answers
            question_embedding = question_embeddings_train[i]
            # Get 100 nearest examples via DPR
            retrieved_examples = aqa_data_train[i]['retrieved_papers']
            answers = aqa_data_train[i]['pids']
            data = get_data_amr(retrieved_examples, answers, embeddings_dict, amr_data, args.amr_number_of_links, question_embedding)
            data_list.append(data)
            count += 1
        data_loader_train = DataLoader(data_list, batch_size=args.batch_size)
        torch.save(data_loader_train, "data_loader_train_amr.pth")
    if args.model_name == 'amr+kg':
        for i in tqdm(range(len(question_embeddings_train)), desc='Creating amr+kg dataset'):
            # Get question and answers
            question_embedding = question_embeddings_train[i]
            # Get 100 nearest examples via DPR
            retrieved_examples = aqa_data_train[i]['retrieved_papers']
            answers = aqa_data_train[i]['pids']
            pkl_path_kg = f'data/train_kgs/{i}.pkl'
            pkl_path_amr = f'data/train_amrs/{i}.pkl'
            data = get_data_amg_plus_kg(pkl_path_kg, pkl_path_amr, retrieved_examples, answers, embeddings_dict, amr_data, args.amr_number_of_links, question_embedding)
            data_list.append(data)
        data_loader_train = DataLoader(data_list, batch_size=args.batch_size)
        torch.save(data_loader_train, "data_loader_train_amr_kg.pth")

#Initialize hyperparameter sweep

sweep_config = {
    'method': 'grid'
}

metric = {
    'name': 'mrr',
    'goal': 'maximize'
}

sweep_config['metric'] = metric

parameters_dict = {
    'l2': {
        'values': [1e-1, 1e-3, 0]
    },
    'architecture': {
        'values': ['gcn', 'gat', 'sage']
    }
}

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="AQA")

def train_and_evaluate(config=None):
    with wandb.init(config=config):

        config = wandb.config

        # Get the reranker GNN
        if config.architecture == 'gcn':
            model = GCN().to(device)
        elif config.architecture == 'gat':
            model = GAT().to(device)
        elif config.architecture == 'sage':
            model = GraphSAGE().to(device)
        else:
            raise Exception("Invalid value for gnn_type.")

        # Get the loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=config.l2)
        ce_loss = torch.nn.CrossEntropyLoss()
        
        # Start training the GNN
        num_edge_indices = 0
        for i in tqdm(range(args.num_epochs), desc='Training the reranker...'):
            model.train()
            epoch_loss = 0

            for batch in data_loader_train:
                batch.x = batch.x.to(device)
                batch.y = batch.y.to(device)
                batch.edge_index = batch.edge_index.to(device)
                num_edge_indices += batch.edge_index.shape[1]
                question_embedding = batch.question_embedding
                outputs = model(batch)
                outputs = torch.split(outputs, 100)
                outputs = torch.stack(outputs, dim=0)
                question_embedding = torch.tensor(question_embedding).to(device)
                scores = torch.matmul(outputs, question_embedding.transpose(1, 2)).squeeze(-1)
                # Get loss
                loss = ce_loss(scores.view(-1), batch.y.squeeze())
                # Perform backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Log training metrics
            wandb.log({
                'epoch': i,
                'train_loss': epoch_loss / len(data_loader_train),
                'l2': config.l2
            })

        print(f'The average number of edge indices per graph is: {num_edge_indices / (args.train_num_samples * args.num_epochs)}')

        # Start the evaluation process
        model.eval()
        accuracy_5 = 0
        accuracy_10 = 0
        accuracy_20 = 0
        mhits_5 = 0
        mhits_10 = 0
        mhits_20 = 0
        mrr = 0

        with torch.no_grad():
            for i in tqdm(range(len(question_embeddings_test)), desc='Evaluating over each question/answer'):
                # Get question and answers
                question_embedding = question_embeddings_test[i]
                answers = answers_array_test[i]

                if args.model_name == 'amr':
                    retrieved_examples = aqa_data_test[i]['retrieved_papers']
                    answers = aqa_data_test[i]['pids']
                    data = get_data_amr(retrieved_examples, answers, embeddings_dict, amr_data, args.amr_number_of_links, question_embedding)
                    data = data.to(device)
                    y = data.y
                elif args.model_name == 'amr+kg':
                    retrieved_examples = aqa_data_test[i]['retrieved_papers']
                    answers = aqa_data_test[i]['pids']
                    pkl_path_kg = f'data/test_kgs/{i}.pkl'
                    pkl_path_amr = f'data/test_amrs/{i}.pkl'
                    data = get_data_amg_plus_kg(pkl_path_kg, pkl_path_amr, retrieved_examples, answers, embeddings_dict, amr_data, args.amr_number_of_links, question_embedding)
                    data = data.to(device)
                    y = data.y

                # Calculate the scores
                outputs = model(data)
                question_embedding = torch.tensor(question_embedding).to(device)
                scores = torch.flatten(torch.matmul(outputs, question_embedding.t()))

                # Get the labels
                doc_indices_5 = torch.topk(scores, 5).indices
                doc_labels_5 = y[doc_indices_5]
                doc_indices_10 = torch.topk(scores, 10).indices
                doc_labels_10 = y[doc_indices_10]
                doc_indices_20 = torch.topk(scores, 20).indices
                doc_labels_20 = y[doc_indices_20]
                doc_indices_100 = torch.topk(scores, 100).indices
                doc_labels_100 = y[doc_indices_100]
                doc_labels_100 = torch.squeeze(doc_labels_100)
                ranks = torch.arange(1, 100 + 1).to(device)

                # Calculate the accuracy scores
                if torch.sum(doc_labels_5) != 0:
                    accuracy_5 += 1
                if torch.sum(doc_labels_10) != 0:
                    accuracy_10 += 1
                if torch.sum(doc_labels_20) != 0:
                    accuracy_20 += 1
                
                # Calculate the mmr scores
                rankings = 1 / (doc_labels_100 * ranks)
                rankings = torch.where(torch.isinf(rankings), torch.tensor(0.0), rankings)
                safe_coe = (1 / torch.sum(doc_labels_100))
                if torch.isinf(safe_coe):
                    safe_coe = 0
                mrr += safe_coe * torch.sum(rankings)

                # Calculate the mhits scores
                mhits_5 += safe_coe * torch.sum(doc_labels_5)
                mhits_10 += safe_coe * torch.sum(doc_labels_10)
                mhits_20 += safe_coe * torch.sum(doc_labels_20)

                wandb.log({
                    'accuracy_5': accuracy_5 / len(question_embeddings_test),
                    'accuracy_10': accuracy_10 / len(question_embeddings_test),
                    'accuracy_20': accuracy_20 / len(question_embeddings_test),
                    'mhits_5': mhits_5 / len(question_embeddings_test),
                    'mhits_10': mhits_10 / len(question_embeddings_test),
                    'mhits_20': mhits_20 / len(question_embeddings_test),
                    'mrr': mrr / len(question_embeddings_test),
                    'l2': config.l2
                })

        # Final score calculations...
        accuracy_5 = accuracy_5 / len(question_embeddings_test)
        accuracy_10 = accuracy_10 / len(question_embeddings_test)
        accuracy_20 = accuracy_20 / len(question_embeddings_test)
        mhits_5 = mhits_5 / len(question_embeddings_test)
        mhits_10 = mhits_10 / len(question_embeddings_test)
        mhits_20 = mhits_20 / len(question_embeddings_test)
        mrr = mrr / len(question_embeddings_test)

        # Print out each of the scores
        print(f'The accuracy top 5 is: {accuracy_5}')
        print(f'The accuracy top 10 is: {accuracy_10}')
        print(f'The accuracy top 20 is: {accuracy_20}')
        print(f'The mhits top 5 is: {mhits_5}')
        print(f'The mhits top 10 is: {mhits_10}')
        print(f'The mhits top 20 is: {mhits_20}')
        print(f'The MRR is: {mrr}')

wandb.agent(sweep_id, train_and_evaluate, count=9)