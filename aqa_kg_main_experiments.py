import argparse
import os
import pickle
import pprint

import spacy
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

import constants
from amr_bart_utils import load_data_aqa, load_data_aqa_val
from models import GCN, GAT, GraphSAGE
from utils import get_data_kg_update_y

# Make sure cuda is being used
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'The device I am using is: {device}')

# Get the necessary constants
NQ_TEST_FILE_NAME = constants.NQ_TEST_FILE_NAME
TRIVIA_TEST_FILE_NAME = constants.TRIVIA_TEST_FILE_NAME
AMR_NQ_TEST_FILE_NAME = constants.AMR_NQ_TEST_FILE_NAME
AMR_NQ_TRAIN_FILE_NAME = constants.AMR_NQ_TRAIN_FILE_NAME
DATASETS_DIR = constants.DATASETS_DIR
AQA_TEST_FILE_NAME = constants.AQA_TEST_FILE_NAME
AQA_TRAIN_FILE_NAME = constants.AQA_TRAIN_FILE_NAME
AQA_VAL_FILE_NAME = constants.AQA_VAL_FILE_NAME
AQA_VAL_ANS = constants.AQA_VAL_ANS

# Get arguments for experiments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="kg",
                    help='Model used in the experiments. Can be "kg", "amr" or "amr+kg"')
parser.add_argument("--train_num_samples", type=int, default=None, help='Number of samples used to train the model.')
parser.add_argument("--test_num_samples", type=int, default=None, help='Number of samples used to test the model.')
parser.add_argument("--kg_link_type", type=str,
                    help='Method that will be used to creat the graph. Can be "ssr" or "se".')
parser.add_argument("--kg_number_of_links", type=int,
                    help='Number of connections needed to create an edge for the reranking graph.')
parser.add_argument("--amr_number_of_links", type=int,
                    help='Number of connections needed to create an edge for the reranking graph.')
parser.add_argument("--gnn_type", type=str, default="gcn",
                    help='Type of GNN for the reranker. Can be "gcn", "gat", or "sage".')
parser.add_argument("--num_epochs", type=int, default=20, help='Number of epochs the GNN model will be trained over.')
parser.add_argument("--batch_size", default=8, type=int, help='Batch size for training the gnn.')
parser.add_argument("--weight_decay", default=1e-1, type=int, help='Batch size for training the gnn.')

args = parser.parse_args()

# Load in the pretrained context encoders over the multiset datasets
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(device).eval()
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
# Load in the pretrained question encoders over the multiset datasets
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base").to(device).eval()
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")

aqa_data_train = load_data_aqa(AQA_TRAIN_FILE_NAME, args.train_num_samples)
questions_array_train = [example['question'] for example in aqa_data_train]
answers_array_train = [example['pids'] for example in aqa_data_train]

aqa_data_test = load_data_aqa_val(AQA_VAL_FILE_NAME, AQA_VAL_ANS, args.test_num_samples)
questions_array_test = [example['question'] for example in aqa_data_test]
answers_array_test = [example['pids'] for example in aqa_data_test]

# Get embeddings to each of the questions
if os.path.exists("question_embeddings_train.pickle"):
    with open('question_embeddings_train.pickle', 'rb') as handle:
        question_embeddings_train = pickle.load(handle)
else:
    question_embeddings_train = []
    for question in tqdm(questions_array_train, desc='Embedding questions'):
        question_tokens = q_tokenizer(question, max_length=512, truncation=True, padding='max_length',
                                      return_tensors='pt').to(device)
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
        question_tokens = q_tokenizer(question, max_length=512, truncation=True, padding='max_length',
                                      return_tensors='pt').to(device)
        question_embedding = q_encoder(**question_tokens)
        question_embedding = question_embedding.pooler_output
        question_embedding = question_embedding.cpu().detach().numpy()
        question_embeddings_test.append(question_embedding)
    with open('question_embeddings_test.pickle', 'wb') as handle:
        pickle.dump(question_embeddings_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Iniitialize the language model
nlp = spacy.load("en_core_web_sm")

# Add the spacey entity link pipeline
nlp.add_pipe("entityLinker", last=True)

# Get the reranker GNN
if args.gnn_type == 'gcn':
    model = GCN().to(device)
elif args.gnn_type == 'gat':
    model = GAT().to(device)
elif args.gnn_type == 'sage':
    model = GraphSAGE().to(device)
else:
    raise Exception("Invalid value for gnn_type.")

# Get the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=config.l2)
ce_loss = torch.nn.CrossEntropyLoss()

# Obtain the dataset/dataloader for both train and test
# Define the path for the pickle file
pickle_file_path = 'data_loader_train.pkl'

# Check if the pickle file exists
if os.path.exists(pickle_file_path):
    # Load the existing pickle file
    with open(pickle_file_path, 'rb') as f:
        data_loader_train = pickle.load(f)
    print("Loaded data_loader_train from the pickle file.")
else:
    data_list = []
    for i in tqdm(range(len(question_embeddings_train)), desc='Creating kg dataset'):
        # Get question and answers
        question_embedding = question_embeddings_train[i]
        answers = answers_array_train[i]
        # Get k nearest examples via DPR
        retrieved_examples = aqa_data_train[i]['retrieved_papers']
        pkl_path = f'data/train_kgs/{i}.pkl'
        data = get_data_kg_update_y(pkl_path, retrieved_examples, answers, question_embedding)
        data_list.append(data)

    data_loader_train = DataLoader(data_list, batch_size=args.batch_size)

    # Save the data_loader_train to a pickle file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(data_loader_train, f)
    print("Saved data_loader_train to a pickle file.")

# Start training the GNN
num_edge_indices = 0
model.train()
for i in tqdm(range(args.num_epochs), desc='Training the reranker...'):
    epoch_loss = 0

    for batch in data_loader_train:
        # Get data
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)
        batch.edge_index = batch.edge_index.to(device)
        num_edge_indices += batch.edge_index.shape[1]
        question_embedding = batch.question_embedding

        # Get loss
        outputs = model(batch)
        outputs = torch.split(outputs, 100)
        outputs = torch.stack(outputs, dim=0)

        question_embedding = torch.tensor(question_embedding).to(device)
        scores = torch.matmul(outputs, question_embedding.transpose(1, 2)).squeeze(-1)
        loss = ce_loss(scores.view(-1), batch.y)
        # Perform backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

print(f'The average number of edge indices per graph is: {num_edge_indices / (len(questions_array_train) * args.num_epochs)}')

torch.save(model.state_dict(), f"KG_{args.gnn_type}_RerankerModel{args.num_epochs}Epochs.pth")

print(
    f'The average number of edge indices per graph is: {num_edge_indices / (len(aqa_data_train) * args.num_epochs)}')

# Start the evaluation process
model.eval()
accuracy_5 = 0
accuracy_10 = 0
accuracy_20 = 0
mhits_5 = 0
mhits_10 = 0
mhits_20 = 0
mrr = 0

for i in tqdm(range(len(question_embeddings_test)), desc='Evaluating over each question/answer'):
    # Get question and answers
    question_embedding = question_embeddings_test[i]
    answers = answers_array_test[i]

    # Get 100 nearest examples via DPR
    retrieved_examples = aqa_data_test[i]['retrieved_papers']
    pkl_path = f'data/test_kgs/{i}.pkl'
    data = get_data_kg_update_y(pkl_path, retrieved_examples, answers, question_embedding)
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
