from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
from datasets import load_dataset
import csv
import numpy as np
import ast
from tqdm import tqdm
from utils import get_exact_match_score, get_data
import constants
import argparse
import spacy  # version 3.5
from torch.nn import CrossEntropyLoss
from models import GCN, GAT, GraphSAGE
import torch
from torch_geometric.loader import DataLoader


# Make sure cuda is being used
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'The device I am using is: {device}')

# Get the necessary constants
NQ_TEST_FILE_NAME = constants.NQ_TEST_FILE_NAME
TRIVIA_TEST_FILE_NAME = constants.TRIVIA_TEST_FILE_NAME
DATASETS_DIR = constants.DATASETS_DIR


# Get arguments for experiments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", required=True, type=str, help='Dataset used in the experiments. Can be "nq" or "trivia".')
parser.add_argument("--train_num_samples", required=True, type=int, help='Number of samples used to train the model.')
parser.add_argument("--eval_num_samples", required=True, type=int, help='Number of samples used to evaluate the model.')
parser.add_argument("--link_type", required=True, type=str, help='Method that will be used to creat the graph. Can be "ssr" or "se".')
parser.add_argument("--number_of_links", required=True, type=int, help='Number of connections needed to create an edge for the reranking graph.')
parser.add_argument("--gnn_type", required=True, type=str, help='Type of GNN for the reranker. Can be "gcn", "gat", or "sage".')
parser.add_argument("--num_epochs", required=True, type=int, help='Number of epochs the GNN model will be trained over.')
parser.add_argument("--num_dpr_samples", required=True, type=int, help='Number of samples DPR will retrieve before the second reranking step.')
parser.add_argument("--num_eval_samples", default=10, type=int, help='Number of samples we evaluate over.')
parser.add_argument("--batch_size", default=8, type=int, help='Batch size for training the gnn.')
args = parser.parse_args()



# This loads in the wiki datasets automatically, which will be used to obtain all
# relevant documents via DPR. We are using the compressed version, which only
# requires around 3GB of memory at runtime, compared to 35GB for the 'exact',
# uncompressed version. This will make DPR slightly less accurate, but it is needed
# due to computational constraints. Importantly, this was derived from the multiset
# encoders, so it will work for both TrivaQA/NQ
wiki_dataset = load_dataset('wiki_dpr', 'psgs_w100.multiset.compressed', cache_dir=DATASETS_DIR)
wiki_dataset = wiki_dataset['train'] # There is only a train set

# Load in the pretrained context encoders over the multiset datasets
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(device).eval()
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
# Load in the pretrained question encoders over the multiset datasets
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base").to(device).eval()
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")

# Get the questions and answer columns
# I don't know how to disable comments... by default, it is '#', which causes an error
if args.dataset_name == 'trivia':
    test_set = np.loadtxt(TRIVIA_TEST_FILE_NAME, delimiter="\t", dtype=object, comments='@!$$@$@')
    questions_array = test_set[:, 0]
    answers_array = test_set[:, 1]
    tmp_answer_array = []
    tmp_questions_array = []
    num_of_errors = 0
    for i in range(len(answers_array)):
        try:
            tmp_answer_array.append(ast.literal_eval(answers_array[i]))
            tmp_questions_array.append(questions_array[i])
        except:
            num_of_errors += 1
    print(f'Had to discard {num_of_errors} samples due to processing errors')
    answers_array = tmp_answer_array
    questions_array = tmp_questions_array
elif args.dataset_name == 'nq':
    test_set = np.loadtxt(NQ_TEST_FILE_NAME, delimiter="\t", dtype=object, comments='@!$$@$@')
    questions_array = test_set[:, 0]
    answers_array = test_set[:, 1]
    answers_array = [ast.literal_eval(answers) for answers in answers_array]
else:
    raise Exception("Value provided for dataset argument is invalid...")



# Get embeddings for each of the questions
question_embeddings = []
for question in tqdm(questions_array, desc='Embedding questions'):
    question_tokens = q_tokenizer(question ,max_length=512,truncation=True,padding='max_length',return_tensors='pt').to(device)
    question_embedding = q_encoder(**question_tokens)
    question_embedding = question_embedding.pooler_output
    question_embedding = question_embedding.cpu().detach().numpy()
    question_embeddings.append(question_embedding)

train_questions_embeddings = question_embeddings[0:args.train_num_samples]
eval_questions_embeddings = question_embeddings[args.train_num_samples: args.train_num_samples + args.eval_num_samples]

train_answers_array = answers_array[0:args.train_num_samples]
eval_answers_array = answers_array[args.train_num_samples: args.train_num_samples + args.eval_num_samples]

# initialize language model
nlp = spacy.load("en_core_web_sm")

# add pipeline (declared through entry_points in setup.py)
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
ce_loss = torch.nn.CrossEntropyLoss()

# Obtain the dataset/dataloader
data_list = []
for i in tqdm(range(len(train_questions_embeddings)), desc='Creating dataset'):
    # Get question and answers
    question_embedding = train_questions_embeddings[i]
    answers = train_answers_array[i]
    # Get k nearest examples via DPR
    scores, retrieved_examples = wiki_dataset.get_nearest_examples('embeddings', question_embedding, k=args.num_dpr_samples)
    data= get_data(retrieved_examples, answers, nlp, args.link_type, args.number_of_links, args.num_dpr_samples)
    data_list.append(data)
data_loader = DataLoader(data_list, batch_size=args.batch_size)

# Start training the GNN
for i in tqdm(range(args.num_epochs), desc='Training...'):
    for batch in data_loader:

        # Get data
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)
        batch.edge_index = batch.edge_index.to(device)

        # Get loss
        outputs = model(batch)
        question_embedding = torch.tensor(question_embedding).to(device)
        scores = torch.matmul(outputs, question_embedding.t())
        loss = ce_loss(scores.t(), batch.y.t())

        # Perform backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




# Start the evaluation process
model.eval()
accuracy = 0
for i in tqdm(range(len(eval_questions_embeddings)), desc='Evaluating over each question/answer'):
    # Get question and answers
    question_embedding = eval_questions_embeddings[i]
    answers = eval_answers_array[i]
    # Get k nearest examples via DPR
    scores, retrieved_examples = wiki_dataset.get_nearest_examples('embeddings', question_embedding, k=args.num_dpr_samples)
    data = get_data(retrieved_examples, answers, nlp, args.link_type, args.number_of_links, args.num_dpr_samples)
    data = data.to(device)
    y = data.y

    outputs = model(data)
    question_embedding = torch.tensor(question_embedding).to(device)
    scores = torch.flatten(torch.matmul(outputs, question_embedding.t()))
    doc_indices = torch.topk(scores, args.num_eval_samples).indices
    doc_labels = y[doc_indices]
    if torch.sum(doc_labels) !=0:
        accuracy += 1

accuracy = accuracy / len(question_embeddings)
print(f'The accuracy is: {accuracy}')