from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
from datasets import load_dataset
import csv
import numpy as np
import ast
from tqdm import tqdm
from utils import get_exact_match_score, get_data_kg, get_data_amr, get_data_kg_dpr, get_data_amg_plus_kg
import constants
import argparse
import spacy
from torch.nn import CrossEntropyLoss
from models import GCN, GAT, GraphSAGE
import torch
from torch_geometric.loader import DataLoader
from amr_bart_utils import load_data
import pickle
import os

# Make sure cuda is being used
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'The device I am using is: {device}')

# Get the necessary constants
NQ_TEST_FILE_NAME = constants.NQ_TEST_FILE_NAME
TRIVIA_TEST_FILE_NAME = constants.TRIVIA_TEST_FILE_NAME
AMR_NQ_TEST_FILE_NAME = constants.AMR_NQ_TEST_FILE_NAME
AMR_NQ_TRAIN_FILE_NAME = constants.AMR_NQ_TRAIN_FILE_NAME
DATASETS_DIR = constants.DATASETS_DIR


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

amr_nq_data_train = load_data(AMR_NQ_TRAIN_FILE_NAME, args.train_num_samples)
questions_array_train = [example['question'] for example in amr_nq_data_train]
answers_array_train = [example['answers'] for example in amr_nq_data_train]

amr_nq_data_test = load_data(AMR_NQ_TEST_FILE_NAME, args.test_num_samples)
questions_array_test = [example['question'] for example in amr_nq_data_test]
answers_array_test = [example['answers'] for example in amr_nq_data_test]


# Check if the file exists
if os.path.exists('/home/dylanz/eecs576/pickled/question_data.pkl'):
    # File exists, load the data
    with open('/home/dylanz/eecs576/pickled/question_data.pkl', 'rb') as f:
        data_list = pickle.load(f)
else:
    # Saving the objects
    with open('/home/dylanz/eecs576/pickled/question_data.pkl', 'wb') as f:
        pickle.dump(question_embeddings_train, f)  # Serialize and save the list of objects


# Obtain the dataset/dataloader for both train and test
if args.model_name == 'kg':


    # Check if the file exists
    if os.path.exists('/home/dylanz/eecs576/pickled/kg_data.pkl'):
        # File exists, load the data
        with open('/home/dylanz/eecs576/pickled/kg_data.pkl', 'rb') as f:
            data_list = pickle.load(f)
    else:

        data_list = []
        for i in tqdm(range(len(question_embeddings_train)), desc='Creating kg dataset'):
            # Get question and answers
            answers = answers_array_train[i]
            # Get k nearest examples via DPR
            retrieved_examples = amr_nq_data_train[i]['ctxs']
            data = get_data_kg(retrieved_examples, answers, nlp, args.kg_number_of_links, args.kg_link_type, ctx_encoder, ctx_tokenizer)
            data_list.append(data)

        # Saving the objects
        with open('/home/dylanz/eecs576/pickled/kg_data.pkl', 'wb') as f:
            pickle.dump(data_list, f)  # Serialize and save the list of objects


elif args.model_name == 'amr':

    # Check if the file exists
    if os.path.exists('/home/dylanz/eecs576/pickled/amr_data_new.pkl'):
        # File exists, load the data
        with open('/home/dylanz/eecs576/pickled/amr_data_new.pkl', 'rb') as f:
            data_list = pickle.load(f)

    else:

        data_list = []
        for i in tqdm(range(0, args.train_num_samples), desc='Creating dataset'):
            # Get question and answers
            answers = answers_array_train[i]
            # Get 100 nearest examples via DPR
            retrieved_examples = amr_nq_data_train[i]['ctxs']
            data = get_data_amr(retrieved_examples, answers, ctx_encoder, ctx_tokenizer, args.amr_number_of_links)
            data_list.append(data)

        # Saving the objects
        with open('/home/dylanz/eecs576/pickled/amr_data_new.pkl', 'wb') as f:
            pickle.dump(data_list, f)  # Serialize and save the list of objects


elif args.model_name == 'amr+kg':

    # Check if the file exists
    if os.path.exists('/home/dylanz/eecs576/pickled/amr_kg_data.pkl'):
        # File exists, load the data
        with open('/home/dylanz/eecs576/pickled/amr_kg_data.pkl', 'rb') as f:
            data_list = pickle.load(f)

    else:

        data_list = []
        for i in tqdm(range(len(question_embeddings_train)), desc='Creating amr+kg dataset'):
            # Get question and answers
            answers = answers_array_train[i]
            # Get k nearest examples via DPR
            retrieved_examples = amr_nq_data_train[i]['ctxs']
            data = get_data_amg_plus_kg(retrieved_examples, answers, nlp, args.kg_number_of_links, args.kg_link_type, args.amr_number_of_links, ctx_encoder, ctx_tokenizer)
            data_list.append(data)

        # Saving the objects
        with open('/home/dylanz/eecs576/pickled/amr_kg_data.pkl', 'wb') as f:
            pickle.dump(data_list, f)  # Serialize and save the list of objects