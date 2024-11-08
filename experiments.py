
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
from datasets import load_dataset
import pandas as pd
from pandas import read_csv
import csv
import numpy as np
import ast
from tqdm import tqdm
from utils import get_exact_match_score
import constants
import argparse

# Make sure cuda is being used
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'The device I am using is: {device}')

# Get the necessary constants
NQ_TEST_FILE_NAME = constants.NQ_TEST_FILE_NAME
TRIVIA_TEST_FILE_NAME = constants.TRIVIA_TEST_FILE_NAME
DATASETS_DIR = constants.DATASETS_DIR


# Get arguments for experiments
    
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", required=True)
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


# Evaluate vanilla dpr
k_5_score = get_exact_match_score(question_embeddings, answers_array, wiki_dataset, 5)
print(f'Exact match score when k=5 is: {k_5_score}')
k_10_score = get_exact_match_score(question_embeddings, answers_array, wiki_dataset, 10)
print(f'Exact match score when k=10 is: {k_10_score}')
k_50_score = get_exact_match_score(question_embeddings, answers_array, wiki_dataset, 50)
print(f'Exact match score when k=50 is: {k_50_score}')