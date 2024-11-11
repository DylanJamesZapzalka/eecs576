from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
from datasets import load_dataset
import csv
import numpy as np
import ast
from tqdm import tqdm
from utils import get_exact_match_score
import constants
import argparse
import spacy  # version 3.5


# Make sure cuda is being used
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'The device I am using is: {device}')

# Get the necessary constants
NQ_TEST_FILE_NAME = constants.NQ_TEST_FILE_NAME
TRIVIA_TEST_FILE_NAME = constants.TRIVIA_TEST_FILE_NAME
DATASETS_DIR = constants.DATASETS_DIR

k=100

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

dataset_name = 'trivia'
# Get the questions and answer columns
# I don't know how to disable comments... by default, it is '#', which causes an error
if dataset_name == 'trivia':
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
elif dataset_name == 'nq':
    test_set = np.loadtxt(NQ_TEST_FILE_NAME, delimiter="\t", dtype=object, comments='@!$$@$@')
    questions_array = test_set[:, 0]
    answers_array = test_set[:, 1]
    answers_array = [ast.literal_eval(answers) for answers in answers_array]
else:
    raise Exception("Value provided for dataset argument is invalid...")


# Get embeddings for each of the questions
question_embeddings = []
for question in tqdm(questions_array[0:100], desc='Embedding questions'):
    question_tokens = q_tokenizer(question ,max_length=512,truncation=True,padding='max_length',return_tensors='pt').to(device)
    question_embedding = q_encoder(**question_tokens)
    question_embedding = question_embedding.pooler_output.detach()
    question_embeddings.append(question_embedding)


# initialize language model
nlp = spacy.load("en_core_web_sm")

# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    set(a).intersection(b)

for i in tqdm(range(len(question_embeddings)), desc='Evaluating over each question/answer'):
    # Get question and answers
    question_embedding = question_embeddings[i]
    # Get k nearest examples via DPR
    scores, retrieved_examples = wiki_dataset.get_nearest_examples('embeddings', question_embedding.cpu().numpy(), k=k)
    retrieved_examples = retrieved_examples['text']
    
    # Check each of the nearest passages for an exact match
    
    entities_matrix = []
    for retrieved_example in retrieved_examples:
        doc = nlp(retrieved_example)
        entities_list = []
        entitiels_label = []
        # returns all entities in the whole document
        entities = doc._.linkedEntities
        entities_list = []
        for entity in entities:
            # Get entity tokens
            entities_list.append(entity.get_label())
        entities_matrix.append(entities_list)


    adjacency_matrix = torch.zeros((k, k))

    for i in range(k):
        child_array_1 = entities_matrix[i]
        for j in range(k):
            if i == j:
                continue
            child_array_2 = entities_matrix[j]
            # linked = common_member(child_array_1, child_array_2)
            linked = len(set(child_array_1).intersection(set(child_array_2)))
            if linked >=4:
                adjacency_matrix[i,j] = 1

    print(torch.sum(adjacency_matrix))
