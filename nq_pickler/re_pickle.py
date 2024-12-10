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
from torch_geometric.data import Data

# File exists, load the data
with open('/home/dylanz/eecs576/pickled/kg_data.pkl', 'rb') as f:
    data_list = pickle.load(f)

# File exists, load the data
with open('/home/dylanz/eecs576/pickled/question_data.pkl', 'rb') as f:
    question_list = pickle.load(f)

new_data_list = []
for i in tqdm(range(len(data_list)), desc='Training the reranker...'):
    data = data_list[i]
    question = torch.tensor(question_list[i])
    print(question.shape)
    new_data_list.append(Data(data.x, data.edge_index, y=data.y, question_embedding=question))
print(new_data_list)
# Saving the objects
with open('/home/dylanz/eecs576/pickled/kg_data_new.pkl', 'wb') as f:
    pickle.dump(new_data_list, f)  # Serialize and save the list of objects