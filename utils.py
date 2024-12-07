import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from transformers import pipeline

import constants
from fb_dpr_utils import has_answer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_exact_match_score(question_embeddings, answers_array, dataset, k):
    exact_matches = 0
    for i in tqdm(range(len(question_embeddings)), desc='Evaluating over each question/answer'):
        # Get question and answers
        question_embedding = question_embeddings[i]
        answers = answers_array[i]

        # Get k nearest examples via DPR
        scores, retrieved_examples = dataset.get_nearest_examples('embeddings', question_embedding, k=k)
        retrieved_examples = retrieved_examples['text']

        # Check each of the nearest passages for an exact match
        for retrieved_example in retrieved_examples:
            match = has_answer(answers, retrieved_example)
            if match:
                exact_matches += 1
                break

    # Return the score
    score = exact_matches / len(question_embeddings)
    return score


def get_data_kg_dpr(retrieved_examples, answers, nlp, link_type, number_of_links):
    # Get node feature vectors
    x = torch.tensor(retrieved_examples['embeddings']).to(device)

    # Get the edge index
    edge_index = None
    if link_type == 'ssr':
        edge_index = get_edge_index_shared_spacy_relationships(retrieved_examples['text'], nlp, number_of_links)
    elif link_type == 'se':
        edge_index = get_edge_index_shared_entities(retrieved_examples['text'], nlp, number_of_links)
    else:
        raise Exception('Invalid value for link_type.')

    # Get the labels and create the Data object
    y = get_labels_dpr(retrieved_examples['text'], answers)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    return data


def get_data_amr(retrieved_examples, answers, ctx_encoder, ctx_tokenizer, amr_number_of_links):
    # Get passage embeddings and node features of amr graphs
    passage_embeddings = []
    nodes_list = []
    for passage in retrieved_examples:
        text = passage['text']
        passage_tokens = ctx_tokenizer(text, max_length=512, truncation=True, padding='max_length',
                                       return_tensors='pt').to(device)
        passage_embedding = ctx_encoder(**passage_tokens)
        passage_embedding = passage_embedding.pooler_output
        passage_embedding = passage_embedding.cpu().detach().numpy()
        passage_embeddings.append(passage_embedding)

        # Get nodes and filter
        nodes = passage['nodes']
        filtered_nodes = [node for node in nodes if
                          len(node) > 3 and node != 'amr-unknown' and node != 'this' and node != 'person' and node != 'person' and node != 'name' and node != 'also' and node != 'multi-sentence']
        nodes_list.append(filtered_nodes)

    passage_embeddings = np.array(passage_embeddings)

    # Get node feature vectors
    x = torch.tensor(passage_embeddings)
    x = torch.squeeze(x)

    # Get the edge index
    edge_index = get_edge_index_amr(nodes_list, amr_number_of_links)

    # Get the labels and create the Data object
    y = get_labels(retrieved_examples, answers)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    return data


def get_data_kg(retrieved_examples, answers, embeddings_dict, subgraphs):
    # Get passage embeddings and node features of amr graphs
    passage_embeddings = []
    passage_texts = []
    pids = []
    for passage in retrieved_examples:
        text = passage['text']
        pid = passage['pid']
        passage_texts.append(text)
        passage_embedding = embeddings_dict[pid]
        passage_embeddings.append(passage_embedding)
        pids.append(pid)

    passage_embeddings = np.array(passage_embeddings)

    # Get node feature vectors
    x = torch.tensor(passage_embeddings, dtype=torch.float32)
    x = torch.squeeze(x)

    # Get the edge index
    edge_index = get_edge_index_shared_entities_kg(retrieved_examples, subgraphs)

    # Get the labels and create the Data object
    y = get_labels_aqa(pids, answers)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    return data

def get_data_kg_update_y(pkl_path, retrieved_examples, answers):
    # Get passage embeddings and node features of amr graphs
    passage_embeddings = []
    passage_texts = []
    pids = []
    for passage in retrieved_examples:
        pids.append(passage['pid'])

    # Get the labels and create the Data object
    y = get_labels_aqa(pids, answers)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    data.y = y
    return data


def get_data_amg_plus_kg(retrieved_examples, answers, nlp, kg_number_of_links, kg_link_type, amr_number_of_links,
                         ctx_encoder, ctx_tokenizer):
    # Get passage embeddings and node features of amr graphs
    passage_embeddings = []
    nodes_list = []
    passage_texts = []
    for passage in retrieved_examples:
        text = passage['text']
        passage_texts.append(text)
        passage_tokens = ctx_tokenizer(text, max_length=512, truncation=True, padding='max_length',
                                       return_tensors='pt').to(device)
        passage_embedding = ctx_encoder(**passage_tokens)
        passage_embedding = passage_embedding.pooler_output
        passage_embedding = passage_embedding.cpu().detach().numpy()
        passage_embeddings.append(passage_embedding)

        # Get nodes and filter
        nodes = passage['nodes']
        filtered_nodes = [node for node in nodes if
                          len(node) > 3 and node != 'amr-unknown' and node != 'this' and node != 'person' and node != 'person' and node != 'name' and node != 'also' and node != 'multi-sentence']
        nodes_list.append(filtered_nodes)

    passage_embeddings = np.array(passage_embeddings)

    # Get node feature vectors
    x = torch.tensor(passage_embeddings)
    x = torch.squeeze(x)

    # Get the edge index for kg
    edge_index_kg = None
    if kg_link_type == 'ssr':
        edge_index_kg = get_edge_index_shared_spacy_relationships(passage_texts, nlp, kg_number_of_links,
                                                                  return_list=True)
    elif kg_link_type == 'se':
        edge_index_kg = get_edge_index_shared_entities(passage_texts, nlp, kg_number_of_links, return_list=True)
    else:
        raise Exception('Invalid value for link_type.')

    # Get the edge index for amr graph
    edge_index_amr = get_edge_index_amr(nodes_list, amr_number_of_links, return_list=True)

    # Combine the kg and amr edge index
    edge_index = edge_index_kg + edge_index_amr
    edge_index = list(map(list, set(map(tuple, edge_index))))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Get the labels and create the Data object
    y = get_labels(retrieved_examples, answers)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    return data


def get_edge_index_amr(nodes_list, amr_number_of_links, return_list=False):
    edge_index = []
    for i in range(len(nodes_list)):
        array_1 = nodes_list[i]
        for j in range(len(nodes_list)):
            if i == j:
                continue
            array_2 = nodes_list[j]
            links = len(set(array_1).intersection(set(array_2)))
            if links >= amr_number_of_links:
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index.append([i, i])

    edge_index = list(map(list, set(map(tuple, edge_index))))  # Remove duplicates
    if return_list:
        return edge_index
    else:
        return torch.tensor(edge_index, dtype=torch.long)


def get_edge_index_shared_spacy_relationships(retrieved_examples, nlp, number_of_links, return_list=False):
    entities_array = []
    children_array = []
    for retrieved_example in retrieved_examples:
        doc = nlp(retrieved_example)

        entities = doc._.linkedEntities
        children = []
        entities_list = []
        for entity in entities:
            entities_list.append(entity.get_id())
            for collection in entity.get_super_entities():
                children.append(collection.get_id())
        entities_array.append(entities_list)
        children_array.append(children)

    edge_index = []
    for i in range(len(entities_array)):
        array_1 = entities_array[i]
        for j in range(len(children_array)):
            if i == j:
                continue
            array_2 = children_array[j]
            links = len(set(array_1).intersection(set(array_2)))
            if links >= number_of_links:
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index.append([i, i])

    edge_index = list(map(list, set(map(tuple, edge_index))))  # Remove duplicates
    if return_list:
        return edge_index
    else:
        return torch.tensor(edge_index, dtype=torch.long)


def get_edge_index_shared_entities_kg(retrieved_examples, subgraphs, return_list=False):
    seen_entities = defaultdict(set)
    for i, d in enumerate(retrieved_examples):
        pid = d['pid']
        kg = subgraphs[pid]
        for triplet in kg:
            seen_entities[triplet['head'].lower()].add(i)
            seen_entities[triplet['tail'].lower()].add(i)
    edge_index = []
    for entity, key_set in tqdm(seen_entities.items()):
        key_list = list(key_set)
        for i in range(len(key_list)):
            key_i = key_list[i]
            for j in range(i + 1, len(key_list)):
                key_j = key_list[j]
                edge_index.append((key_i, key_j))
                edge_index.append((key_j, key_i))
    if return_list:
        return edge_index
    else:
        return torch.tensor(edge_index, dtype=torch.long)


def get_edge_index_shared_entities(retrieved_examples, nlp, number_of_links, return_list=False):
    entities_array = []
    children_array = []
    for retrieved_example in retrieved_examples:
        doc = nlp(retrieved_example)

        entities = doc._.linkedEntities
        children = []
        entities_list = []
        for entity in entities:
            entities_list.append(entity.get_id())
        entities_array.append(entities_list)

    edge_index = []
    for i in range(len(entities_array)):
        array_1 = entities_array[i]
        for j in range(len(entities_array)):
            if i == j:
                continue
            array_2 = entities_array[j]
            links = len(set(array_1).intersection(set(array_2)))
            if links >= number_of_links:
                edge_index.append([i, j])
                edge_index.append([j, i])
        edge_index.append([i, i])

    edge_index = list(map(list, set(map(tuple, edge_index))))  # Remove duplicates
    if return_list:
        return edge_index
    else:
        return torch.tensor(edge_index, dtype=torch.long)


def get_labels_dpr(retrieved_examples, answers):
    labels = torch.zeros(len(retrieved_examples), dtype=torch.float)

    # Check each of the nearest passages for an exact match
    for i in range(len(retrieved_examples)):
        # Get question and answers
        retrieved_example = retrieved_examples[i]
        match = has_answer(answers, retrieved_example)
        if match:
            labels[i] = 1
            continue
    labels = torch.unsqueeze(labels, dim=1)
    return labels


def get_labels_aqa(pids, answers):
    labels = torch.zeros(len(pids), dtype=torch.float)

    # Check each of the nearest passages for an exact match
    for i in range(len(pids)):
        # Get question and answers
        if pids[i] in answers:
            labels[i] = 1
    labels = torch.unsqueeze(labels, dim=1)
    return labels

def get_labels(retrieved_examples, answers):
    labels = torch.zeros(len(retrieved_examples), dtype=torch.float)

    # Check each of the nearest passages for an exact match
    for i in range(len(retrieved_examples)):
        # Get question and answers
        retrieved_example = retrieved_examples[i]['text']
        match = has_answer(answers, retrieved_example)
        if match:
            labels[i] = 1
            continue
    labels = torch.unsqueeze(labels, dim=1)
    return labels