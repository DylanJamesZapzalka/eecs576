from fb_dpr_utils import has_answer
from tqdm import tqdm
from torch_geometric.data import Data
import torch

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
            match = has_answer(answers,retrieved_example)
            if match:
                exact_matches +=1
                break
    
    # Return the score
    score = exact_matches / len(question_embeddings)
    return score


def get_data(retrieved_examples, answers, nlp, link_type, number_of_links, num_dpr_samples):

    # Get node feature vectors
    x = torch.tensor(retrieved_examples['embeddings']).to(device)

    # Get the edge index
    edge_index = None
    if link_type == 'ssr':
        edge_index = get_edge_index_shared_spacy_relationships(retrieved_examples['text'], nlp, number_of_links, num_dpr_samples)
    elif link_type == 'se':
        edge_index = get_edge_index_shared_entities(retrieved_examples['text'], nlp, number_of_links, num_dpr_samples)
    else:
        raise Exception('Invalid value for link_type.')

    # Get the labels and create the Data object
    y = get_labels(retrieved_examples['text'], answers)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    return data


def get_edge_index_shared_spacy_relationships(retrieved_examples, nlp, number_of_links, num_dpr_samples):

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

    adjacency_matrix = torch.zeros((num_dpr_samples, num_dpr_samples))
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

    return torch.tensor(edge_index, dtype=torch.long)


def get_edge_index_shared_entities(retrieved_examples, nlp, number_of_links, num_dpr_samples):

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

    adjacency_matrix = torch.zeros((num_dpr_samples, num_dpr_samples))
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

    return torch.tensor(edge_index, dtype=torch.long)



def get_labels(retrieved_examples, answers):

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