from fb_dpr_utils import has_answer
from tqdm import tqdm
from torch_geometric.data import Data
import torch_geometric
import torch
import numpy as np
import time

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

def get_data_amr(retrieved_examples, answers, embeddings_dict, amr_data, amr_number_of_links):

    # Get passage embeddings and node features of amr graphs
    passage_embeddings = []
    passage_texts = []
    nodes_list = []
    pids = []
    for passage in retrieved_examples:

        text = passage['text']
        pid = passage['pid']
        passage_texts.append(text)
        passage_embedding = embeddings_dict[pid]
        passage_embeddings.append(passage_embedding)

        # Get nodes and filter
        nodes = amr_data[pid]['nodes']
        # print(nodes)
        filtered_nodes = [node for node in nodes if node is not None and len(node) > 3 and node != 'amr-unknown' and node != 'this' and node != 'person' and node != 'person' and node != 'name' and node != 'also' and node != 'multi-sentence']
        nodes_list.append(filtered_nodes)
        pids.append(pid)
    
    passage_embeddings = np.array(passage_embeddings)

    # Get node feature vectors
    x = torch.tensor(passage_embeddings, dtype=torch.float32)
    x = torch.squeeze(x)

    # Get the edge index
    edge_index = get_edge_index_amr(nodes_list, amr_number_of_links)

    # Get the labels and create the Data object
    y = get_labels(pids, answers)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    return data

# def get_data_amr(retrieved_examples, answers, ctx_encoder, ctx_tokenizer, amr_number_of_links):

#     timings = {}
#     start_total = time.time()
    
#     passage_embeddings = []
#     nodes_list = []
    
#     for idx, passage in enumerate(retrieved_examples):
#         print(f"\nProcessing passage {idx + 1}/{len(retrieved_examples)}")
        
#         # Line 1
#         start = time.time()
#         text = passage['text']
#         # timings['line1_text_extraction'] = time.time() - start
#         # print(f"Line 1 - Text extraction: {timings['line1_text_extraction']:.4f} seconds")
        
#         # Line 2-3
#         start = time.time()
#         passage_tokens = ctx_tokenizer(text, max_length=512, truncation=True, 
#                                      padding='max_length', return_tensors='pt').to(device)
#         # timings['line2_tokenization'] = time.time() - start
#         # print(f"Line 2 - Tokenization: {timings['line2_tokenization']:.4f} seconds")
        
#         # Line 4
#         start = time.time()
#         passage_embedding = ctx_encoder(**passage_tokens)
#         # timings['line4_encoding'] = time.time() - start
#         # print(f"Line 4 - Encoding: {timings['line4_encoding']:.4f} seconds")
        
#         # Line 5
#         start = time.time()
#         passage_embedding = passage_embedding.pooler_output
#         # timings['line5_pooler'] = time.time() - start
#         # print(f"Line 5 - Pooler output: {timings['line5_pooler']:.4f} seconds")
        
#         # Line 6
#         start = time.time()
#         passage_embedding = passage_embedding.cpu().detach().numpy()
#         # timings['line6_to_numpy'] = time.time() - start
#         # print(f"Line 6 - To numpy: {timings['line6_to_numpy']:.4f} seconds")
        
#         # Line 7
#         start = time.time()
#         passage_embeddings.append(passage_embedding)
#         # timings['line7_append'] = time.time() - start
#         # print(f"Line 7 - Append: {timings['line7_append']:.4f} seconds")
        
#         # Line 8
#         start = time.time()
#         nodes = passage['nodes']
#         # timings['line8_get_nodes'] = time.time() - start
#         # print(f"Line 8 - Get nodes: {timings['line8_get_nodes']:.4f} seconds")
        
#         # Line 9
#         start = time.time()
#         filtered_nodes = [node for node in nodes if len(node) > 3 and 
#                          node != 'amr-unknown' and node != 'this' and 
#                          node != 'person' and node != 'name' and 
#                          node != 'also' and node != 'multi-sentence']
#         # timings['line9_filter_nodes'] = time.time() - start
#         # print(f"Line 9 - Filter nodes: {timings['line9_filter_nodes']:.4f} seconds")
        
#         # Line 10
#         start = time.time()
#         nodes_list.append(filtered_nodes)
#         # timings['line10_append_nodes'] = time.time() - start
#         # print(f"Line 10 - Append nodes: {timings['line10_append_nodes']:.4f} seconds")
    
#     # Lines 11-12
#     start = time.time()
#     passage_embeddings = np.array(passage_embeddings)
#     # timings['line11_12_to_array'] = time.time() - start
#     # print(f"\nLines 11-12 - To array: {timings['line11_12_to_array']:.4f} seconds")
    
#     # Line 13
#     start = time.time()
#     x = torch.tensor(passage_embeddings)
#     # timings['line13_to_tensor'] = time.time() - start
#     # print(f"Line 13 - To tensor: {timings['line13_to_tensor']:.4f} seconds")
    
#     # Line 14
#     start = time.time()
#     x = torch.squeeze(x)
#     # timings['line14_squeeze'] = time.time() - start
#     # print(f"Line 14 - Squeeze: {timings['line14_squeeze']:.4f} seconds")
    
#     # Line 15
#     start = time.time()
#     edge_index = get_edge_index_amr(nodes_list, amr_number_of_links)
#     # timings['line15_edge_index'] = time.time() - start
#     # print(f"Line 15 - Edge index: {timings['line15_edge_index']:.4f} seconds")
    
#     # Line 16
#     start = time.time()
#     y = get_labels(retrieved_examples, answers)
#     # timings['line16_get_labels'] = time.time() - start
#     # print(f"Line 16 - Get labels: {timings['line16_get_labels']:.4f} seconds")
    
#     # Line 17
#     start = time.time()
#     data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
#     # timings['line17_create_data'] = time.time() - start
#     # print(f"Line 17 - Create data: {timings['line17_create_data']:.4f} seconds")
    
#     # total_time = time.time() - start_total
#     # print(f"\nTotal execution time: {total_time:.4f} seconds")
    
#     # # Calculate and print percentage of total time for each operation
#     # print("\nPercentage of total time for each operation:")
#     # for operation, timing in timings.items():
#         # percentage = (timing / total_time) * 100
#         # print(f"{operation}: {percentage:.2f}%")

#     return data


def get_data_kg(retrieved_examples, answers, nlp, number_of_links, link_type, ctx_encoder, ctx_tokenizer):

    # Get passage embeddings and node features of amr graphs
    passage_embeddings = []
    passage_texts = []
    for passage in retrieved_examples:

        text = passage['text']
        passage_texts.append(text)
        passage_tokens = ctx_tokenizer(text ,max_length=512,truncation=True,padding='max_length',return_tensors='pt').to(device)
        passage_embedding = ctx_encoder(**passage_tokens)
        passage_embedding = passage_embedding.pooler_output
        passage_embedding = passage_embedding.cpu().detach().numpy()
        passage_embeddings.append(passage_embedding)
    
    passage_embeddings = np.array(passage_embeddings)

    # Get node feature vectors
    x = torch.tensor(passage_embeddings)
    x = torch.squeeze(x)

    # Get the edge index
    edge_index = None
    if link_type == 'ssr':
        edge_index = get_edge_index_shared_spacy_relationships(passage_texts, nlp, number_of_links)
    elif link_type == 'se':
        edge_index = get_edge_index_shared_entities(passage_texts, nlp, number_of_links)
    else:
        raise Exception('Invalid value for link_type.')

    # Get the labels and create the Data object
    y = get_labels(retrieved_examples, answers)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    return data


def get_data_amg_plus_kg(retrieved_examples, answers, nlp, kg_number_of_links, kg_link_type, amr_number_of_links, ctx_encoder, ctx_tokenizer):

    # Get passage embeddings and node features of amr graphs
    passage_embeddings = []
    nodes_list = []
    passage_texts = []
    for passage in retrieved_examples:

        text = passage['text']
        passage_texts.append(text)
        passage_tokens = ctx_tokenizer(text ,max_length=512,truncation=True,padding='max_length',return_tensors='pt').to(device)
        passage_embedding = ctx_encoder(**passage_tokens)
        passage_embedding = passage_embedding.pooler_output
        passage_embedding = passage_embedding.cpu().detach().numpy()
        passage_embeddings.append(passage_embedding)

        # Get nodes and filter
        nodes = passage['nodes']
        filtered_nodes = [node for node in nodes if len(node) > 3 and node != 'amr-unknown' and node != 'this' and node != 'person' and node != 'person' and node != 'name' and node != 'also' and node != 'multi-sentence']
        nodes_list.append(filtered_nodes)
    
    passage_embeddings = np.array(passage_embeddings)

    # Get node feature vectors
    x = torch.tensor(passage_embeddings)
    x = torch.squeeze(x)

    # Get the edge index for kg
    edge_index_kg = None
    if kg_link_type == 'ssr':
        edge_index_kg = get_edge_index_shared_spacy_relationships(passage_texts, nlp, kg_number_of_links, return_list=True)
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

    edge_index = list(map(list, set(map(tuple, edge_index)))) # Remove duplicates
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


    edge_index = list(map(list, set(map(tuple, edge_index)))) # Remove duplicates
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

    edge_index = list(map(list, set(map(tuple, edge_index)))) # Remove duplicates
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


def get_labels(pids, answers):

    labels = torch.zeros(len(pids), dtype=torch.float)
  
    # Check each of the nearest passages for an exact match
    for i in range(len(pids)):
        # Get question and answers
        if pids[i] in answers:
            labels[i] = 1
    labels = torch.unsqueeze(labels, dim=1)
    return labels