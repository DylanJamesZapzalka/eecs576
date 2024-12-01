import json
import random
import regex, string, re
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import ZeroPad2d

import unicodedata
import copy
import logging

"""
A bunch of classes and functions from the repo / paper used to generate the AMR graphs:
https://github.com/wangcunxiang/Graph-aS-Tokens/tree/main
https://aclanthology.org/2023.findings-acl.131.pdf
THIS CODE WAS FULLY OBTAINED FROM THE AFOREMENTIONED REPOs / PAPERS
"""


def has_answer1(answers, passage, tokenizer):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    passage = white_space_fix(remove_articles(remove_punc(lower(passage.strip()))))
    for a in answers:
        a_new = white_space_fix(remove_articles(remove_punc(lower(a.strip()))))
        if a_new in passage:
            return True
    return False


class Tokens(object):
    """A class to represent a list of tokenized text."""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]

        # Concatenate into strings
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )
        if len(kwargs.get("annotators", {})) > 0:
            logger.warning(
                "%s only tokenizes! Skipping annotators: %s" % (type(self).__name__, kwargs.get("annotators"))
            )
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append(
                (
                    token,
                    text[start_ws:end_ws],
                    span,
                )
            )
        return Tokens(data, self.annotators)


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def has_answer2(answers, text, tokenizer, match_type='string') -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False

#
# if __name__ == '__main__':
#     tok_opts = {}
#     tokenizer = SimpleTokenizer(**tok_opts)
#     answers = ['answer1', 'answer2']
#     print(has_answer(answers, p['title'] + ' . ' + p['text'], tokenizer, 'string')) # p is a passage in 'ctxs'

check_answers = has_answer2

def load_data(data_path=None, num_samples=4000):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in tqdm(enumerate(data), desc='Loading: '):
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
        if len(examples) >= num_samples:
            break
    return examples

def cal_metrics(ranks, mhits_bar=10):
    mrr = 0
    mhit = 0
    top5_tmp = 0
    top10_tmp = 0.
    top20_tmp = 0
    for rank in ranks:
        mrr += 1 / (rank + 1) / len(ranks)
        if rank < mhits_bar:
            mhit += 1
        if rank < 5:
            top5_tmp = 1
        if rank < 10:
            top10_tmp = 1
        if rank < 20:
            top20_tmp = 1
    return mrr, mhit, top5_tmp, top10_tmp, top20_tmp

def write_pos_ids():
    data_path = "../{}/{}.json"
    datasets = ['TQ', 'NQ']
    splits = ['dev', 'test']

    for dataset in datasets:
        for split in splits:
            examples = []
            data = load_data(data_path.format(dataset, split))
            tmp_tknzir = SimpleTokenizer()
            for d in tqdm(data, desc='Processing: '):
                ex = dict()
                ex['question'] = d['question']
                ex['answers'] = d['answers']
                # retrieved_ids = [ctx['id'] for ctx in d['ctxs']]
                golden_ids = [ctx['id'] for ctx in d['ctxs'] if
                              check_answers(d['answers'], ctx['title'] + '.' + ctx['text'], tmp_tknzir)]
                ex['positive_ids'] = golden_ids
                ex['positive_ids'] = golden_ids
                examples.append(ex)
            fout = open('positives/{}_{}_positives.json'.format(dataset, split), 'w', encoding='utf8')
            json.dump(examples, fout, indent=1)

def get_rerank_examples_odqa(args, data_path, status):
    data = load_data(data_path)
    examples = []
    mrrs = 0.
    mhits = 0.
    top5 = 0.
    top10 = 0.
    top20 = 0.
    tmp_tknzir = SimpleTokenizer()
    for d in tqdm(data, desc='Processing: '):
        ex = dict()
        ex['question'] = d['question']
        ex['answers'] = d['answers']
        retrieved_ids = [ctx['id'] for ctx in d['ctxs']]

        if status == 'eval':
            golden_ids = [ctx['id'] for ctx in d['ctxs'] if check_answers(d['answers'], ctx['title'] + '.' + ctx['text'], tmp_tknzir)]
            # if len(golden_ids) == 0 or len(golden_ids) >= 50:
            #     continue
            ex['positive_ids'] = golden_ids
            ranks = [retrieved_ids.index(positive_id) for positive_id in golden_ids]
            mrr, mhit, top5_tmp, top10_tmp, top20_tmp = cal_metrics(ranks, args.mhits_bar)
            top5 += top5_tmp
            top10 += top10_tmp
            top20 += top20_tmp
            mrrs += mrr
            if len(ranks) > 0:
                mhits += (mhit / len(ranks))

        # get negative passages
        if status == 'train':
            golden_ids = [ctx['id'] for ctx in d['ctxs'] if check_answers(d['answers'], ctx['title'] + '.' + ctx['text'], tmp_tknzir)]
            if len(golden_ids) == 0:
                continue
            ex['positive_ids'] = golden_ids
            non_golden_ids = [ctx['id'] for ctx in d['ctxs'] if ctx['id'] not in golden_ids]
            if len(non_golden_ids) < args.num_negative_psg:
                    continue
            ex['negative_ids'] = non_golden_ids
        else:
            ex['retrieved_ids'] = retrieved_ids
        examples.append(ex)

    if status == 'eval':
        print('DPR results:')
        print(f'Num Examples {len(examples)}')
        print(f'MRR: {round(100 * mrrs / len(examples), 1)}')
        print(f'MHits@{args.mhits_bar}: {round(100 * mhits / len(examples), 1)}')
        print(f'TOP5: {round(100 * top5 / len(examples), 1)}')
        print(f'TOP10: {round(100 * top10 / len(examples), 1)}')
        print(f'TOP20: {round(100 * top20 / len(examples), 1)}')
    return examples

def get_rerank_examples_odqa_amr(args, data_path, status):
    data = load_data(data_path)
    examples = []
    mrrs = 0.
    mhits = 0.
    top5 = 0.
    top10 = 0.
    top20 = 0.
    tmp_tknzir = SimpleTokenizer()
    node_lens = []
    edge_lens = []
    for it, d in tqdm(enumerate(data), desc='Processing: '):
        ex = dict()
        ex['question'] = d['question']
        ex['answers'] = d['answers']
        retrieved_ids = [ctx['id'] for ctx in d['ctxs']]

        if status == 'train' or status == 'eval':
            golden_ids = [ctx['id'] for ctx in d['ctxs'] if check_answers(d['answers'], ctx['title'] + ctx['text'], tmp_tknzir)]
            if status == 'train' and (len(golden_ids) == 0 or len(golden_ids)>50):
                continue
            non_golden_ids = [ctx['id'] for ctx in d['ctxs'] if ctx['id'] not in golden_ids]
            if status == 'train' and len(non_golden_ids) <= args.num_negative_psg:
                continue
            if status == 'eval':
                ranks = [retrieved_ids.index(positive_id) for positive_id in golden_ids]
                mrr, mhit, top5_tmp, top10_tmp, top20_tmp = cal_metrics(ranks, args.mhits_bar)
                mrrs += mrr
                if len(ranks) > 0:
                    mhits += (mhit / len(ranks))
                top5 += top5_tmp
                top10 += top10_tmp
                top20 += top20_tmp

        # nodess = [ctx['nodes'].strip('|$').strip().split('|$') for ctx in d['ctxs']]
        # edgess = [ctx['edges'].strip('|$').strip().split('|$') for ctx in d['ctxs']]
        id2node_edge = {}
        for i, ctx in enumerate(d['ctxs']):
            # node_lens.append(len(nodess[i]))
            # edge_lens.append(len(edgess[i]))
            nodes = ctx['nodes'][:args.node_length]
            edges = ctx['edges'][:args.edge_length]
            nodes_tokens = nodes[:]
            nodes_tokens_len = len(nodes_tokens)
            edges_tokens = [[nodes_tokens[int(edge[0])], edge[1], nodes_tokens[int(edge[2])]] for edge in edges if int(edge[0]) < nodes_tokens_len and int(edge[2]) < nodes_tokens_len]
            # nodes = [node.split('\t') for node in nodess_]
            # edges = [edge.split('\t') for edge in edgess_]
            # nodes_id2token = {node[0].strip(): re.sub(re.compile(r"-([0-9]([0-9]))"), "", node[1].strip()).strip() for
            #                   i, node in enumerate(nodes)}
            # nodes_tokens = [re.sub(re.compile(r"-([0-9]([0-9]))"), "", node[1].strip()).strip() for node in nodes]
            # edges_ = [edge for edge in edges if edge[0] in nodes_tokens and edge[2] in nodes_tokens]
            # edges_tokens = [node if i==1 else nodes_tokens[node] for edge in edges_ for i, node in enumerate(edge)]
            # edges_tokens = [[edges_tokens[j], edges_tokens[j + 1], edges_tokens[j + 2]] for j in
            #                 range(0, len(edges_tokens), 3)]
            id2node_edge[ctx['id']] = [nodes_tokens, edges_tokens]

        if status == 'train' or status == 'eval':
            ex['positive_ids'] = golden_ids
            ex['negative_ids'] = non_golden_ids
            ex['pos_nodes_edges'] = [id2node_edge[id] for id in golden_ids]
            ex['neg_nodes_edges'] = [id2node_edge[id] for id in non_golden_ids]
        if status != 'train':
            ex['retrieved_ids'] = retrieved_ids
        ex['ret_nodes_edges'] = [id2node_edge[id] for id in retrieved_ids]
        # # get negative passages
        # if status == 'train':
        #     golden_ids = [ctx['id'] for ctx in d['ctxs'] if check_answers(d['answers'], ctx['title'] + ctx['text'])]
        #     if len(golden_ids) == 0:
        #         continue
        #     positive_ids = random.sample(golden_ids, k=1)[0]
        #     ex['positive_ids'] = positive_ids
        #     non_golden_ids = [ctx['id'] for ctx in d['ctxs'] if ctx['id'] not in golden_ids]
        #     if len(non_golden_ids) < args.num_negative_psg:
        #         continue
        #     negative_ids = random.sample(non_golden_ids[:50], k=args.num_negative_psg)
        #     ex['negative_ids'] = negative_ids
        examples.append(ex)
    if status == 'eval':
        print('DPR results:')
        print(f'Num Examples {len(examples)}')
        print(f'MRR: {round(100 * mrrs / len(examples), 1)}')
        print(f'MHits@{args.mhits_bar}: {round(100 * mhits / len(examples), 1)}')
        print(f'TOP5: {round(100 * top5 / len(examples), 1)}')
        print(f'TOP10: {round(100 * top10 / len(examples), 1)}')
        print(f'TOP20: {round(100 * top20 / len(examples), 1)}')
    return examples

def encode_nodes_edges(nodes, edges, graph_max_length, words_dict=None):
    node_ids, node_masks = [], []
    for j in range(len(nodes)):
        nodes_ = nodes[j][:graph_max_length-1]
        edges_ = edges[j][:max(graph_max_length-len(nodes_),1)]
        if len(nodes_) == 0:
            nodes_ = ["", ]
        tmp_node_ids = torch.cat([words_dict[node]['input_ids'] for node in nodes_]).unsqueeze(0)
        tmp_node_ids = tmp_node_ids.unsqueeze(-2)
        tmp_node_ids = torch.cat([tmp_node_ids, tmp_node_ids, tmp_node_ids], dim=-2)

        tmp_node0_ids = torch.cat([words_dict[edge[0]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_rel_ids = torch.cat([words_dict[edge[1]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_node1_ids = torch.cat([words_dict[edge[2]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_edge_ids = torch.cat(
            [tmp_node0_ids.unsqueeze(-2), tmp_rel_ids.unsqueeze(-2), tmp_node1_ids.unsqueeze(-2)], dim=-2)
        tmp_node_ids = torch.cat([tmp_node_ids, tmp_edge_ids], dim=1)
        padding_num = graph_max_length - tmp_node_ids.shape[1]
        zero_pad = ZeroPad2d(padding=(0, 0, 0, 0, 0, padding_num))
        tmp_node_ids = zero_pad(tmp_node_ids)

        tmp_node_attention_mask = torch.LongTensor([2 for t in range(len(nodes_))]).unsqueeze(0)
        tmp_edge_attention_mask = torch.LongTensor([3 for t in range(len(edges_))]).unsqueeze(0)
        tmp_pad_attention_mask = torch.LongTensor([0 for t in range(padding_num)]).unsqueeze(0)
        # tmp_attention_mask = torch.cat([tmp_node_attention_mask, tmp_pad_attention_mask], dim=-1)
        tmp_attention_mask = torch.cat([tmp_node_attention_mask, tmp_edge_attention_mask, tmp_pad_attention_mask],dim=-1)
        node_masks.append(tmp_attention_mask)
        node_ids.append(tmp_node_ids)
    node_ids = torch.cat(node_ids, dim=0)
    node_masks = torch.cat(node_masks, dim=0)
    return node_ids, node_masks

def encode_nodes(nodes, node_max_length, words_dict=None):
    node_ids, node_masks = [], []
    for j in range(len(nodes)):
        nodes_ = nodes[j][:node_max_length]
        if len(nodes_) == 0:
            nodes_ = ["", ]
        tmp_node_ids = torch.cat([words_dict[node]['input_ids'] for node in nodes_]).unsqueeze(0)
        tmp_node_ids = tmp_node_ids.unsqueeze(-2)
        tmp_node_ids = torch.cat([tmp_node_ids, tmp_node_ids, tmp_node_ids], dim=-2)

        padding_num = node_max_length - tmp_node_ids.shape[1]
        zero_pad = ZeroPad2d(padding=(0, 0, 0, 0, 0, padding_num))
        tmp_node_ids = zero_pad(tmp_node_ids)

        tmp_node_attention_mask = torch.LongTensor([2 for t in range(len(nodes_))]).unsqueeze(0)
        tmp_pad_attention_mask = torch.LongTensor([0 for t in range(padding_num)]).unsqueeze(0)
        tmp_attention_mask = torch.cat([tmp_node_attention_mask, tmp_pad_attention_mask],dim=-1)
        node_masks.append(tmp_attention_mask)
        node_ids.append(tmp_node_ids)
    node_ids = torch.cat(node_ids, dim=0)
    node_masks = torch.cat(node_masks, dim=0)
    return node_ids, node_masks

def encode_edges(edges, edge_max_length, words_dict=None):
    edge_ids, edge_masks = [], []
    for j in range(len(edges)):
        edges_ = edges[j][:max(edge_max_length,1)]

        tmp_node0_ids = torch.cat([words_dict[edge[0]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_rel_ids = torch.cat([words_dict[edge[1]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_node1_ids = torch.cat([words_dict[edge[2]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_edge_ids = torch.cat(
            [tmp_node0_ids.unsqueeze(-2), tmp_rel_ids.unsqueeze(-2), tmp_node1_ids.unsqueeze(-2)], dim=-2)
        padding_num = edge_max_length - tmp_edge_ids.shape[1]
        zero_pad = ZeroPad2d(padding=(0, 0, 0, 0, 0, padding_num))
        tmp_edge_ids = zero_pad(tmp_edge_ids)

        tmp_edge_attention_mask = torch.LongTensor([3 for t in range(len(edges_))]).unsqueeze(0)
        tmp_pad_attention_mask = torch.LongTensor([0 for t in range(padding_num)]).unsqueeze(0)
        tmp_attention_mask = torch.cat([tmp_edge_attention_mask, tmp_pad_attention_mask],dim=-1)
        edge_masks.append(tmp_attention_mask)
        edge_ids.append(tmp_edge_ids)
    edge_ids = torch.cat(edge_ids, dim=0)
    edge_masks = torch.cat(edge_masks, dim=0)
    return edge_ids, edge_masks

def get_rerank_dataloader(
        examples,
        args,
        rank,
        bsz,
        shuffle,
        is_train,
        words_dict,
        pid_to_psg,
        tokenizer,
        is_amr=False,
        is_inference=False):

    def _collate_fn(batch):
        random.seed(args.seed)
        ret_ex = {}
        ret_ex['retrieved_pids'] = []
        ret_ex['positive_pids'] = []
        labels = []
        query_psgs = []
        for i, ex in enumerate(batch):
            truncated_query = tokenizer.tokenize(ex['question'])[:args.max_query_length]
            truncated_query = tokenizer.convert_tokens_to_string(truncated_query)
            if is_train:
                if args.all4train:
                    positive_pids = ex['positive_ids']
                    negative_pids = ex['negative_ids']
                    label = torch.cat([torch.ones(len(positive_pids)), torch.zeros(len(negative_pids))], dim=0).type(torch.long)
                    pos_psgs = [pid_to_psg[positive_pid]['text'] for positive_pid in positive_pids]
                    neg_psgs = [pid_to_psg[negative_pid]['text'] for negative_pid in negative_pids]
                    psgs = pos_psgs + neg_psgs
                else:
                    positive_id = random.randint(0, len(ex['positive_ids'])-1)
                    neg_num = [i for i in range(len(ex['negative_ids'][:50]))]
                    negative_ids = random.sample(neg_num, k=args.num_negative_psg)
                    positive_pid = ex['positive_ids'][positive_id]
                    psgs = [pid_to_psg[positive_pid]['text']]
                    for id in negative_ids:
                        psgs.append(pid_to_psg[ex['negative_ids'][id]]['text'])
                    label = torch.cat([torch.ones(1), torch.zeros(args.num_negative_psg)], dim=0).type(torch.long)
                labels.append(label)
            elif is_inference:
                psgs = []
                for id, pid in enumerate(ex['retrieved_ids']):
                    psgs.append(pid_to_psg[pid]['text'])
                ret_ex['retrieved_pids'].append(ex['retrieved_ids'])
            else:
                psgs = []
                for id, pid in enumerate(ex['retrieved_ids']):
                    psgs.append(pid_to_psg[pid]['text'])
                ret_ex['positive_pids'].append(ex['positive_ids'])
                ret_ex['retrieved_pids'].append(ex['retrieved_ids'])

            for psg in psgs:
                query_psgs.append(truncated_query + ' ? ' + psg)

        inputs = tokenizer(
            query_psgs,
            max_length=args.max_combined_length,
            truncation=True,
            return_tensors='pt',
            padding=True)
        ret_ex['inputs'] = inputs
        ret_ex['labels'] = torch.cat(labels, dim=0) if len(labels) > 0 else None
        return ret_ex

    def _collate_fn_amr(batch):
        random.seed(args.seed)
        ret_ex = {}
        ret_ex['retrieved_pids'] = []
        ret_ex['positive_pids'] = []
        labels = []
        query_psgs = []
        nodes = []
        edges = []
        for i, ex in enumerate(batch):
            truncated_query = tokenizer.tokenize(ex['question'])[:args.max_query_length]
            truncated_query = tokenizer.convert_tokens_to_string(truncated_query)
            if is_train:
                positive_id = random.randint(0, len(ex['positive_ids'])-1)
                neg_num = [i for i in range(len(ex['negative_ids'][:50]))]
                negative_ids = random.sample(neg_num, k=args.num_negative_psg)
                positive_pid = ex['positive_ids'][positive_id]
                psgs = [pid_to_psg[positive_pid]['title'] + ' . ' + pid_to_psg[positive_pid]['text']]
                amrs = [ex['ret_nodes_edges'][positive_id]]
                for id in negative_ids:
                    psgs.append(pid_to_psg[ex['negative_ids'][id]]['title']+' . '+pid_to_psg[ex['negative_ids'][id]]['text'])
                    amrs.append(ex['ret_nodes_edges'][id])
                label = torch.cat([torch.ones(1), torch.zeros(args.num_negative_psg)], dim=0).type(torch.long)
                labels.append(label)
            elif is_inference:
                psgs = []
                amrs = []
                for id, pid in enumerate(ex['retrieved_ids']):
                    psgs.append(pid_to_psg[pid]['title'] + ' . ' + pid_to_psg[pid]['text'])
                    amrs.append(ex['ret_nodes_edges'][id])
                ret_ex['retrieved_pids'].append(ex['retrieved_ids'])
            else:
                psgs = []
                amrs = []
                for id, pid in enumerate(ex['retrieved_ids']):
                    psgs.append(pid_to_psg[pid]['title'] + ' . ' + pid_to_psg[pid]['text'])
                    amrs.append(ex['ret_nodes_edges'][id])
                ret_ex['positive_pids'].append(ex['positive_ids'])
                ret_ex['retrieved_pids'].append(ex['retrieved_ids'])

            for psg in psgs:
                query_psgs.append(truncated_query + ' ? '  + psg)
            for amr in amrs:
                nodes.append(amr[0])
                edges.append(amr[1])

        inputs = tokenizer(
            query_psgs,
            max_length=args.max_combined_length,
            truncation=True,
            return_tensors='pt',
            padding=True)

        if args.only_nodes:
            graph_ids, graph_mask = encode_nodes(nodes, args.node_length, words_dict)
        elif args.only_edges:
            graph_ids, graph_mask = encode_edges(edges, args.edge_length, words_dict)
        else:
            graph_ids, graph_mask = encode_nodes_edges(nodes, edges, args.node_length+args.edge_length, words_dict)

        inputs.data.update({'graph_ids':graph_ids, 'graph_mask':graph_mask})
        ret_ex['inputs'] = inputs
        ret_ex['labels'] = torch.cat(labels, dim=0) if len(labels) > 0 else None
        return ret_ex

    # sampler = DistributedSampler(examples, num_replicas=args.world_size, rank=rank, shuffle=shuffle)
    # dl = DataLoader(examples, batch_size=1, collate_fn=_collate_fn, sampler=sampler)
    if is_amr:
        dl = DataLoader(examples, batch_size=bsz, collate_fn=_collate_fn_amr, shuffle=shuffle)
    else:
        dl = DataLoader(examples, batch_size=bsz, collate_fn=_collate_fn, shuffle=shuffle)
    return dl

def get_words_dict(tokenizer, args):
    words_set = set(['Self', ':same'])
    nodes_edges_file = open(args.nodes_edges_file, 'r', encoding='utf8')
    nodes_edges = [n_e.strip() for n_e in nodes_edges_file.readlines()]
    words_set.update(nodes_edges)
    dict = {
        word: tokenizer.encode_plus(word, max_length=args.word_length, padding='max_length', return_tensors='pt', truncation=True)
        for word in tqdm(words_set)}
    dict['Padding-None'] = {'input_ids':torch.LongTensor([[0 for i in range(args.word_length)]])}
    return dict