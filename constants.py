import os


# DO NOT CHANGE
TRIVIA_TEST_URL = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz'
NQ_TEST_URL = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv'
# DO NOT CHANGE

# SHOULD CHANGE
DATASETS_DIR = '/scratch/chaijy_root/chaijy2/josuetf/eecs576_datasets/NQ_amr/nodes_edges_qtc/amrbart/in_link'
# SHOULD CHANGE

# DO NOT CHANGE
TRIVIA_TEST_FILE_NAME = os.path.join(DATASETS_DIR, 'trivia-test.qa.csv')
NQ_TEST_FILE_NAME = os.path.join(DATASETS_DIR, 'nq-test.qa.csv')
AMR_NQ_TRAIN_FILE_NAME = os.path.join(DATASETS_DIR, 'train.jsonl')
AMR_NQ_TEST_FILE_NAME = os.path.join(DATASETS_DIR, 'test.jsonl')