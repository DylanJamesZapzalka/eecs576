import os


# DO NOT CHANGE
TRIVIA_TEST_URL = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz'
NQ_TEST_URL = 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv'
# DO NOT CHANGE

# SHOULD CHANGE
DATASETS_DIR = '/mnt/c/Users/Andy/Github/rebel/AQA'
# SHOULD CHANGE

# DO NOT CHANGE
TRIVIA_TEST_FILE_NAME = os.path.join(DATASETS_DIR, 'trivia-test.qa.csv')
NQ_TEST_FILE_NAME = os.path.join(DATASETS_DIR, 'nq-test.qa.csv')
AMR_NQ_TRAIN_FILE_NAME = os.path.join(DATASETS_DIR, 'train.jsonl')
AMR_NQ_TEST_FILE_NAME = os.path.join(DATASETS_DIR, 'test.jsonl')

#
AQA_TRAIN_FILE_NAME = os.path.join(DATASETS_DIR, 'retrieval_results_qa_train.json')
AQA_TEST_FILE_NAME = os.path.join(DATASETS_DIR, 'retrieval_results_qa_test_wo_ans.json')
AQA_VAL_FILE_NAME = os.path.join(DATASETS_DIR, 'retrieval_results_qa_valid_wo_ans.json')
AQA_VAL_ANS = os.path.join(DATASETS_DIR, "qa_valid_flag.txt")