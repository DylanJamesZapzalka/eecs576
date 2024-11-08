
import wget
import constants
import gzip
import shutil

# Get file urls
TRIVIA_TEST_URL = constants.TRIVIA_TEST_URL
NQ_TEST_URL = constants.NQ_TEST_URL
# Get file download locations
DATASETS_DIR = constants.DATASETS_DIR

# Download files
trivia_filename = wget.download(TRIVIA_TEST_URL, DATASETS_DIR)
nq_filename = wget.download(NQ_TEST_URL, DATASETS_DIR)

# Unzip
with gzip.open(trivia_filename, 'rb') as f_in:
    with open(trivia_filename[0:len(trivia_filename) - 3], 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    
# THIS FILE MAY ALSO NEED TO BE DOWNLOADED AND PUT INTO
# '<DATASETS_DIR>/wiki_dpr/psgs_w100.multiset.compressed/0.0.0/'
# https://huggingface.co/datasets/facebook/wiki_dpr/blob/main/index/psgs_w100.multiset.IVF4096_HNSW128_PQ128-IP-train.faiss