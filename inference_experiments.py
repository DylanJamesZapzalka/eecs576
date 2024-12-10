import torch
from transformers import pipeline
from huggingface_hub import login
from amr_bart_utils import get_rerank_examples_odqa_amr, load_data
# import torch
# print(torch.cuda.is_available())

# Make sure this is a read token, otherwise you will get an error
# Also, make sure you delete your token before pushing any changes to the github repo
# login(token="")

# model_id = "meta-llama/Llama-3.2-1B"

# pipe = pipeline(
#     "text-generation", 
#     model=model_id, 
#     torch_dtype=torch.bfloat16, 
#     device_map="cuda",
#     max_new_tokens=128
# )

# what = pipe("What species of fish is Nemo?")
# print(what)

data = load_data(data_path="/home/dylanz/eecs576_data/test_paraphrased.jsonl")

data[i]
for i in range(len(data)):
    print(data[i]['question'])
# get_rerank_examples_odqa_amr("/home/dylanz/eecs576_data/dev.jsonl")