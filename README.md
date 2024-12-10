# Academic Article Ranking With AMR Graphs and Knowledge Graphs in QA Tasks
## About
This repo contains the experiments ran for our project. It investigates methods to retrieve and rank articles for answering academic questions by using Abstract Meaning Representation (AMR) graphs and Knowledge Graphs (KG). The study compares the effectiveness of different graph types (AMR, KG, and AMR+KG) and Graph Neural Network (GNN) architectures (GAT, GCN, and GraphSAGE) on the Academic Question Answering (AQA) and Natural Questions (NQ) datasets. The results show that GraphSAGE performs best across all graph types due to high heterophily, and combining AMR and KG graphs can lead to more robust reranking schemes. These findings offer insights into the interaction between graph representations and GNN architectures in passage reranking tasks for academic question answering.
## Experiment artifacts
AMR, KG graphs, and datasets artifacts are generated if they do not exist.
Pregenerated graphs and datasets artifacts can be downloaded here.
https://drive.google.com/drive/folders/12oD41P7yVl23RkG0o0rb1IC4FgHILuVe


Place all files in the root folder.
### Generate AMR graphs
```bash
python faster_custom_prediction.py --output-dir /scratch/chaijy_root/chaijy2/josuetf/chunked_amr_results --chunk-id 0 --total-chunks 4 --num-workers 4 --batch-size 128
```
## How to run experiments
Directly run the scripts located in the scripts directory or run these commands:

### NQ Experiments
```bash
python nq_main_experiments.py --model_name amr  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --amr_number_of_links 20 --num_epochs 100 
python nq_main_experiments.py --model_name kg  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --kg_link_type ssr --kg_number_of_links 3 --num_epochs 100
python nq_main_experiments.py --model_name kg  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --kg_link_type se --kg_number_of_links 7 --num_epochs 100 
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --kg_link_type ssr --kg_number_of_links 3 --amr_number_of_links 20 --num_epochs 100
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --kg_link_type se --kg_number_of_links 7 --amr_number_of_links 20 --num_epochs 100

python nq_main_experiments.py --model_name amr  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --amr_number_of_links 20 --num_epochs 300 
python nq_main_experiments.py --model_name kg  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --kg_link_type ssr --kg_number_of_links 3 --num_epochs 300 
python nq_main_experiments.py --model_name kg  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --kg_link_type se --kg_number_of_links 7 --num_epochs 300 
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --kg_link_type ssr --kg_number_of_links 3 --amr_number_of_links 20 --num_epochs 300
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 5000 --test_num_samples 1000 --gnn_type gat --kg_link_type se --kg_number_of_links 7 --amr_number_of_links 20 --num_epochs 300
```

### AQA KG experiments
```bash
python aqa_kg_main_experiments.py  --gnn_type gcn
python aqa_kg_main_experiments.py  --gnn_type gat
python aqa_kg_main_experiments.py  --gnn_type sage
```

### AQA AMR experiments
```bash
python aqa_amr_main_experiments_hyper_search.py --model_name amr+kg  --train_num_samples 8757 --test_num_samples 2919 --gnn_type gcn --amr_number_of_links 20 --num_epochs 20
python aqa_amr_main_experiments_hyper_search.py --model_name amr+kg  --train_num_samples 8757 --test_num_samples 2919 --gnn_type gat --amr_number_of_links 20 --num_epochs 20
python aqa_amr_main_experiments_hyper_search.py --model_name amr+kg  --train_num_samples 8757 --test_num_samples 2919 --gnn_type sage --amr_number_of_links 20 --num_epochs 20
python aqa_amr_main_experiments_hyper_search.py --model_name amr  --train_num_samples 8757 --test_num_samples 2919 --gnn_type gcn --amr_number_of_links 20 --num_epochs 20
python aqa_amr_main_experiments_hyper_search.py --model_name amr  --train_num_samples 8757 --test_num_samples 2919 --gnn_type gat --amr_number_of_links 20 --num_epochs 20
python aqa_amr_main_experiments_hyper_search.py --model_name amr  --train_num_samples 8757 --test_num_samples 2919 --gnn_type sage --amr_number_of_links 20 --num_epochs 20
```
## License
No license. Feel free to do whatever you want with this code.
