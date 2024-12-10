# Academic Article Ranking With AMR Graphs and Knowledge Graphs in QA Tasks
## About
This repo contains the experiments ran for our project. It investigates methods to retrieve and rank articles for answering academic questions by using Abstract Meaning Representation (AMR) graphs and Knowledge Graphs (KG). The study compares the effectiveness of different graph types (AMR, KG, and AMR+KG) and Graph Neural Network (GNN) architectures (GAT, GCN, and GraphSAGE) on the Academic Question Answering (AQA) and Natural Questions (NQ) datasets. The results show that GraphSAGE performs best across all graph types due to high heterophily, and combining AMR and KG graphs can lead to more robust reranking schemes. These findings offer insights into the interaction between graph representations and GNN architectures in passage reranking tasks for academic question answering.

## Datasets used in the experiment
- NQ dataset downoload: https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv
- KDD-AQA can be downloaded from the official website: https://www.biendata.xyz/competition/aqa_kdd_2024/

## Experiment artifacts
AMR, KG graphs, and datasets artifacts are generated if they do not exist.
Pregenerated graphs and datasets artifacts can be downloaded here.
https://drive.google.com/drive/folders/12oD41P7yVl23RkG0o0rb1IC4FgHILuVe

Here is the list of files needed to run each experiment:
### For NQ Experiments
- nq_pickled directory

### For AQA Experiments
- pid_embeddings.pickle
- pid_to_title_abs_new.json
- retrieval_results_qa_valid_wo_ans.json
- retrieval_results_qa_train.json 
- retrieval_results_qa_test_wo_ans.json


#### For AMR AQA Experiments
- amr_graphs.pickle
- qa_valid_flag.txt

#### For KG AQA Experiments
- data directory
- data_loader_train.pkl
- question_embeddings_test.pickle
- question_embeddings_train.pickle


Place all files in the root folder.
### Generate AMR graphs
```bash
python faster_custom_prediction.py --output-dir /scratch/chaijy_root/chaijy2/josuetf/chunked_amr_results --chunk-id 0 --total-chunks 4 --num-workers 4 --batch-size 128
```
## How to run experiments
Directly run the scripts located in the scripts directory or run these commands:

### NQ Experiments
To run the exact experiments used to generate the NQ experiments used in the report, simply run /scripts/run_nq_experiments.sh

To run the NQ experiments, simply download the pickled data as described above.
Or, if you for some reason want to pickle the data yourself, download the following files, and set the appropriate constansts in constants.py
The pre-generated AMR graphs are in the drive:
- retrieval_results_qa_train.json
- retrieval_results_qa_test_wo_ans.json
- retrieval_results_qa_valid_wo_ans.json
- qa_valid_flag.txt
The NQ dataset can be found here:
- https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv
Then, run /nq_pickler pickle_data.py followed by /nq_pickler/re_pickle.py

As an example, to run the experiments with the GNN, run

```bash
python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-3 --num_sims 5
python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-1 --num_sims 5
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-1 --num_sims 5

python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-3 --num_sims 5
python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-3 --num_sims 5

python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-3 --num_sims 5
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-3 --num_sims 5
```

To run the DPR experiment, run

```bash
python nq_dpr_experiments.py  --train_num_samples 3000 --test_num_samples 500
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
