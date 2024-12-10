
# This was the script used to generate all of the midterm results
python main_tr_nq_experiments.py --dataset_name nq --train_num_samples 2500 --eval_num_samples 500 --link_type ssr --number_of_links 3 --gnn_type gat --num_epochs 100 --num_dpr_samples 100
python main_tr_nq_experiments.py --dataset_name nq --train_num_samples 2500 --eval_num_samples 500 --link_type ssr --number_of_links 3 --gnn_type gcn --num_epochs 100 --num_dpr_samples 100
python main_tr_nq_experiments.py --dataset_name nq --train_num_samples 2500 --eval_num_samples 500 --link_type ssr --number_of_links 3 --gnn_type sage --num_epochs 100 --num_dpr_samples 100

python main_tr_nq_experiments.py --dataset_name nq --train_num_samples 2500 --eval_num_samples 500 --link_type se --number_of_links 3 --gnn_type gat --num_epochs 100 --num_dpr_samples 100
python main_tr_nq_experiments.py --dataset_name nq --train_num_samples 2500 --eval_num_samples 500 --link_type se --number_of_links 3 --gnn_type gcn --num_epochs 100 --num_dpr_samples 100
python main_tr_nq_experiments.py --dataset_name nq --train_num_samples 2500 --eval_num_samples 500 --link_type se --number_of_links 3 --gnn_type sage --num_epochs 100 --num_dpr_samples 100


python main_tr_nq_experiments.py --dataset_name trivia --train_num_samples 2500 --eval_num_samples 500 --link_type ssr --number_of_links 3 --gnn_type gat --num_epochs 100 --num_dpr_samples 100
python main_tr_nq_experiments.py --dataset_name trivia --train_num_samples 2500 --eval_num_samples 500 --link_type ssr --number_of_links 3 --gnn_type gcn --num_epochs 100 --num_dpr_samples 100
python main_tr_nq_experiments.py --dataset_name trivia --train_num_samples 2500 --eval_num_samples 500 --link_type ssr --number_of_links 3 --gnn_type sage --num_epochs 100 --num_dpr_samples 100

python main_tr_nq_experiments.py --dataset_name trivia --train_num_samples 2500 --eval_num_samples 500 --link_type se --number_of_links 3 --gnn_type gat --num_epochs 100 --num_dpr_samples 100
python main_tr_nq_experiments.py --dataset_name trivia --train_num_samples 2500 --eval_num_samples 500 --link_type se --number_of_links 3 --gnn_type gcn --num_epochs 100 --num_dpr_samples 100
python main_tr_nq_experiments.py --dataset_name trivia --train_num_samples 2500 --eval_num_samples 500 --link_type se --number_of_links 3 --gnn_type sage --num_epochs 100 --num_dpr_samples 100