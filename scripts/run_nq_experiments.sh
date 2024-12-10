# This is the script to generate all of the final results over the nq dataset


# python nq_main_experiments.py --model_name kg --train_num_samples 100 --test_num_samples 1 --kg_link_type ssr --kg_number_of_links 3 --gnn_type gat --num_epochs 100
# python nq_main_experiments.py --model_name kg --train_num_samples 100 --test_num_samples 1 --kg_link_type se --kg_number_of_links 7 --gnn_type gat --num_epochs 100
# python nq_main_experiments.py --model_name amr --train_num_samples 100 --test_num_samples 1 --gnn_type gat --num_epochs 1 --amr_number_of_links 20



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