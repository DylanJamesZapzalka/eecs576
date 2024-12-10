# This is the script to generate all of the final results over the nq dataset




# Validation experiments start here -------------------------------------

python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-1 --num_sims 5 # 1e-3 is the best
python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-3 --num_sims 5

python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-1 --num_sims 5 # 1e-1 is the best
python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-3 --num_sims 5

python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-1 --num_sims 5 # 1e-1 is the best
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-3 --num_sims 5


python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-1 --num_sims 5 # 1e-3 is the best
python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-3 --num_sims 5

python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-1 --num_sims 5 # 0 is the best
python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-3 --num_sims 5

python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-1 --num_sims 5 # 1e-3 is the best
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-3 --num_sims 5


python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-1 --num_sims 5 # 0 is the best
python nq_main_experiments.py --model_name amr  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-3 --num_sims 5

python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-1 --num_sims 5 # 1e-3 is the best
python nq_main_experiments.py --model_name kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-3 --num_sims 5

python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-1 --num_sims 5 # 1e-3 is the best
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 2500 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-3 --num_sims 5

# Validation experiments end here -------------------------------------



# Test  experiments start here -------------------------------------

python nq_main_experiments.py --model_name amr  --train_num_samples 1000 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-3 --num_sims 5
python nq_main_experiments.py --model_name kg  --train_num_samples 1000 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-1 --num_sims 5
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 1000 --test_num_samples 500 --gnn_type gcn --num_epochs 20 --weight_decay 1e-1 --num_sims 5

python nq_main_experiments.py --model_name amr  --train_num_samples 1000 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-3 --num_sims 5
python nq_main_experiments.py --model_name kg  --train_num_samples 1000 --test_num_samples 500 --gnn_type gat  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 1000 --test_num_samples 500 --gnn_type gat --num_epochs 20 --weight_decay 1e-3 --num_sims 5

python nq_main_experiments.py --model_name amr  --train_num_samples 1000 --test_num_samples 500 --gnn_type sage  --num_epochs 20 --weight_decay 0 --num_sims 5
python nq_main_experiments.py --model_name kg  --train_num_samples 1000 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-3 --num_sims 5
python nq_main_experiments.py --model_name amr+kg  --train_num_samples 1000 --test_num_samples 500 --gnn_type sage --num_epochs 20 --weight_decay 1e-3 --num_sims 5

# Test  experiments start here -------------------------------------


# Experiments for DPR
python nq_dpr_experiments.py  --train_num_samples 3000 --test_num_samples 500