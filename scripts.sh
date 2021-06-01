###-------------------------------------------- Train-------------------------------------------###

### Train Layer 1 Agent (solves first layer)
python ctg_approx/avi.py --env cube3_layer1 --states_per_update 500000
--batch_size 1000 --nnet_name cube3layer1
--max_itrs 50000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --model_name singleH

### Train Layer 2 Agent (solves first two layers)
python ctg_approx/avi.py --env cube3_layer2 --states_per_update 500000
--batch_size 1000 --nnet_name cube3layer2
--max_itrs 50000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --model_name singleH

### Train Layer 3 Agent (solves entire cube)
python ctg_approx/avi.py --env cube3 --states_per_update 500000
--batch_size 1000 --nnet_name cube3layer3
--max_itrs 50000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --model_name singleH

### Multihead Model Baseline
python ctg_approx/avi_multihead.py --env cube3 --states_per_update 500000
--batch_size 1000 --nnet_name final_cube3multihead_baseline
 --max_itrs 50000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --model_name multiH_baseline
### Multihead Model 4+1
python ctg_approx/avi_multihead.py --env cube3 --states_per_update 500000
--batch_size 1000 --nnet_name final_cube3multihead4plus1
--max_itrs 50000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --model_name multiH_4plus1

###--------------------------------------------Search -------------------------------------------###
# *setting start_idx = 900 solves 100 cubes, setting start_idx = 0 solves 1000 cubes
# *setting start_idx = 900 solves 100 cubes, setting start_idx = 0 solves 1000 cubes

#1. A* search, layer 1
python search_methods/astar.py --states data/cube3/test/data_0.pkl
--model saved_models/singleH-cube3layer1_baseline/target/
--env cube3_layer1 --weight 0.2 --batch_size 100
--results_dir results/cube3layer1_baseline/
--language python --nnet_batch_size 10000 --start_idx 900 --max_nodes 600000


#4. A* search using original deepcubea model
python search_methods/astar.py --states data/cube3/test/data_0.pkl
--model saved_models/cube3/target/ --env cube3 --weight 0.2
--batch_size 100 --results_dir results/deepcubea_original --language python
--nnet_batch_size 10000 --start_idx 900 --max_nodes 1800000

##Sequential Search using SingleHead Models
python search_methods/sequential.py
--states data/cube3/test/data_0.pkl
--weight 0.2 --batch_size 100
--nnet_batch_size 10000 --start_idx 900
--model_dir_layer1 saved_models/singleH-cube3layer1_baseline/target/
--model_dir_layer2 saved_models/singleH_cube3layer2_baseline/target/
--model_dir_layer3 saved_models/singleH_cube3layer3_baseline/target/
--results_dir results/cube3_sequential/


##Sequential Search using Multihead Models
python search_methods/sequential_multi_head.py --states data/cube3/test/data_0.pkl
--weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 900
--model_dir saved_models/cube3multihead_baseline/target --results_dir results/cube3_multihead_sequential/

