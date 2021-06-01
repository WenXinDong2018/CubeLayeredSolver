# A layer-by-layer cube solver using Deep RL and Multihead training

This is a CS299 final project.

This repository is adapted from [DeepCubeA](https://www.ics.uci.edu/~fagostin/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf).

# Setup
For required python packages, please see requirements.txt.
You should be able to install these packages with pip or conda

Python version used: 3.7.2

IMPORTANT! Before running anything, please execute: `source setup.sh` in the DeepCubeA directory to add the current
directory to your python path.

The number of GPUs used can be controlled by setting the `CUDA_VISIBLE_DEVICES` environment variable.

i.e. `export CUDA_VISIBLE_DEVICES="0"` if you have only one GPU or `export CUDA_VISIBLE_DEVICES="0,1"` if you have two




# Training and A* Search
`scripts.sh` contains the commands to train/ perform search.

There are pre-trained models in the `saved_models/` directory as well as `output.txt` files to let you know what output to expect.

These models were trained with 1 GPU, for 8-10 hours.

There are pre-computed results of A* search in the `results/` directory.

### Commands to train a multihead model.
###### Train cost-to-go function
`python ctg_approx/avi_multihead.py --env cube3 --states_per_update 500000 --batch_size 1000 --nnet_name final_cube3multihead_baseline --max_itrs 50000 --loss_thresh 0.2 --back_max 30 --num_update_procs 30 --model_name multiH_baseline`

###### Solve with A* search, use --verbose for more information
`python search_methods/sequential_multi_head.py --states data/cube3/test/data_0.pkl --weight 0.6 --batch_size 1000 --nnet_batch_size 10000 --start_idx 900 --model_dir saved_models/cube3multihead_baseline/target --results_dir results/cube3_multihead_sequential/`

### Improving Results
During approximate value iteration (AVI), one can get better results by increasing the batch size (`--batch_size`) and number of states per update (`--states_per_update`).
Decreasing the threshold before the target network is updated (`--loss_thresh`) can also help.

One can also add additional states to training set by doing greedy best-first search (GBFS) during the update stage and adding the states encountered during GBFS to the states used for approximate value iteration (`--max_update_steps`). Setting `--max_update_steps` to 1 is the same as doing approximate value iteration.

During A* search, increasing the weight on the path cost (`--weight`, range should be [0,1]) and the batch size (`--batch_size`) generally improves results.

These improvements often come at the expense of time.


# Memory
When obtaining training data with approximate value iteration and solving using A* search, the batch size of the data
given to the DNN can be controlled with `--update_nnet_batch_size` for the `avi.py` file and `--nnet_batch_size` for
the `astar.py` file. Reduce this value if your GPUs are running out of memory during approximate value iteration or
during A* search.
