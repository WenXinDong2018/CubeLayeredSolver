from utils import data_utils, nnet_utils_multihead, env_utils
from typing import Dict, List, Tuple, Any

from environments.environment_abstract import Environment
from updaters.updater_multihead import Updater
from search_methods.gbfs_multihead import gbfs_test
import torch
import wandb
import torch.nn as nn
import os
import pickle

from argparse import ArgumentParser
import numpy as np
import time

import sys
import shutil


WANDB_PROJECT = None
WANDB_ENTITY = None

def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    # Environment
    parser.add_argument('--env', type=str, required=True, help="Environment")

    # Debug
    parser.add_argument('--debug', action='store_true', default=False, help="")

    # Gradient Descent
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--lr_d', type=float, default=0.9999993, help="Learning rate decay for every iteration. "
                                                                      "Learning rate is decayed according to: "
                                                                      "lr * (lr_d ^ itr)")

    # Training
    parser.add_argument('--max_itrs', type=int, default=1000000, help="Maxmimum number of iterations")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size")
    parser.add_argument('--single_gpu_training', action='store_true',
                        default=True, help="If set, train only on one GPU. Update step will still use "
                                            "all GPUs given by CUDA_VISIBLE_DEVICES")

    # Update
    parser.add_argument('--loss_thresh', type=float, default=0.05, help="When the loss falls below this value, "
                                                                        "the target network is updated to the current "
                                                                        "network.")
    parser.add_argument('--states_per_update', type=int, default=1000, help="How many states to train on before "
                                                                            "checking if target network should be "
                                                                            "updated")
    parser.add_argument('--epochs_per_update', type=int, default=1, help="How many epochs to train for. "
                                                                         "Making this greater than 1 could increase "
                                                                         "risk of overfitting, however, one can train "
                                                                         "for more iterations without having to "
                                                                         "generate more data.")
    parser.add_argument('--num_update_procs', type=int, default=1, help="Number of parallel workers used to "
                                                                        "compute updated cost-to-go function")
    parser.add_argument('--update_nnet_batch_size', type=int, default=10000, help="Batch size of each nnet used for "
                                                                                  "each process update. "
                                                                                  "Make smaller if running out of "
                                                                                  "memory.")
    parser.add_argument('--max_update_steps', type=int, default=1, help="Number of steps to take when trying to "
                                                                        "solve training states with "
                                                                        "greedy best-first search (GBFS) or A* search. "
                                                                        "Each state "
                                                                        "encountered when solving is added to the "
                                                                        "training set. Number of steps starts at "
                                                                        "1 and is increased every update until "
                                                                        "the maximum number is reached. "
                                                                        "Value of 1 is the same as doing "
                                                                        "value iteration on only given training "
                                                                        "states. Increasing this number "
                                                                        "can make the cost-to-go function more "
                                                                        "robust by exploring more of the "
                                                                        "state space.")

    parser.add_argument('--update_method', type=str, default="GBFS", help="GBFS or ASTAR. If max_update_steps is 1 "
                                                                          "then either one is the same as doing value "
                                                                          "iteration")

    parser.add_argument('--eps_max', type=float, default=0, help="When addings training states with GBFS, each "
                                                                 "instance will have an eps that is distributed "
                                                                 "randomly between 0 and epx_max.")
    # Testing
    parser.add_argument('--num_test', type=int, default=10000, help="Number of test states.")

    # data
    parser.add_argument('--back_max', type=int, required=True, help="Maximum number of backwards steps from goal")
    parser.add_argument('--dynamic_back_max', action='store_true', default=False, help="Whether to dynamically increase the difficulty of the training exercises")
    parser.add_argument('--dynamic_back_max_per', type = float,  default=25, help="Minimum required solve-percentage to level up difficulty of the training exercises.")
    parser.add_argument("--fixed_difficulty", action='store_true', default=False, help = "fix difficulty of generated training examples during each lesson, to be used in combination with dynamic_back_max=True")
    parser.add_argument("--uniform_data_gen", action='store_true', default=False, help = "toggle the random flag in generate_state method in layer 2. Right now only to be used for layer 2. If turned on, backwards steps from goal disabled. Data generated randomly")
    parser.add_argument("--normal_dist", action='store_true', default=False, help = "Use a normal distirbution with mean = back_max and std = 3 to generate examples")

    # model
    parser.add_argument('--nnet_name', type=str, required=True, help="Name of neural network")
    parser.add_argument('--update_num', type=int, default=0, help="Update number")
    parser.add_argument('--save_dir', type=str, default="saved_models", help="Director to which to save model")
    parser.add_argument('--model_name', type=str, required=True, help="Name of model")

    # parse arguments
    args = parser.parse_args()

    args_dict: Dict[str, Any] = vars(args)

    # make save directory
    model_dir: str = "%s/%s/" % (args_dict['save_dir'], args_dict['nnet_name'])
    args_dict['targ_dir'] = "%s/%s/" % (model_dir, 'target')
    args_dict['curr_dir'] = "%s/%s/" % (model_dir, 'current')

    if not os.path.exists(args_dict['targ_dir']):
        os.makedirs(args_dict['targ_dir'])

    if not os.path.exists(args_dict['curr_dir']):
        os.makedirs(args_dict['curr_dir'])

    args_dict["output_save_loc"] = "%s/output.txt" % model_dir

    # save args
    args_save_loc = "%s/args.pkl" % model_dir
    print("Saving arguments to %s" % args_save_loc)
    with open(args_save_loc, "wb") as f:
        pickle.dump(args, f, protocol=-1)

    print("Batch size: %i" % args_dict['batch_size'])

    return args_dict


def copy_files(src_dir: str, dest_dir: str):
    src_files: List[str] = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name: str = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)


def do_update(back_max: int, update_num: int, env: List[Environment], max_update_steps: int, update_method: str,
              num_states: int, eps_max: float, heur_fn_i_q, heur_fn_o_qs, fixed_difficulty = False, random=False, normal_dist = False) -> Tuple[List[np.ndarray], np.ndarray]:
    '''Generate randomly scrambled states as training examples, do one step look ahead to get training labels '''
    '''Generate num_states training examples'''
    update_steps: int = min(update_num + 1, max_update_steps) #1 in our case
    num_states: int = int(np.ceil(num_states / update_steps))

    # Do updates
    output_time_start = time.time()

    print("Updating cost-to-go with value iteration")
    if max_update_steps > 1:
        print("Using %s with %i step(s) to add extra states to training set" % (update_method.upper(), update_steps))
    updater: Updater = Updater(env, num_states, back_max, heur_fn_i_q, heur_fn_o_qs, update_steps, update_method,
                               update_batch_size=10000, eps_max=eps_max, fixed_difficulty = fixed_difficulty, random=random, normal_dist= normal_dist)

    states_update_nnet: List[np.ndarray]
    output_update: np.ndarray
    states_update_nnet, output_update, is_solved = updater.update()
    print("output_update", output_update.shape)
    # Print stats
    # if max_update_steps > 1:
        # print("%s produced %s states, %.2f%% solved (%.2f seconds) for layer 1" % (update_method.upper(),
        #                                                                format(output_update[0].shape[0], ","),
        #                                                                100.0 * np.mean(is_solved[0]),
        #                                                                time.time() - output_time_start))
        # print("%s produced %s states, %.2f%% solved (%.2f seconds) for layer 2" % (update_method.upper(),
        #                                                                format(output_update[1].shape[0], ","),
        #                                                                100.0 * np.mean(is_solved[1]),
        #                                                                time.time() - output_time_start))
        # print("%s produced %s states, %.2f%% solved (%.2f seconds) for layer 3" % (update_method.upper(),
        #                                                                format(output_update[2].shape[0], ","),
        #                                                                100.0 * np.mean(is_solved[2]),
        #                                                                time.time() - output_time_start))
    mean_ctg1 = output_update[ :, 0].mean()
    mean_ctg2 = output_update[:, 1].mean()
    mean_ctg3 = output_update[:, 2].mean()
    min_ctg1 = output_update[:, 0].min()
    max_ctg1 = output_update[ :, 1].max()
    min_ctg2 = output_update[ :, 2].min()
    max_ctg2 = output_update[ :, 0].max()
    min_ctg3 = output_update[ :, 1].min()
    max_ctg3 = output_update[:, 2].max()
    print("Cost-to-go (mean/min/max) for layer 1: %.2f/%.2f/%.2f" % (mean_ctg1, min_ctg1, max_ctg1))
    print("Cost-to-go (mean/min/max) for layer 2: %.2f/%.2f/%.2f" % (mean_ctg2, min_ctg2, max_ctg2))
    print("Cost-to-go (mean/min/max) for layer 3: %.2f/%.2f/%.2f" % (mean_ctg3, min_ctg3, max_ctg3))
    wandb.log({"mean_ctg_layer_1": mean_ctg1, "is_solved_layer_1": is_solved[0]})
    wandb.log({"mean_ctg_layer_2": mean_ctg2, "is_solved_layer_2": is_solved[1]})
    wandb.log({"mean_ctg_layer_3": mean_ctg3, "is_solved_layer_3": is_solved[2]})

    return states_update_nnet, output_update


def load_nnet(nnet_dir: str, env: Environment, model_name:str) -> Tuple[nn.Module, int, int]:
    nnet_file: str = "%s/model_state_dict.pt" % nnet_dir
    if os.path.isfile(nnet_file):
        nnet = nnet_utils_multihead.load_nnet(nnet_file, env[0].get_nnet_model(model_name))
        itr: int = pickle.load(open("%s/train_itr.pkl" % nnet_dir, "rb"))
        update_num: int = pickle.load(open("%s/update_num.pkl" % nnet_dir, "rb"))
    else:
        nnet: nn.Module = env[0].get_nnet_model(model_name)
        itr: int = 0
        update_num: int = 0

    return nnet, itr, update_num


def main():

    assert(WANDB_PROJECT and WANDB_ENTITY)

    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser)

    if not args_dict["debug"]:
        sys.stdout = data_utils.Logger(args_dict["output_save_loc"], "a")

    # environment
    env: List[Environment] = [env_utils.get_environment("cube3_layer1"), env_utils.get_environment("cube3_layer2"), env_utils.get_environment("cube3")]

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils_multihead.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    # load nnet
    nnet: nn.Module
    itr: int
    update_num: int
    nnet, itr, update_num = load_nnet(args_dict['curr_dir'], env, args_dict["model_name"])

    nnet.to(device)
    if on_gpu and (not args_dict['single_gpu_training']):
        nnet = nn.DataParallel(nnet)

    #initialize data visualizer
    run_id = "{}-{}".format(args_dict["env"], args_dict["nnet_name"])
    wandb.init(project=WANDB_PROJECT,entity = WANDB_ENTITY, id = run_id, name = run_id, config = args_dict)


    dynamic_back_max = 0
    can_increase_dynamic_back_max = False
    # training
    '''In every itr:
    1 we generate args_dict['states_per_update'] random cubes (training examples), and corresponding labels using
    value iteration.
    2. Train DNN on these examples (for `epochs_per_update` epochs), using a batch size of args_dict['batch_size']
    3. Check if DNN's loss if below threshold, if yes, update target network with current network.
    4. Test the DNN on random cubes (different from training cubes)
    '''
    while itr < args_dict['max_itrs']:
        # update
        targ_file: str = "%s/model_state_dict.pt" % args_dict['targ_dir']
        all_zeros: bool = not os.path.isfile(targ_file)
        heur_fn_i_q, heur_fn_o_qs, heur_procs = nnet_utils_multihead.start_heur_fn_runners(args_dict['num_update_procs'],
                                                                                 args_dict['targ_dir'],
                                                                                 device, on_gpu, env,
                                                                                 all_zeros=all_zeros,
                                                                                 clip_zero=True,
                                                                                 batch_size=args_dict[
                                                                                     "update_nnet_batch_size"], model_name = args_dict['model_name'])

        states_nnet: List[List[np.ndarray]]
        outputs: List[np.ndarray]

        if args_dict["dynamic_back_max"]:
            states_nnet, outputs = do_update(dynamic_back_max, update_num, env,
                                            args_dict['max_update_steps'], args_dict['update_method'],
                                            args_dict['states_per_update'], args_dict['eps_max'],
                                            heur_fn_i_q, heur_fn_o_qs, fixed_difficulty=args_dict["fixed_difficulty"], random=False, normal_dist = args_dict["normal_dist"])
        elif args_dict["uniform_data_gen"]:
            states_nnet, outputs = do_update(dynamic_back_max, update_num, env,
                                            args_dict['max_update_steps'], args_dict['update_method'],
                                            args_dict['states_per_update'], args_dict['eps_max'],
                                            heur_fn_i_q, heur_fn_o_qs, random=args_dict["uniform_data_gen"], fixed_difficulty=False)
        else:
            states_nnet, outputs = do_update(args_dict["back_max"], update_num, env,
                                         args_dict['max_update_steps'], args_dict['update_method'],
                                         args_dict['states_per_update'], args_dict['eps_max'],
                                         heur_fn_i_q, heur_fn_o_qs, fixed_difficulty = False, random=False)

        nnet_utils_multihead.stop_heuristic_fn_runners(heur_procs, heur_fn_i_q)

        # train nnet
        num_train_itrs: int = args_dict['epochs_per_update'] * np.ceil(outputs.shape[0] / args_dict['batch_size'])
        print("Training model for update number %i for %i iterations" % (update_num, num_train_itrs))
        # needs change for train_nnet
        last_loss = nnet_utils_multihead.train_nnet(nnet, states_nnet, outputs, device, args_dict['batch_size'], num_train_itrs,
                                          itr, args_dict['lr'], args_dict['lr_d'])
        itr += num_train_itrs
        wandb.log({"loss": last_loss})
        # save nnet
        torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % args_dict['curr_dir'])
        pickle.dump(itr, open("%s/train_itr.pkl" % args_dict['curr_dir'], "wb"), protocol=-1)
        pickle.dump(update_num, open("%s/update_num.pkl" % args_dict['curr_dir'], "wb"), protocol=-1)

        # test
        start_time = time.time()
        heuristic_fn = nnet_utils_multihead.get_heuristic_fn(nnet, device, env, batch_size=args_dict['update_nnet_batch_size'])
        max_solve_steps: int = min(update_num + 1, args_dict['back_max'])
        if args_dict["dynamic_back_max"]:
            per_solved = gbfs_test(args_dict['num_test'], args_dict['back_max'], env, heuristic_fn, max_solve_steps=max_solve_steps, dynamic_back_max = dynamic_back_max)
            #if agents does decently well on problems generated dynamic_back_max steps, then increase dynamic_back_max
            if (per_solved>args_dict["dynamic_back_max_per"]): #If percentage solved pass this number we increase the difficulty of the generated problems
                can_increase_dynamic_back_max = True
            wandb.log({"dynamic_back_max": dynamic_back_max})

        else:
            gbfs_test(args_dict['num_test'], args_dict['back_max'], env, heuristic_fn, max_solve_steps=max_solve_steps, random=args_dict["uniform_data_gen"])

        wandb.log({"max_solve_steps": max_solve_steps})
        print("Test time: %.2f" % (time.time() - start_time))

        # clear cuda memory
        torch.cuda.empty_cache()

        print("Last loss was %f" % last_loss)
        if last_loss < args_dict['loss_thresh']:
            # Update nnet
            print("Updating target network")
            copy_files(args_dict['curr_dir'], args_dict['targ_dir'])
            update_num = update_num + 1
            pickle.dump(update_num, open("%s/update_num.pkl" % args_dict['curr_dir'], "wb"), protocol=-1)
            if args_dict["dynamic_back_max"] and can_increase_dynamic_back_max:
                dynamic_back_max = min(args_dict["back_max"], dynamic_back_max+1)
                can_increase_dynamic_back_max = False
        wandb.log({"update_num": update_num})
    print("Done")


if __name__ == "__main__":
    main()
