from typing import List, Tuple, Dict, Callable, Optional, Any
from environments.environment_abstract import Environment, State
import numpy as np
from heapq import heappush, heappop
from subprocess import Popen, PIPE

from argparse import ArgumentParser
import torch
from utils import env_utils, nnet_utils, search_utils, misc_utils, data_utils
import pickle
import time
import sys
import os
import socket
from torch.multiprocessing import Process



def main():
    #time how long it takes to come up with heuristic for 1000 states
    device, devices, on_gpu = nnet_utils.get_device()
    model_dir = "saved_models/cube3/current"
    args_states = "data/cube3/test/data_0.pkl"
    env: Environment = env_utils.get_environment("cube3")
    print(on_gpu)
    heuristic_fn = nnet_utils.load_heuristic_fn(model_dir, device, on_gpu, env.get_nnet_model(),
                                                env, clip_zero=True, batch_size=1)
    input_data = pickle.load(open(args_states, "rb"))
    states: List[State] = input_data['states'][0:1000]
    start_time = time.time()

    heuristics = heuristic_fn(states)
    print(time.time() - start_time)
if __name__ == "__main__":

    main()


