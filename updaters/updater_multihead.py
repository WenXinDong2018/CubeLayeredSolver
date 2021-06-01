from typing import List, Tuple
import numpy as np
from utils import nnet_utils, misc_utils, nnet_utils_multihead
from environments.environment_abstract import Environment, State
from search_methods.gbfs_multihead import GBFS
from search_methods.astar import AStar, Node
from torch.multiprocessing import Queue, get_context
from environments.cube3_layer1 import Cube3Layer1
from environments.cube3_layer2 import Cube3Layer2
from environments.cube3 import Cube3Layer3
import time


def gbfs_update(states: List[State], env: List[Environment], num_steps: int, heuristic_fn, eps_max: float):
    eps: List[float] = list(np.random.rand(len(states)) * eps_max)

    gbfs1 = GBFS(states, env[0], eps=eps)
    gbfs2 = GBFS(states, env[1], eps=eps)
    gbfs3 = GBFS(states, env[2], eps=eps)
    for _ in range(num_steps):
        gbfs1.step(heuristic_fn, idx = 0)
        gbfs2.step(heuristic_fn, idx= 1)
        gbfs3.step(heuristic_fn, idx=2)

    trajs1: List[List[Tuple[State, float]]] = gbfs1.get_trajs()
    trajs2: List[List[Tuple[State, float]]] = gbfs2.get_trajs()
    trajs3: List[List[Tuple[State, float]]] = gbfs3.get_trajs()

    trajs_flat1: List[Tuple[State, float]]
    trajs_flat1, _ = misc_utils.flatten(trajs1)
    trajs_flat2: List[Tuple[State, float]]
    trajs_flat2, _ = misc_utils.flatten(trajs2)
    trajs_flat3: List[Tuple[State, float]]
    trajs_flat3, _ = misc_utils.flatten(trajs3)

    is_solved1: np.ndarray = np.array(gbfs1.get_is_solved())
    is_solved2: np.ndarray = np.array(gbfs2.get_is_solved())
    is_solved3: np.ndarray = np.array(gbfs3.get_is_solved())

    states_update: List = []
    cost_to_go_update_l1: List[float] = []
    cost_to_go_update_l2: List[float] = []
    cost_to_go_update_l3: List[float] = []
    for id, traj in enumerate(trajs_flat1):
        states_update.append(traj[0])
        cost_to_go_update_l1.append(trajs_flat1[id][1])
        cost_to_go_update_l2.append(trajs_flat2[id][1])
        cost_to_go_update_l3.append(trajs_flat3[id][1])

    cost_to_go_update1 = np.array(cost_to_go_update_l1)
    cost_to_go_update2 = np.array(cost_to_go_update_l2)
    cost_to_go_update3 = np.array(cost_to_go_update_l3)

    cost_to_go_update = np.stack([cost_to_go_update1, cost_to_go_update2, cost_to_go_update3]).T
    assert(cost_to_go_update.shape == (cost_to_go_update.shape[0], 3))
    is_solved = np.stack([is_solved1, is_solved2, is_solved3]).T
    assert(is_solved.shape == (is_solved1.shape[0], 3))
    return states_update, cost_to_go_update, is_solved

#function below is not actually used
def astar_update(states: List[State], env: List[Environment], num_steps: int, heuristic_fn):
    weights: List[float] = list(np.random.rand(len(states)))
    astar1 = AStar(states, env[0], heuristic_fn[0], weights)
    astar2 = AStar(states, env[1], heuristic_fn[1], weights)
    astar3 = AStar(states, env[2], heuristic_fn[2], weights)
    for _ in range(num_steps):
        astar1.step(heuristic_fn[0], 1, verbose=False)
        astar2.step(heuristic_fn[1], 1, verbose=False)
        astar3.step(heuristic_fn[2], 1, verbose=False)

    nodes_popped_layer1: List[List[Node]] = astar1.get_popped_nodes()
    nodes_popped_layer2: List[List[Node]] = astar2.get_popped_nodes()
    nodes_popped_layer3: List[List[Node]] = astar3.get_popped_nodes()
    nodes_popped_flat_layer1: List[Node]
    nodes_popped_flat_layer2: List[Node]
    nodes_popped_flat_layer3: List[Node]
    nodes_popped_flat_layer1, _ = misc_utils.flatten(nodes_popped_layer1)
    nodes_popped_flat_layer2, _ = misc_utils.flatten(nodes_popped_layer2)
    nodes_popped_flat_layer3, _ = misc_utils.flatten(nodes_popped_layer3)

    for node in nodes_popped_flat_layer1 + nodes_popped_flat_layer2 + nodes_popped_flat_layer3:
        node.compute_bellman()

    states_update_layer1: List[State] = [node.state for node in nodes_popped_flat_layer1]
    states_update_layer2: List[State] = [node.state for node in nodes_popped_flat_layer2]
    states_update_layer3: List[State] = [node.state for node in nodes_popped_flat_layer3]
    cost_to_go_update_layer1: np.array = np.array([node.bellman for node in nodes_popped_flat_layer1])
    cost_to_go_update_layer2: np.array = np.array([node.bellman for node in nodes_popped_flat_layer2])
    cost_to_go_update_layer3: np.array = np.array([node.bellman for node in nodes_popped_flat_layer3])

    is_solved_layer1: np.array = np.array(astar1.has_found_goal())
    is_solved_layer2: np.array = np.array(astar2.has_found_goal())
    is_solved_layer3: np.array = np.array(astar3.has_found_goal())

    return [states_update_layer1, states_update_layer2, states_update_layer3], [cost_to_go_update_layer1, cost_to_go_update_layer2, cost_to_go_update_layer3], [is_solved_layer1, is_solved_layer2, is_solved_layer3]

'''update_batch_size is not important, it controls for how many random cubes gbfs performs one-step-look-ahead for at once'''
'''num_states is important, it controls how many random cubes we generate in total '''
def update_runner(num_states: int, back_max: int, update_batch_size: int, heur_fn_i_q, heur_fn_o_q,
                  proc_id: int, env: Environment, result_queue: Queue, num_steps: int, update_method: str,
                  eps_max: float, fixed_difficulty:bool, random: bool, normal_dist:bool):
    heuristic_fn = nnet_utils_multihead.heuristic_fn_queue(heur_fn_i_q, heur_fn_o_q, proc_id, env)

    start_idx: int = 0
    while start_idx < num_states:
        end_idx: int = min(start_idx + update_batch_size, num_states)
        states_itr, _ = env[0].generate_states(end_idx - start_idx, (0, back_max), fixed_difficulty=fixed_difficulty, random=random, normal_dist = normal_dist)

        if update_method.upper() == "GBFS":
            states_update, cost_to_go_update, is_solved = gbfs_update(states_itr, env, num_steps, heuristic_fn, eps_max)
        elif update_method.upper() == "ASTAR":
            states_update, cost_to_go_update, is_solved = astar_update(states_itr, env, num_steps, heuristic_fn)
        else:
            raise ValueError("Unknown update method %s" % update_method)

        states_update_nnet: List[np.ndaray] = env[0].state_to_nnet_input(states_update)

        result_queue.put((states_update_nnet, cost_to_go_update, is_solved))

        start_idx: int = end_idx

    result_queue.put(None)

class Updater:
    def __init__(self, env: List[Environment], num_states: int, back_max: int, heur_fn_i_q, heur_fn_o_qs,
                 num_steps: int, update_method: str, update_batch_size: int = 1000, eps_max: float = 0.0, fixed_difficulty = False, random=False, normal_dist = False):
        super().__init__()
        ctx = get_context("spawn")
        self.num_steps = num_steps
        num_procs = len(heur_fn_o_qs)

        # initialize queues
        self.result_queue: ctx.Queue = ctx.Queue()

        # num states per process
        num_states_per_proc: List[int] = misc_utils.split_evenly(num_states, num_procs)

        self.num_batches: int = int(np.ceil(np.array(num_states_per_proc)/update_batch_size).sum())

        # initialize processes
        self.procs: List[ctx.Process] = []
        for proc_id in range(len(heur_fn_o_qs)):
            num_states_proc: int = num_states_per_proc[proc_id]
            if num_states_proc == 0:
                continue

            proc = ctx.Process(target=update_runner, args=(num_states_proc, back_max, update_batch_size,
                                                           heur_fn_i_q, heur_fn_o_qs[proc_id], proc_id, env,
                                                           self.result_queue, num_steps, update_method, eps_max, fixed_difficulty, random, normal_dist))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

    def update(self):
        states_update_nnet: List[List[np.ndarray]]
        cost_to_go_update: List[np.ndarray]
        is_solved: List[np.ndarray]
        states_update_nnet, cost_to_go_update, is_solved = self._update()

        return states_update_nnet, cost_to_go_update, is_solved

    def _update(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        # process results
        states_update_nnet_l: List[List[np.ndarray]] = []
        cost_to_go_update_l: List = []
        is_solved_l: List = []

        none_count: int = 0
        result_count: int = 0
        display_counts: List[int] = list(np.linspace(1, self.num_batches, 10, dtype=np.int))

        start_time = time.time()

        while none_count < len(self.procs):
            result = self.result_queue.get()
            if result is None:
                none_count += 1
                continue
            result_count += 1

            states_nnet_q: List[np.ndarray]
            states_nnet_q, cost_to_go_q, is_solved_q = result
            states_update_nnet_l.append(states_nnet_q)

            cost_to_go_update_l.append(cost_to_go_q)
            is_solved_l.append(is_solved_q)

            if result_count in display_counts:
                print("%.2f%% (Total time: %.2f)" % (100 * result_count/self.num_batches, time.time() - start_time))

        num_states_nnet_np: int = len(states_update_nnet_l[0])
        states_update_nnet: List[np.ndarray] = []
        for np_idx in range(num_states_nnet_np):
            states_nnet_idx: np.ndarray = np.concatenate([x[np_idx] for x in states_update_nnet_l], axis=0)
            states_update_nnet.append(states_nnet_idx)

        cost_to_go_update: np.ndarray = np.concatenate(cost_to_go_update_l, axis=0)
        is_solved: np.ndarray = np.concatenate(is_solved_l, axis=0)

        for proc in self.procs:
            proc.join()

        print("_update", len(states_update_nnet), cost_to_go_update.shape)
        return states_update_nnet, cost_to_go_update, is_solved
