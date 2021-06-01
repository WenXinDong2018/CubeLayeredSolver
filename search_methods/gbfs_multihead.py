from typing import List, Tuple, Set, Callable, Optional
from environments.environment_abstract import Environment, State
import numpy as np
from utils import search_utils, env_utils, nnet_utils, misc_utils
from argparse import ArgumentParser

import torch
import wandb


class Instance:

    def __init__(self, state: State, eps: float):
        self.curr_state: State = state
        self.is_solved: bool = False
        self.num_steps: int = 0
        self.trajs: List[Tuple[State, float]] = []
        self.seen_states: Set[State] = set()

        self.eps = eps

    def add_to_traj(self, state: State, cost_to_go: float):
        self.trajs.append((state, cost_to_go))
        self.seen_states.add(state)

    def next_state(self, state: State):
        self.curr_state = state
        self.num_steps += 1


class GBFS:
    def __init__(self, states: List[State], env: Environment, eps: Optional[List[float]] = None):
        self.curr_states: List[State] = states
        self.env: Environment = env

        if eps is None:
            eps = [0] * len(self.curr_states)

        self.instances: List[Instance] = []
        for state, eps_inst in zip(states, eps):
            instance: Instance = Instance(state, eps_inst)
            self.instances.append(instance)

    def step(self, heuristic_fn: Callable, idx:int = 0) -> None:
        # check which are solved
        self._record_solved()

        # take a step for unsolved states
        self._move(heuristic_fn, idx = idx)

    def get_trajs(self) -> List[List[Tuple[State, float]]]:
        trajs_all: List[List[Tuple[State, float]]] = []
        for instance in self.instances:
            trajs_all.append(instance.trajs)

        return trajs_all

    def get_is_solved(self) -> List[bool]:
        is_solved: List[bool] = [x.is_solved for x in self.instances]

        return is_solved

    def get_num_steps(self) -> List[int]:
        num_steps: List[int] = [x.num_steps for x in self.instances]

        return num_steps

    def _record_solved(self) -> None:
        # get unsolved instances
        instances: List[Instance] = self._get_unsolved_instances()
        if len(instances) == 0:
            return

        states: List[State] = [instance.curr_state for instance in instances]

        is_solved: np.ndarray = self.env.is_solved(states)

        solved_idxs: List[int] = list(np.where(is_solved)[0])
        if len(solved_idxs) > 0:
            instances_solved: List[Instance] = [instances[idx] for idx in solved_idxs]
            states_solved: List[State] = [instance.curr_state for instance in instances_solved]

            for instance, state in zip(instances_solved, states_solved):
                instance.add_to_traj(state, 0.0)
                instance.is_solved = True

    def _move(self, heuristic_fn: Callable, idx:int=0) -> None:
        # get unsolved instances
        instances: List[Instance] = self._get_unsolved_instances()
        if len(instances) == 0:
            return
        states: List[State] = [instance.curr_state for instance in instances]

        # get backed-up ctg and cost of each move
        ctg_backups: np.ndarray
        ctg_next_p_tcs: List[np.ndarray]
        states_exp: List[List[State]]
        ctg_backups, ctg_next_p_tcs, states_exp = search_utils.bellman(states, heuristic_fn, self.env, idx = idx)

        # make move
        for idx in range(len(instances)):
            # add state to trajectory
            instance: Instance = instances[idx]
            state: State = states[idx]
            ctg_backup: float = ctg_backups[idx]

            instance.add_to_traj(state, ctg_backup)

            # get next state
            ctg_next_p_tc: np.ndarray = ctg_next_p_tcs[idx]
            state_exp: List[State] = states_exp[idx]
            state_next: State = state_exp[int(np.argmin(ctg_next_p_tc))]

            # make random move with probability eps
            eps_rand_move = np.random.random(1)[0] < instance.eps
            seen_state: bool = state_next in instance.seen_states
            if eps_rand_move or seen_state:
                rand_state_idx = np.random.choice(len(state_exp))
                state_next = state_exp[rand_state_idx]

            instance.next_state(state_next)

    def _get_unsolved_instances(self) -> List[Instance]:
        instances_unsolved: List[Instance] = [instance for instance in self.instances if not instance.is_solved]
        return instances_unsolved


###heuristic_fn is function that return an array
def gbfs_test(num_states: int, back_max: int, env: List[Environment], heuristic_fn: Callable,
              max_solve_steps: Optional[int] = None, dynamic_back_max = None, random: bool=False):

    env1 = env[0]
    env2 = env[1]
    env3 = env[2]
    # get data
    back_steps: List[int]
    if random:
        print("gbfs_test, random = True")
        back_steps = [back_max]
    else:
        print("gbfs_test, random = False")
        back_steps = list(np.linspace(0, back_max, 30, dtype=np.int))
    num_states_per_back_step: List[int] = misc_utils.split_evenly(num_states, len(back_steps))
    states: List[State] = []
    state_back_steps_l: List[int] = []

    for back_step, num_states_i in zip(back_steps, num_states_per_back_step):
        if num_states_i > 0:
            states_i, back_steps_i = env3.generate_states(num_states_i, (back_step, back_step), random=random)
            states.extend(states_i)
            state_back_steps_l.extend(back_steps_i)

    state_back_steps: np.ndarray = np.array(state_back_steps_l)
    if max_solve_steps is None:
        max_solve_steps = max(np.max(state_back_steps), 1)

    # Do GBFS for each back step
    print("Solving %i states with GBFS with %i steps" % (len(states), max_solve_steps))

    # Solve with GBFS
    gbfs_layer1 = GBFS(states, env1, eps=None)
    gbfs_layer2 = GBFS(states, env2, eps=None)
    gbfs_layer3 = GBFS(states, env3, eps=None)


    for _ in range(max_solve_steps):
        gbfs_layer1.step(heuristic_fn, idx = 0)
        gbfs_layer2.step(heuristic_fn, idx = 1)
        gbfs_layer3.step(heuristic_fn, idx = 2)


    is_solved_all_layer1: np.ndarray = np.array(gbfs_layer1.get_is_solved())
    num_steps_all_layer1: np.ndarray = np.array(gbfs_layer1.get_num_steps())
    is_solved_all_layer2: np.ndarray = np.array(gbfs_layer2.get_is_solved())
    num_steps_all_layer2: np.ndarray = np.array(gbfs_layer2.get_num_steps())
    is_solved_all_layer3: np.ndarray = np.array(gbfs_layer3.get_is_solved())
    num_steps_all_layer3: np.ndarray = np.array(gbfs_layer3.get_num_steps())

    # Get state cost-to-go
    # state_ctg_all: np.ndarray = heuristic_fn(states)
    per_solved_record = None  #per_solved of problems generated with dynamic_back_max steps
    for back_step_test in np.unique(state_back_steps):
        # Get states
        step_idxs = np.where(state_back_steps == back_step_test)[0]
        if len(step_idxs) == 0:
            continue

        is_solved_layer1: np.ndarray = is_solved_all_layer1[step_idxs]
        num_steps_layer1: np.ndarray = num_steps_all_layer1[step_idxs]
        is_solved_layer2: np.ndarray = is_solved_all_layer2[step_idxs]
        num_steps_layer2: np.ndarray = num_steps_all_layer2[step_idxs]
        is_solved_layer3: np.ndarray = is_solved_all_layer3[step_idxs]
        num_steps_layer3: np.ndarray = num_steps_all_layer3[step_idxs]

        # Get stats
        per_solved_layer1 = 100 * float(sum(is_solved_layer1)) / float(len(is_solved_layer1))
        per_solved_layer2 = 100 * float(sum(is_solved_layer2)) / float(len(is_solved_layer2))
        per_solved_layer3 = 100 * float(sum(is_solved_layer3)) / float(len(is_solved_layer3))

        avg_solve_steps_layer1 = 0.0
        avg_solve_steps_layer2 = 0.0
        avg_solve_steps_layer3 = 0.0

        if per_solved_layer1 > 0.0:
            avg_solve_steps_layer1 = np.mean(num_steps_layer1[is_solved_layer1])
        if per_solved_layer2 > 0.0:
            avg_solve_steps_layer2 = np.mean(num_steps_layer2[is_solved_layer2])
        if per_solved_layer3 > 0.0:
            avg_solve_steps_layer3 = np.mean(num_steps_layer3[is_solved_layer3])

        # Print results
        wandb.log({"GBFS-Solved-L1" + str(back_step_test) : per_solved_layer1})
        wandb.log({"GBFS-Solved-L2" + str(back_step_test) : per_solved_layer2})
        wandb.log({"GBFS-Solved-L3" + str(back_step_test) : per_solved_layer3})

        wandb.log({"GBFS-avgSolveSteps-L1" + str(back_step_test): avg_solve_steps_layer1})
        wandb.log({"GBFS-avgSolveSteps-L2" + str(back_step_test): avg_solve_steps_layer2})
        wandb.log({"GBFS-avgSolveSteps-L3" + str(back_step_test): avg_solve_steps_layer3})

    return per_solved_record


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of nnet model")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of data")

    parser.add_argument('--env', type=str, required=True, help="Environment: cube3, 15-puzzle, 24-puzzle")
    parser.add_argument('--max_steps', type=int, default=None, help="Maximum number ofsteps to take when solving "
                                                                    "with GBFS. If none is given, then this "
                                                                    "is set to the maximum number of "
                                                                    "backwards steps taken to create the "
                                                                    "data")
    parser.add_argument('--model_name', type=str, required=True, help="Which model to use: options listed in pytorch_models.py")

    args = parser.parse_args()

    # environment
    env: Environment = env_utils.get_environment(args.env)

    # get device and nnet
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    heuristic_fn = nnet_utils.load_heuristic_fn(args.model_dir, device, on_gpu, env.get_nnet_model(nnet_type= "multihead", model_name = args["model_name"]),
                                                env, clip_zero=False)

    gbfs_test(args.data_dir, env, heuristic_fn, max_solve_steps=args.max_steps)


if __name__ == "__main__":
    main()