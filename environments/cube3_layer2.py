from typing import List, Dict, Tuple, Union
import numpy as np
from sympy.combinatorics.permutations import Permutation
from torch import nn
from random import randrange
from environments.generate_cubes import *
from utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State


class Cube3State(State):
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: np.ndarray):
        self.colors: np.ndarray = colors
        self.hash = None
        # print("colors", len(colors), colors)

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.colors.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.colors, other.colors)


class Cube3Layer2(Environment):
    moves: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]
    moves_rev: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [1, -1]]
    # print("moves", moves)
    # print("moves_rev", moves_rev)
    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3

        # solved state
        self.goal_colors: np.ndarray = np.arange(0, (self.cube_len ** 2) * 6, 1, dtype=self.dtype)
        # print("goal_colors", self.goal_colors)
        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, np.ndarray]
        self.rotate_idxs_old: Dict[str, np.ndarray]

        self.adj_faces: Dict[int, np.ndarray]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(self.cube_len, self.moves)

    def next_state(self, states: List[Cube3State], action: int) -> Tuple[List[Cube3State], List[float]]:
        states_np = np.stack([x.colors for x in states], axis=0)
        # print("states_np", states_np)
        states_next_np, transition_costs = self._move_np(states_np, action)

        states_next: List[Cube3State] = [Cube3State(x) for x in list(states_next_np)]

        return states_next, transition_costs

    def prev_state(self, states: List[Cube3State], action: int) -> List[Cube3State]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

        """
    This part is the added part that generates states that fix layer 1.
    """
    def generate_states(self, num_states: int, backwards_range: Tuple[int, int], fixed_difficulty: bool = False, random: bool = False, normal_dist: bool = False) -> Tuple[List[Cube3State], List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        '''first generate perfect cubes'''
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)
        '''from there, generate cubes with the first two layers fixed, and the third layer random'''
        for i in range(num_states):
            args = generate_random_config(fix=3)
            states_np[i] = generate_config([states_np[i]], args[0], args[1], args[2], args[3])[0]
        '''then apply random walk'''
        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()
        # Scrambles
        scramble_nums: np.array
        if normal_dist:
            scramble_nums = np.maximum(0, np.random.normal(backwards_range[1], 3, size=(num_states,))).astype(int)
        else:
            scramble_nums= np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)
        # Go backward
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)
            # print("move states_np", states_np[idxs])
            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: List[Cube3State] = [Cube3State(x) for x in list(states_np)]

        return states, scramble_nums.tolist()

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[Cube3State], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_colors.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[Cube3State] = [Cube3State(self.goal_colors.copy()) for _ in range(num_states)]
        return solved_states

    def is_solved(self, states: List[Cube3State]) -> np.ndarray:
        states_np = np.stack([state.colors for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.goal_colors, 0))

        #the stickers that we care about are the U cubies'
        layer1_and2_cubies_pos = np.array([ 0, 1, 2, 3, 5, 6,7, 8,20 ,23 ,26 ,19,25,47, 50 ,53 , 46, 52,29 ,32 ,35 , 28, 34,38 ,41, 44, 37,43], dtype = np.int32)
        layer1_and2_cubies_one_hot = np.zeros((1, 54), dtype=bool)
        #1 for the stickers we care, 0 for the stickers we don't care
        layer1_and2_cubies_one_hot[0, layer1_and2_cubies_pos] = True
        #there are 28 stickers that we care
        goal_match = np.sum(layer1_and2_cubies_one_hot)

        curr_match = np.sum(is_equal, axis=1, where =layer1_and2_cubies_one_hot)
        #a cube is solved (first layer solved) if all the stickers we care about are in the right positions
        is_solved =  curr_match == goal_match
        return is_solved


    def state_to_nnet_input(self, states: List[Cube3State]) -> List[np.ndarray]:
        states_np = np.stack([state.colors for state in states], axis=0)

        representation_np: np.ndarray = states_np / (self.cube_len ** 2)
        representation_np: np.ndarray = representation_np.astype(self.dtype)

        representation: List[np.ndarray] = [representation_np]

        return representation

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self, nnet_type: str = "baseline", model_name) -> nn.Module:
        state_dim: int = (self.cube_len ** 2) * 6
        out_dim = None
        if nnet_type == "baseline":
            out_dim = 1
        elif nnet_type == "multihead":
            out_dim = 3
        model = getModel(model_name)
        nnet = model(state_dim, 6, 5000, 1000, 4, out_dim, True)
        return nnet
    def expand(self, states: List[State], options: List[List[str]]= []) -> Tuple[List[List[State]], List[np.ndarray]]:
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()
        num_options = len(options)

        states_exp: List[List[State]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves+ num_options])

        # numpy states
        states_np: np.ndarray = np.stack([state.colors for state in states])

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        move: int
        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np(states_np, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(Cube3State(states_next_np[idx]))

        for move_idx in range(num_options):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np_option(states_np, options[move_idx])

            # transition cost
            tc[:, move_idx+num_env_moves] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(Cube3State(states_next_np[idx]))


        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def _move_np(self, states_np: np.ndarray, action: int):
        action_str: str = self.moves[action]
        # print("action_str", action_str)
        # print("states_np before move", states_np)
        states_next_np: np.ndarray = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_str]] = states_np[:, self.rotate_idxs_old[action_str]]
        # print("self.rotate_idxs_new[action_str]", self.rotate_idxs_new[action_str])
        # print("self.rotate_idxs_old[action_str]", self.rotate_idxs_old[action_str])

        # print("states_next_np after move", states_next_np)
        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

    def _move_np_option(self, states_np: np.ndarray, option: List[str]):

        states_next_np: np.ndarray = states_np.copy()
        for action_str in option:
            states_next_np[:, self.rotate_idxs_new[action_str]] = states_next_np[:, self.rotate_idxs_old[action_str]]
        #transition cost = length of option
        transition_costs: List[float] = [len(option) for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs


    def _get_adj(self) -> None:
        # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
        self.adj_faces: Dict[int, np.ndarray] = {0: np.array([2, 5, 3, 4]),
                                                 1: np.array([2, 4, 3, 5]),
                                                 2: np.array([0, 4, 1, 5]),
                                                 3: np.array([0, 5, 1, 4]),
                                                 4: np.array([0, 3, 1, 2]),
                                                 5: np.array([0, 2, 1, 3])
                                                 }

    def _compute_rotation_idxs(self, cube_len: int,
                               moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        rotate_idxs_new: Dict[str, np.ndarray] = dict()
        rotate_idxs_old: Dict[str, np.ndarray] = dict()

        for move in moves:
            f: str = move[0]
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5

            adj_idxs = {0: {2: [range(0, cube_len), cube_len - 1], 3: [range(0, cube_len), cube_len - 1],
                            4: [range(0, cube_len), cube_len - 1], 5: [range(0, cube_len), cube_len - 1]},
                        1: {2: [range(0, cube_len), 0], 3: [range(0, cube_len), 0], 4: [range(0, cube_len), 0],
                            5: [range(0, cube_len), 0]},
                        2: {0: [0, range(0, cube_len)], 1: [0, range(0, cube_len)],
                            4: [cube_len - 1, range(cube_len - 1, -1, -1)], 5: [0, range(0, cube_len)]},
                        3: {0: [cube_len - 1, range(0, cube_len)], 1: [cube_len - 1, range(0, cube_len)],
                            4: [0, range(cube_len - 1, -1, -1)], 5: [cube_len - 1, range(0, cube_len)]},
                        4: {0: [range(0, cube_len), cube_len - 1], 1: [range(cube_len - 1, -1, -1), 0],
                            2: [0, range(0, cube_len)], 3: [cube_len - 1, range(cube_len - 1, -1, -1)]},
                        5: {0: [range(0, cube_len), 0], 1: [range(cube_len - 1, -1, -1), cube_len - 1],
                            2: [cube_len - 1, range(0, cube_len)], 3: [0, range(cube_len - 1, -1, -1)]}
                        }
            face_dict = {'U': 0, 'D': 1, 'L': 2, 'R': 3, 'B': 4, 'F': 5}
            face = face_dict[f]

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[(np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to))) % len(faces_to)]

            cubes_idxs = [[0, range(0, cube_len)], [range(0, cube_len), cube_len - 1],
                          [cube_len - 1, range(cube_len - 1, -1, -1)], [range(cube_len - 1, -1, -1), 0]]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[(np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to))) % len(cubes_to)]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_to[i]][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_from[i]][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            for i in range(0, len(faces_to)):
                face_to = faces_to[i]
                face_from = faces_from[i]
                idxs_new = [[idx1, idx2] for idx1 in np.array([face_idxs[face_to][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_to][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([face_idxs[face_from][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_from][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face_to, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face_from, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

        # print("rotate_idxs_new[U1]", rotate_idxs_new["U1"])
        # print("rotate_idxs_old[U1]", rotate_idxs_old["U1"])

        # print("rotate_idxs_new[F1]", rotate_idxs_new["F1"])
        # print("rotate_idxs_old[F1]", rotate_idxs_old["F1"])

        # print("rotate_idxs_new[B1]", rotate_idxs_new["B1"])
        # print("rotate_idxs_old[B1]", rotate_idxs_old["B1"])

        # print("rotate_idxs_new[L1]", rotate_idxs_new["L1"])
        # print("rotate_idxs_old[L1]", rotate_idxs_old["L1"])

        # print("rotate_idxs_new[R1]", rotate_idxs_new["R1"])
        # print("rotate_idxs_old[R1]", rotate_idxs_old["R1"])

        # print("rotate_idxs_new[D1]", rotate_idxs_new["D1"])
        # print("rotate_idxs_old[D1]", rotate_idxs_old["D1"])

        return rotate_idxs_new, rotate_idxs_old
