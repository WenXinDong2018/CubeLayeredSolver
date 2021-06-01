from typing import List, Tuple
import numpy as np
from environments.environment_abstract import Environment, State
from utils import misc_utils


def is_valid_soln(state: State, soln: List[int], env: Environment) -> bool:
    soln_state: State = state
    move: int
    for move in soln:
        soln_state = env.next_state([soln_state], move)[0][0]

    return env.is_solved([soln_state])[0]


def bellman(states: List, heuristic_fn, env: Environment, idx:int = -1) -> Tuple[np.ndarray, List[np.ndarray], List[List[State]]]:
    # expand states
    states_exp, tc_l = env.expand(states)
    tc = np.concatenate(tc_l, axis=0)

    # get cost-to-go of expanded states
    states_exp_flat, split_idxs = misc_utils.flatten(states_exp)
    ctg_next: np.ndarray
    if idx>=0:
        ctg_next = heuristic_fn(states_exp_flat)
        ctg_next = ctg_next[:, idx]
    else:
        ctg_next = heuristic_fn(states_exp_flat)

    # backup cost-to-go
    ctg_next_p_tc = tc + ctg_next
    ctg_next_p_tc_l = np.split(ctg_next_p_tc, split_idxs)

    is_solved = env.is_solved(states)
    ctg_backup = np.array([np.min(x) for x in ctg_next_p_tc_l]) * np.logical_not(is_solved)

    return ctg_backup, ctg_next_p_tc_l, states_exp

def create_options(strs: List[str]) -> List[List[str]]:
    # parse moves in string to
    move_dict = {'u': 'D', 'd': 'U', 'r': 'L', 'l': 'R', 'f': 'B', 'b': 'F', 'D': 'D', 'U': 'U', 'L': 'L', 'R': 'R', 'B': 'B', 'F': 'F'}
    output = []
    for i in range(len(strs)):
        moves = strs[i].split(' ')
        new_moves = []
        for move in moves:
            if move[0] == 'M':
                if len(move) == 1:
                    new_moves.append('R' + '1')
                    new_moves.append('L' + '-1')
                    continue
                sign1 = '-1' if '\'' in move[1] else '1'
                sign2 = '1' if '\'' in move[1] else '-1'
                if move[-1] == '2':
                    new_moves.append('R' + sign1)
                    new_moves.append('L' + sign2)
                    new_moves.append('R' + sign1)
                    new_moves.append('L' + sign2)
                else:
                    new_moves.append('R' + sign1)
                    new_moves.append('L' + sign2)
                continue
            if len(move) == 1:
                new_moves.append(move_dict[move[0]] + '1')
                continue
            sign = '-1' if '\'' in move else '1'
            if '2' in move:
                new_moves.append(move_dict[move[0]] + sign)
                new_moves.append(move_dict[move[0]] + sign)
            else:
                new_moves.append(move_dict[move[0]] + sign)

        output.append(new_moves)
    return output

