import numpy as np
from typing import List, Tuple, Set, Callable, Optional
from sympy.combinatorics.permutations import Permutation


def get_fixed_moves() -> List[str]:
    moves_with_top_two_layers_fixed = ['edge_perm, edge_twist, corner_perm, corner_twist']

def edge_permutation(states: List[np.ndarray], choice_of_edges: int, sign: int) -> np.ndarray:
    # edges in the last layers are 21, 30, 39, 48. We only can permutes three of them at a time
    # choice_of_edges can take either 0, 1, 2, 3. Each one represents a set of three edges that we will permute.
    # sign: given a set of edges that we want to permute, there's two ways to permute them.
    set_to_choose: Dict[int, np.ndarray] = {0: np.array([21, 30, 39]), 1: np.array([21, 39, 48]), 2: np.array([21, 30, 48]), 3: np.array([30, 39, 48])}
    output_states_np = np.stack([state for state in states])
    indices = set_to_choose[choice_of_edges]
    values = output_states_np[:, indices]

    # if sign == 1, we send the first element in perm_set to the second, second to last, then the last to first
    # if sign == 0, we send the first element in perm_set to the last, second to the first, then the last to the second
    perm_arr = np.array([1, 2, 0]) if sign == 1 else np.array([2, 0, 1])
    output_states_np[:, indices[perm_arr]] = values
    return output_states_np

def edge_twist(states: List[np.ndarray], choice_of_edges: int, sign: int) -> np.ndarray:
    # edges in the last layers are 21, 30, 39, 48. We only can twist two of them at a time
    # choice_of_edges can take either 0, 1, 2, 3, 4, 5. Each one represents a set of two edges that we will twist.
    set_to_choose: Dict[int, np.ndarray] = {0: np.array([21, 30]), 1: np.array([21, 39]), 2: np.array([21, 48]), 3: np.array([30, 39]), 4: np.array([30, 48]), 5: np.array([39, 48])}
    correspondence: Dict[int, np.ndarray] = {21: np.array([10, 21]), 30: np.array([16, 30]), 39: np.array([12, 39]), 48: np.array([14, 48])}
    output_states_np = np.stack([state for state in states])
    perm_set = set_to_choose[choice_of_edges]

    edge_indices = np.concatenate((correspondence[perm_set[0]], correspondence[perm_set[1]]))
    edge_values = output_states_np[:, edge_indices]
    output_states_np[:, edge_indices[np.array([1, 0, 3, 2])]] = edge_values
    return output_states_np
def corner_permutation(states: List[np.ndarray], choice_of_corners: int, sign: int) ->  np.ndarray:
    # edges in the last layers are 21, 30, 39, 48. We only can permutes three of them at a time
    # choice_of_edges can take either 0, 1, 2, 3. Each one represents a set of three edges that we will permute.
    # sign: given a set of edges that we want to permute, there's two ways to permute them.
    set_to_choose: Dict[int, np.ndarray] = {0: np.array([9, 11, 15]), 1: np.array([9, 15, 17]), 2: np.array([9, 11, 17]), 3: np.array([11, 15, 17])}
    correspondence: Dict[int, np.ndarray] = {9: np.array([9, 42, 18]), 11: np.array([11, 24, 45]), 15: np.array([15, 33, 36]), 17: np.array([17, 51, 21])}
    output_states_np = np.stack([state for state in states])
    D_idx = set_to_choose[choice_of_corners]
    indices = np.concatenate((D_idx, [correspondence[D_idx[0]][1], correspondence[D_idx[1]][1], correspondence[D_idx[2]][1]], [correspondence[D_idx[0]][2], correspondence[D_idx[1]][2], correspondence[D_idx[2]][2]]))
    values = output_states_np[:, indices]
    # if sign == 1, we send the first element in perm_set to the last, second to the first, then the last to the second with each corner twisted clockwise
    # if sign == 0, we send the first element in perm_set to the second, second to last, then the last to first with each corner twisted counter-clockwise
    perm_arr: np.ndarray = np.array([1, 2, 0, 4, 5, 3, 7, 8, 6]) if sign == 1 else np.array([2, 0, 1, 5, 3, 4, 8, 6, 7])
    output_states_np[:, indices[perm_arr]] = values
    return output_states_np

def corner_twist(states: List[np.ndarray], choice_of_edges: int, sign: int) -> np.ndarray:
    # edges in the last layers are 21, 30, 39, 48. We only can twist two of them at a time
    # choice_of_edges can take either 0, 1, 2, 3. Each one represents a set of two on one edge.
    set_to_choose: Dict[int, np.ndarray] = {0: np.array([9, 11]), 1: np.array([9, 15]), 2: np.array([11, 17]), 3: np.array([15, 17])}
    correspondence: Dict[int, np.ndarray] = {9: np.array([9, 42, 18]), 11: np.array([11, 24, 45]), 15: np.array([15, 33, 36]), 17: np.array([17, 51, 21])}
    output_states_np = np.stack([state for state in states])
    D_idx = set_to_choose[choice_of_edges]
    corner_indices = np.concatenate((correspondence[D_idx[0]], correspondence[D_idx[1]]))
    corner_values = output_states_np[:, corner_indices]
    # if sign == 1, then we assume that we are doing a counter-clock wise twist on the (D_idx[0], D_idx[1], D_idx[2]) corner, and a clock wise twist on the other corner.
    # if sign == 0, then the opposite twist applies
    perm_arr = np.array([2, 0, 1, 4, 5, 0]) if sign == 1 else np.array([1, 2, 0, 5, 3, 4])
    output_states_np[:, corner_indices[perm_arr]] = corner_values
    return output_states_np

def get_all_possible_fixed_moves() -> List[str]:
    output: List[str]
    # edge_perm: total 8 choices
    edge_perms = ['0 %i %i' % (c, s) for c in range(4) for s in range(2)]
    # edge_twist
    edge_twists = ['1 %i %i' % (c, 1) for c in range(6)]
    # corner_perm
    corner_perms = ['2 %i %i' % (c, s) for c in range(4) for s in range(2)]
    # corner_twist
    corner_twists = ['3 %i %i' % (c, s) for c in range(4) for s in range(2)]
    return edge_perms + edge_twists + corner_perms + corner_twists

def fixed_move_dict():
    d = {0:edge_permutation, 1:edge_twist, 2:corner_permutation, 3:corner_twist}
    return d

def get_corners() -> np.ndarray:
    # 8 corners oriented clockwise.
    return np.array([[0, 47, 26], [2, 20, 44], [8, 38, 35], [6, 29, 53], [11, 24, 45], [9, 42, 18], [15, 33, 36], [17, 51, 27]])

def get_edges() -> np.ndarray:
    # 12 oriented corners.
    return np.array([[3, 50], [1, 23], [5, 41], [7, 32], [46, 25], [19, 43], [37, 34], [28, 52], [14, 48], [10, 21], [12, 31], [16, 30]])

def generate_config(states: List[np.ndarray], corners_perm: np.ndarray, edge_perm: np.ndarray, corner_signs: np.ndarray, edge_signs: np.ndarray) -> np.ndarray:
    # corner_perm is a permutation of 0 to 7, with list[i] being that position that i-th corner will be sent to.
    # edge_perm is a permutation of 0 to 11, with list[i] being that position that i-th edge will be sent to.
    # corner_sign is a length 8 ordered list of number 0 to 2, with i-th number indicating a orientation of the i-th corner
    # edge_sign is a length 12 ordered list of number 0 to 1, with i-th number indicating a orientation of the i-th edge
    output_states_np = np.stack([state for state in states])
    corners =get_corners()
    edges =get_edges()
    corner_perm_map = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    edge_perm_map = np.array([[0, 1], [1, 0]])
    corner_values = np.stack([np.concatenate([state[corners[i]][corner_perm_map[corner_signs[i]]] for i in range(8)]) for state in states])
    edge_values = np.stack([np.concatenate([state[edges[i]][edge_perm_map[edge_signs[i]]] for i in range(12)]) for state in states])
    output_states_np[:, corners.flatten()] = corner_values
    output_states_np[:, edges.flatten()] = edge_values
    return output_states_np

def generate_random_config(fix: int) -> List[np.ndarray]:
    corners_perm: np.ndarray
    edges_perm: np.ndarray
    corner_signs: np.ndarray
    edge_signs: np.ndarray
    while True:
        if fix != 1:
            corners_perm = np.concatenate((np.arange(4), np.random.permutation(4) + 4))
            edges_perm = np.concatenate((np.arange(4), np.random.permutation(8) + 4)) if fix == 2 else np.concatenate((np.arange(8), np.random.permutation(4) + 8))
        else:
            # fixes nothing
            corners_perm = np.random.permutation(8)
            edges_perm = np.random.permutation(12)

        c_perm = Permutation(corners_perm.tolist())
        e_perm = Permutation(edges_perm.tolist())
        if c_perm.signature() == e_perm.signature():
            break
    while True:
        corner_signs = np.concatenate((np.zeros(4, dtype=int), np.random.randint(3, size=4))) if fix != 1 else   np.random.randint(3, size=8)
        if np.sum(corner_signs) % 3 == 0:
            break
    while True:
        if fix != 1:
            edge_signs = np.concatenate((np.zeros(4, dtype=int), np.random.randint(2, size=8))) if fix == 2 else np.concatenate((np.zeros(8, dtype=int), np.random.randint(2, size=4)))
        else:
            edge_signs = np.random.randint(2, size=12)
        if np.sum(edge_signs) % 2 == 0:
            break
    return [corners_perm, edges_perm, corner_signs, edge_signs]

