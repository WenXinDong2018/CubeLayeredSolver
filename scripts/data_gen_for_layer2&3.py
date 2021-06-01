from typing import List, Dict, Tuple, Union
import numpy as np 
from sympy.combinatorics.permutations import Permutation
import pickle
import os
import time
from environments.cube3_layer2 import Cube3State

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
    corners = get_corners()
    edges = get_edges()
    corner_perm_map = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    edge_perm_map = np.array([[0, 1], [1, 0]])
    corner_values = np.stack([np.concatenate([state[corners[i]][corner_perm_map[corner_signs[i]]] for i in range(8)]) for state in states])
    edge_values = np.stack([np.concatenate([state[edges[i]][edge_perm_map[edge_signs[i]]] for i in range(12)]) for state in states])
    output_states_np[:, corners.flatten()] = corner_values
    output_states_np[:, edges.flatten()] = edge_values
    return output_states_np

def generate_random_config(fix: int) -> List[np.ndarray]:
    corners_perm: np.zeros()
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
        corner_signs = np.concatenate((np.zeros(4, dtype=int), np.random.randint(3, size=4))) if fix != 1 else np.random.randint(3, size=8)
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

def generate_data_for_layer(fix: int, num_states: int, filepath):
    # generate data
    start_time = time.time()
    print("Generating data for %s" % filepath)
    states = []
    for _ in range(num_states):
        config = generate_random_config(fix)
        state = generate_config([np.arange(54)], config[0], config[1], config[2], config[3])[0]
        states.append(Cube3State(state))

    data_gen_time = time.time() - start_time

    # save data
    start_time = time.time()

    data = dict()
    data['states'] = states
    data['num_back_steps'] = [0 * num_states]

    pickle.dump(data, open(filepath, "wb"), protocol=-1)

    save_time = time.time() - start_time

    print("%s - Data Gen Time: %s, Save Time: %s" % (filepath, data_gen_time, save_time))

def main():
    generate_data_for_layer(fix=2, num_states=1000, filepath="data/cube3_layer2/test/data_0.pkl")
    generate_data_for_layer(fix=3, num_states=1000, filepath="data/cube3_layer3/test/data_0.pkl")


main()