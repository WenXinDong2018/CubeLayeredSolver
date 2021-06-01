import numpy as np
from sympy.combinatorics.permutations import Permutation
from environments.cube3_layer1 import Cube3Layer1
from environments.cube3_layer2 import Cube3Layer2
env2 = Cube3Layer2()
env1 = Cube3Layer1()
states2, _ = env2.generate_states(1, [0, 1], fixed_difficulty=False, random=True)
env2.is_solved(states2)
