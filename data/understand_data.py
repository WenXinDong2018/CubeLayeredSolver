import pickle5 as pickle
from environments.cube3_layer2 import Cube3State

from utils import search_utils, env_utils
env1 = env_utils.get_environment("cube3_layer1")
env2 = env_utils.get_environment("cube3_layer2")
env3 = env_utils.get_environment("cube3")

results = pickle.load(open("data/cube3/test/data_0.pkl", "rb"))

l1_solved = 0
l2_solved = 0
l3_solved = 0
for state in results['states']:
    if env1.is_solved([state])[0]:
        l1_solved+=1
    if env2.is_solved([state])[0]:
        l2_solved+=1
    if env3.is_solved([state])[0]:
        l3.solved+=1

print("layer 1: ", l1_solved)
print("layer 2: ", l2_solved)
print("layer 3: ", l3_solved)