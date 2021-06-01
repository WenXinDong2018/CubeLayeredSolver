import re
import math
from environments.environment_abstract import Environment


def get_environment(env_name: str) -> Environment:
    env_name = env_name.lower()
    puzzle_n_regex = re.search("puzzle(\d+)", env_name)
    env: Environment

    if env_name == 'cube3': #cube3 is layer 3 environment. Solves entire cube
        from environments.cube3 import Cube3
        env = Cube3()
    elif env_name == 'cube3_layer1':
        from environments.cube3_layer1 import Cube3Layer1
        env = Cube3Layer1()
    elif env_name == 'cube3_layer2':
        from environments.cube3_layer2 import Cube3Layer2
        env = Cube3Layer2()
    else:
        raise ValueError('No known environment %s' % env_name)

    return env
