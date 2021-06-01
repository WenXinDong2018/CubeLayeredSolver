import csv
import pickle
from typing import List, Dict
from utils.search_utils import create_options
def parse():
    sequences = []
    for i in range(10):
        with open("search_methods/dataset/training.{}".format(i), 'r') as custfile:
            rows = csv.reader(custfile, delimiter=',')
            for id, r in enumerate(rows):
                if id % 2 == 1:
                    sequences.append(r[0])
    return create_options(sequences)

def compile_moves_of_length(length: int, moves: List[List[str]]) -> Dict[List[str], int]:
    output = {}
    for id, sequence in enumerate(moves):
        for j in range(len(sequence) - length + 1):
            mov_str = ' '.join(sequence[j : j + length])
            if mov_str not in output:
                output[mov_str] = 0
            output[mov_str] += 1
    return dict(sorted(output.items(), key=lambda item: item[1], reverse=True))

def main():
    sequences = parse()
    for i in range(4, 13):
        print('compiling data for length {}...'.format(i))
        moves = compile_moves_of_length(i, sequences)
        pickle.dump(moves, open("search_methods/options_data/options_of_length{}.pkl".format(i), "wb"), protocol=-1)

    

if __name__ == "__main__":
    # needs to get the data set first.
    main()

