from search_methods.astar import *


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--states', type=str, required=True, help="File containing states to solve")
    parser.add_argument('--model_dir_layer1', type=str, required=True, help="Directory of nnet layer 1 model")
    parser.add_argument('--model_dir_layer2', type=str, required=True, help="Directory of nnet layer 2 model")
    parser.add_argument('--model_dir_layer3', type=str, required=True, help="Directory of nnet layer 3 model")

    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for BWAS")
    parser.add_argument('--weight', type=float, default=1.0, help="Weight of path cost")

    parser.add_argument('--results_dir', type=str, required=True, help="Directory to save results")
    parser.add_argument('--start_idx', type=int, default=0, help="")
    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect final results, "
                                                                          "but will help if nnet is running out of "
                                                                          "memory.")

    parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging")
    parser.add_argument('--max_nodes', type=int, default=600000, help="Set cap on number of nodes to explore, per layer.")
    parser.add_argument('--options', action='store_true', default=False, help="Use options when doing search")
    parser.add_argument('--option_name', type = str, help="Which layer options when doing search: layer2, layer3")
    parser.add_argument('--model_name', type=str, required=True, help="Which model to use: options listed in pytorch_models.py")

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    results_file: str = "%s/results.pkl" % args.results_dir
    output_file: str = "%s/output.txt" % args.results_dir
    if not args.debug:
        sys.stdout = data_utils.Logger(output_file, "w")

    # get data
    input_data = pickle.load(open(args.states, "rb"))
    states: List[State] = input_data['states'][args.start_idx:]
    print("solving #", len(states), "problems")

    # initialize results
    results: Dict[str, Any] = dict()
    results["layer1_states"] = states

    # solve layer 1
    env: Environment = env_utils.get_environment("cube3_layer1")
    args.model_dir = args.model_dir_layer1
    solns_l1, paths_l1, times_l1, num_nodes_gen_l1 = bwas_python(args, env, states, nnet_type="baseline")
    times = list(filter(lambda x: x != None, times_l1))
    paths = list(filter(lambda x: x != None, paths_l1))
    num_nodes_gen = list(filter(lambda x: x != None, num_nodes_gen_l1))
    solns = list(filter(lambda x: x != None, solns_l1))
    #update remaining
    states_l2 = []
    for path in paths_l1:
        if path: states_l2.append(path[-1])
    #solve layer 2
    args.model_dir = args.model_dir_layer2
    results["layer2_states"] = states_l2
    env = env_utils.get_environment("cube3_layer2")
    solns_l2, paths_l2, times_l2, num_nodes_gen_l2 = bwas_python(args, env, states_l2, nnet_type="baseline")
    print("finished solving layer 2", [p!=None for p in paths_l2])
    new_times = []
    new_paths = []
    new_num_nodes_gen = []
    new_sols = []
    states_l3 = []
    for idx, path in enumerate(paths_l2):
        if path:
            states_l3.append(path[-1])
            new_times.append(times[idx] + times_l2[idx])
            new_paths.append(paths[idx] + paths_l2[idx][1:])
            new_sols.append(solns[idx] + solns_l2[idx])
            new_num_nodes_gen.append(num_nodes_gen[idx] + num_nodes_gen_l2[idx])
    times = new_times
    paths = new_paths
    num_nodes_gen = new_num_nodes_gen
    solns = new_sols
    #solve layer 3
    args.model_dir = args.model_dir_layer3
    env = env_utils.get_environment("cube3")
    results["layer3_states"] = states_l3
    solns_l3, paths_l3, times_l3, num_nodes_gen_l3 = bwas_python(args, env, states_l3, nnet_type="baseline")
    new_sols = []
    new_times = []
    new_paths = []
    new_num_nodes_gen = []
    for idx, path in enumerate(paths_l3):
        if path:
            new_sols.append(solns[idx] + solns_l3[idx])
            new_times.append(times[idx] + times_l3[idx])
            new_paths.append(paths[idx] + paths_l3[idx][1:])
            new_num_nodes_gen.append(num_nodes_gen[idx] + num_nodes_gen_l3[idx])
    times = new_times
    paths = new_paths
    num_nodes_gen = new_num_nodes_gen
    solns = new_sols

    results["solutions"] = solns
    results["paths"] = paths
    results["times"] = times
    results["num_nodes_generated"] = num_nodes_gen

    pickle.dump(results, open(results_file, "wb"), protocol=-1)

if __name__ == "__main__":
    main()
