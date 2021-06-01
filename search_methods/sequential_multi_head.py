from utils import nnet_utils_multihead
from search_methods.astar import *


def bwas_python_multi_head(args, env: Environment, states: List[State], layer: int):
    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))
    # we are using the model defined in first layer.
    model_env = env_utils.get_environment("cube3_layer1")
    heuristic_fn = nnet_utils_multihead.load_heuristic_fn_comp(layer, args.model_dir, device, on_gpu, model_env.get_nnet_model(nnet_type = "multihead", model_name = args["model_name"]),
                                                [env], clip_zero=True, batch_size=args.nnet_batch_size)
    solns: List[List[int]] = []
    paths: List[List[State]] = []
    times: List = []
    num_nodes_gen: List[int] = []

    for state_idx, state in enumerate(states):
        start_time = time.time()
        options = getOptions(args.option_name) if args.options else []
        num_itrs: int = 0
        astar = AStar([state], env, heuristic_fn, [args.weight], options=options)
        while not min(astar.has_found_goal()) and astar.get_num_nodes_generated(0) < args.max_nodes:
            astar.step(heuristic_fn, args.batch_size, verbose=args.verbose)
            num_itrs += 1
        #if not found solution
        if not min(astar.has_found_goal()):
            print("State: %i, Failed" % (state_idx))
            # record solution information
            solns.append(None)
            paths.append(None)
            times.append(None)
            num_nodes_gen.append(None)
        else:
            path: List[State]
            soln: List[int]
            path_cost: float
            num_nodes_gen_idx: int
            goal_node: Node = astar.get_goal_node_smallest_path_cost(0)
            path, soln, path_cost = get_path(goal_node)

            num_nodes_gen_idx: int = astar.get_num_nodes_generated(0)

            solve_time = time.time() - start_time

            # record solution information
            solns.append(soln)
            paths.append(path)
            times.append(solve_time)
            num_nodes_gen.append(num_nodes_gen_idx)

            # check soln
            assert search_utils.is_valid_soln(state, soln, env)

            # print to screen
            timing_str = ", ".join(["%s: %.2f" % (key, val) for key, val in astar.timings.items()])
            print("Times - %s, num_itrs: %i" % (timing_str, num_itrs))

            print("State: %i, SolnCost: %.2f, # Moves: %i, "
                "# Nodes Gen: %s, Time: %.2f" % (state_idx, path_cost, len(soln),
                                                format(num_nodes_gen_idx, ","),
                                                solve_time))

    return solns, paths, times, num_nodes_gen


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--states', type=str, required=True, help="File containing states to solve")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of nnet multihead model")

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
    solns_l1, paths_l1, times_l1, num_nodes_gen_l1 = bwas_python_multi_head(args, env, states, layer=0)
    times = list(filter(lambda x: x != None, times_l1))
    paths = list(filter(lambda x: x != None, paths_l1))
    num_nodes_gen = list(filter(lambda x: x != None, num_nodes_gen_l1))
    solns = list(filter(lambda x: x != None, solns_l1))
    #update remaining
    states_l2 = []
    for path in paths_l1:
        if path: states_l2.append(path[-1])
    #solve layer 2
    results["layer2_states"] = states_l2
    env = env_utils.get_environment("cube3_layer2")
    solns_l2, paths_l2, times_l2, num_nodes_gen_l2 = bwas_python_multi_head(args, env, states_l2, layer=1)
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
    env = env_utils.get_environment("cube3")
    results["layer3_states"] = states_l3
    solns_l3, paths_l3, times_l3, num_nodes_gen_l3 = bwas_python_multi_head(args, env, states_l3, layer=2)
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
