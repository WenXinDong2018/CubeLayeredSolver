import pickle
moves= ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]

results = pickle.load(open("results/cube3layer2_dynamic_difficulty_25_fixed/no_fixed_layer_no_options/results.pkl", "rb"))

count= {}
'''Visualise solution paths'''
for solution in results["solutions"]:
    solution =[moves[i] for i in solution]
    print(len(solution), solution)
     #uncomment the following code if you want to see the top moves
#     for i in range(len(solution)-6):
#         key = "".join(solution[i:i+6])
#         count[key] = count.get(key, 0)+1
# for key in count:
#     if count[key]>5:
#         print( count[key], key)

