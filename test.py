import json

with open('json_results/hre_prob.json') as f:
    dict = json.load(f)
    keys = dict.keys()
    print(keys)