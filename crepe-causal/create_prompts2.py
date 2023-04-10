import networkx as nx
import pandas as pd
import sys
import re
import random
import csv
import json

BIG_FULL_TEMPLATE = "You are an expert on counterfactual reasoning. Consider the effect of each action on the specified event. Then reason about the effect of only the final action on the specified event, assuming that the previous actions have already been taken.\t\tRULES:\t1. If any previous event is sufficient for the specified event, then the final action cannot affect the specified event.\t2. If you are unsure or see both possibilities, answer that the final action has no effect ('C. Equally likely'.)\t3. Use only the information provided below. Do not assume anything extra.\t\tQUESTION:\t{0}\tA. Less likely\tB. More likely\tC. Equally likely\t\tINSTRUCTIONS:\tLet's think about the effect of each action step-by-step using the rules above. Then provide your final answer only on the effect of the last action, within the tags, <Answer>A/B/C</Answer>."
FULL_TEMPLATE = "You are an expert on causal reasoning. Consider the effect of each action on the specified event. Then reason about the effect of only the final action on the specified event, assuming that the previous actions have already been taken. It is important to only use the information provided below, do not assume any other information.\t\t{0}\tA. Less likely\tB. More likely\tC. Equally likely\t\tLet's think about the effect of each action step-by-step. Then provide your final answer within the tags, <Answer>A/B/C</Answer>."
SHORT_TEMPLATE = "{0}\tA. Less likely\tB. More likely\tC. Equally likely\t\tLet's think about the effect of each action step-by-step. Then provide your final answer within the tags, <Answer>A/B/C</Answer>."

# load json data from a file
inputs = []
labels = []
with open('crepe-samples.jsonl', 'r') as f:
    json_list = list(f)
    for json_str in json_list:
        inp = json.loads(json_str)
        inputs.append(inp["input"][1]["content"])
        labels.append(inp["ideal"])
# data is now a dict object containing the JSON data

id_arr = []
prompts = []
answers = []
prt_template = BIG_FULL_TEMPLATE #EXPLAIN_TEMPLATE

for i in range(len(inputs)):
    id_str = str(i)
    id_arr.append(id_str)
    prt = prt_template.format(inputs[i].rsplit("?",1)[0] + "?") 
    if labels[i] not in ["less likely", "more likely", "equally likely"]:
        raise ValueError
    ans = "A" if labels[i] == "less likely" else ("B" if labels[i] == "more likely" else ("C" if labels[i]=="equally likely" else None))
    answers.append(ans)
    prompts.append(prt)
    if i <=10:
        print(prt)
        print(labels[i], ans)
        

prdf = pd.DataFrame({'pair_id':id_arr, 'prompt':prompts, 'groundtruth': answers})
print(prdf["prompt"][0])
prdf.to_csv("prompts.csv", index=False, header=True)

grdf = pd.DataFrame({'pair_id': id_arr, 'groundtruth': answers})
print(grdf)
grdf.to_csv("groundtruth.csv", index=False, header=True)
