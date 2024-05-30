import networkx as nx
import pandas as pd
import sys
import re
import random
import csv
import json

SIMPLE_TEMPLATE2 = "{0}\tA: {1}\tB: {2}\t\tLet's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>A/B</Answer>."
SIMPLE_TEMPLATE3 = "{0} {1}\tA: {2}\tB: {3}\tC: {4}\t\tLet's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>A/B/C</Answer>."
SIMPLE_TEMPLATE4 = "{0} {1}\tA: {2}\tB: {3}\tC: {4}\tD: {5}\t\tLet's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>A/B/C/D</Answer>."

dataset = "cw"
dataset_filename=dataset + "_dataset.csv"

# load json data from a filea
q_ids = []
prompts = []
true_answers = []
with open(dataset_filename, "r") as f:
    for line in f.readlines()[1:]:
        line = line.strip()
        cols = line.split(";")
        q_id = cols[1].strip()
        premise = cols[2].strip()
        num_ans = 2 
        r_inds = list(range(3,3+num_ans))
        random.shuffle(r_inds)
        ans1 = cols[r_inds[0]].strip()
        ans2 = cols[r_inds[1]].strip()
        prt_template=SIMPLE_TEMPLATE2
        prt = prt_template.format(premise, ans1, ans2)
        cand_ans = ["A", "B"]
        true_ans = cand_ans[r_inds.index(3)]
        prompts.append(prt)
        q_ids.append(q_id)
        true_answers.append(true_ans)

prdf = pd.DataFrame({'qid':q_ids, 'prompt':prompts, 'groundtruth': true_answers})
print(prdf)
print(prdf["prompt"][1])
prdf.to_csv("prompts_{}.csv".format(dataset), index=False, header=True)

grdf = pd.DataFrame({'qid': q_ids, 'groundtruth': true_answers})
print(grdf)
grdf.to_csv("groundtruth_{}.csv".format(dataset), index=False, header=True)
