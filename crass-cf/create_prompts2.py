import networkx as nx
import pandas as pd
import sys
import re
import random
import csv
import json

SIMPLE_TEMPLATE3 = "{0} {1}\tA: {2}\tB: {3}\tC: {4}\t\tLet's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>A/B/C</Answer>."
SIMPLE_TEMPLATE4 = "{0} {1}\tA: {2}\tB: {3}\tC: {4}\tD: {5}\t\tLet's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>A/B/C/D</Answer>."

# load json data from a filea
q_ids = []
prompts = []
true_answers = []
with open("CRASS_FTM_main_data_set.csv", "r") as f:
    for line in f.readlines()[1:]:
        line = line.strip()
        cols = line.split(";")
        q_id = cols[0].strip()
        premise = cols[2].strip()
        qcc = cols[3].strip()
        num_ans = 3 if cols[7].strip()=="" else 4
        r_inds = list(range(4,4+num_ans))
        random.shuffle(r_inds)
        ans1 = cols[r_inds[0]].strip()
        ans2 = cols[r_inds[1]].strip()
        ans3 = cols[r_inds[2]].strip()
        if num_ans == 3:
            prt_template=SIMPLE_TEMPLATE3
            prt = prt_template.format(premise, qcc, ans1, ans2, ans3)
        elif num_ans == 4:
            prt_template=SIMPLE_TEMPLATE4
            ans4 = cols[r_inds[3]].strip()
            prt = prt_template.format(premise, qcc, ans1, ans2, ans3, ans4)
        else:
            raise ValueError
        cand_ans = ["A", "B", "C", "D"]
        true_ans = cand_ans[r_inds.index(4)]
        prompts.append(prt)
        q_ids.append(q_id)
        true_answers.append(true_ans)

prdf = pd.DataFrame({'qid':q_ids, 'prompt':prompts, 'groundtruth': true_answers})
print(prdf)
print(prdf["prompt"][1])
prdf.to_csv("prompts.csv", index=False, header=True)

grdf = pd.DataFrame({'qid': q_ids, 'groundtruth': true_answers})
print(grdf)
grdf.to_csv("groundtruth.csv", index=False, header=True)
