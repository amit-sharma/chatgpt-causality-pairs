import networkx as nx
import pandas as pd
import sys
import re
import random
import csv
import json

EXPLAIN_TEMPLATE = "{0} Explain what is {1} and {2}. Then reason whether {1} can cause {2}. Provide your final answer within the tags <Answer>Yes/No</Answer>."
STEP_TEMPLATE = "{0} Let's think step-by-step whether {1} can cause {2}. Then provide your final answer on whether {1} causes {2}, within the tags <Answer>Can cause/Cannot cause</Answer>."
SINGLE_TEMPLATE = "Which of the following causal relationship is correct?\tA. Changing {0} can directly change {1}.\tB. Changing {1} can directly change {0}.\tC. Both A and B are true.\tD. None of the above. No direct relationship exists.\t\tLet's think step-by-step to make sure that we have the right answer. Then provide your final answer within the tags, <Answer>A/B/C/D</Answer>."
SINGLE_TEMPLATE2 = "Which of the following causal relationship is correct?\tA. Changing {0} has a strong effect on {1}.\tB. Changing {1} has a strong effect on {0}.\tC. Both A and B are true.\tD. None of the above. No direct relationship exists.\t\tLet's think step-by-step to make sure that we have the right answer. Then provide your final answer within the tags, <Answer>A/B/C/D</Answer>."

nodes = []
# load json data from a file
with open('variables.txt', 'r') as f:
    for l in f.readlines():
        l = l.strip()
        varname = l.split(":")[1]
        nodes.append(varname.strip())


# data is now a dict object containing the JSON data
print(nodes)

id_arr = []
prompts = []
prt_template = SINGLE_TEMPLATE #EXPLAIN_TEMPLATE
for i in range(len(nodes)-1):
    for j in range(i+1, len(nodes)):
        print(i,j)
        pair_id_str = f"pair_{i}_{j}"
        id_arr.append(pair_id_str)
        s = nodes[i]
        t = nodes[j]

        prt = prt_template.format(s, t) 
        prompts.append(prt)

prdf = pd.DataFrame({'pair_id':id_arr, 'prompt':prompts}) 
print(prdf["prompt"][0])
prdf.to_csv("prompts.csv", index=False, header=True)
prdf["prompt"] = prdf["prompt"].str.replace("\t", "\n")
prdf.to_csv("prompts_newline.csv", index=False, header=True)
