import networkx as nx
import pandas as pd
import sys
import re
import random
import csv
import json

SYSTEM_PROMPT = "" #You are a helpful assistant for causal reasoning."
EXPLAIN_TEMPLATE = "{0} Explain what is {1} and {2}. Then reason whether {1} can cause {2}. Provide your final answer within the tags <Answer>Yes/No</Answer>."
STEP_TEMPLATE = "{0} Let's think step-by-step whether {1} can cause {2}. Then provide your final answer on whether {1} causes {2}, within the tags <Answer>Can cause/Cannot cause</Answer>."
SINGLE_TEMPLATE = """{0} Which cause-and-effect relationship is more likely?\tA. {1} causes {2}.\tB. {3} causes {4}.\t\tLet's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags <Answer>A/B</Answer>."""
SINGLE_TEMPLATE2 = """{0} Which cause-and-effect relationship is more likely?\tA. changing {1} causes a change in {2}.\tB. changing {3} causes a change in {4}.\t\tLet's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags <Answer>A/B</Answer>."""

def expand_node_text(s):
    if re.match("[LR] [A-Z][0-9]+", s):
        s = s + " Radiculopathy"
    if "problems" in s:
        s = s.replace("problems", "problem symptoms")
    if s.startswith("R "):
        s = s.replace("R ", "Right ", 1)
    if s.startswith("L "):
        s = s.replace("L ", "Left ", 1)
    return s

df = pd.read_csv("comp_prompts_aug.csv")

id_arr = []
prompts = []
answers = []
prefix = SYSTEM_PROMPT
prt_template = SINGLE_TEMPLATE2 #EXPLAIN_TEMPLATE
pair_id = 0
pair_ids = []

for i in range(len(df)):
    pair_id_str = df.at[i, "pair_id"]
    id_arr.append(pair_id_str)
    s1 = df.at[i, "cause1"]
    s2 = df.at[i, "cause2"]
    t1 = df.at[i, "effect1"]
    t2 = df.at[i, "effect2"]
    
    prt = prt_template.format(prefix, s1, t1, s2, t2) 
    
    prompts.append(prt)
    # writing the groundtruth results
    pair_ids.append(pair_id_str)

prdf = pd.DataFrame({'pair_id':id_arr, 'prompt':prompts}) 
print(prdf["prompt"][0])
prdf.to_csv("prompts.csv", index=False, header=True)

