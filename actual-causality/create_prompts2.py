import networkx as nx
import pandas as pd
import sys
import re
import random
import csv
import json

PR = ["minimal change", "multiple sufficient causes"]
TEMPLATE = "You are an expert in counterfactual reasoning. Given an event, use the principle of {} to answer the following question.\t\t{}\t\tIs {} a {} cause for {}?\t\tAfter your reasoning, provide the final answer within the tags <Answer>Yes/No</Answer>."
TEMPLATE2 = "You are an expert in counterfactual reasoning. A sufficient cause can independently cause a given event even as other variables change their values. Based on this definition, answer the following question.\t\t{}\t\tIs {} a {} cause for {}?\t\tAfter your reasoning, provide the final answer within the tags <Answer>Yes/No</Answer>."
TEMPLATE3 = "You are an expert in counterfactual reasoning. A sufficient cause can independently cause a given event even as other variables change their values. There can be multiple sufficient causes; a sufficient cause need not be necessary. Based on this definition, answer the following question.\t\t{}\t\tIs {} a {} cause for {}?\t\tAfter your reasoning, provide the final answer within the tags <Answer>Yes/No</Answer>."

FILE = "lab_data.csv"
df = pd.read_csv(FILE, sep=",")
print(df)
id_arr = []
prompts = []
prt_template = TEMPLATE #EXPLAIN_TEMPLATE
prt_template_suf = TEMPLATE #EXPLAIN_TEMPLATE
answers = []
anstype = []
pair_id_types = []
for i in range(len(df)):
    pair_id_str = str(i)
    pair_id_type = df.at[i, "type"]
    id_arr.append(pair_id_str)
    pair_id_types.append(pair_id_type)
    prt = prt_template.format(PR[0], df.at[i,"input"], df.at[i,"action"], "necessary", df.at[i,"event"]) 
    prompts.append(prt)
    answers.append(df.at[i,"necessary"])
    anstype.append("nec")

for i in range(len(df)):
    # now for sufficiency
    pair_id_str = str(i)
    pair_id_type = df.at[i, "type"]
    pair_id_types.append(pair_id_type)
    id_arr.append(pair_id_str)
    prt = prt_template_suf.format(PR[1], df.at[i,"input"], df.at[i,"action"], "sufficient", df.at[i,"event"]) 
    prompts.append(prt)
    answers.append(df.at[i,"sufficient"])
    anstype.append("suff")

prdf = pd.DataFrame({'pair_id':id_arr, 'prompt':prompts}) 
print(prdf["prompt"][0])
prdf.to_csv("prompts_{}.csv".format(FILE.split("_")[0]), index=False, header=True)


grdf = pd.DataFrame({'pair_id': id_arr, 'pair_id_type': pair_id_types, 'ans_type': anstype, 'groundtruth': answers})
print(grdf)
grdf.to_csv("groundtruth_{}.csv".format(FILE.split("_")[0]), index=False, header=True)
