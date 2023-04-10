import networkx as nx
import numpy as np
import pandas as pd
import sys
import warnings
from source.sampling import get_var_nms,get_nam_dic
import re
import random
import csv
import json
import pickle 

warnings.filterwarnings('ignore')

load_graph_from_json = False
SYSTEM_PROMPT = "You are a helpful assistant to a neuropathic pain diagnosis expert. "
EXPLAIN_TEMPLATE = "{0}Explain what is {1} and {2}. Then reason whether {1} can cause {2}. Provide your final answer within the tags <Answer>Yes/No</Answer>."
STEP_TEMPLATE = "{0}Let's think step-by-step whether {1} can cause {2}. Then provide your final answer on whether {1} causes {2}, within the tags <Answer>Can cause/Cannot cause</Answer>."
SINGLE_TEMPLATE1 = """{0}Which cause-and-effect relationship is more likely? Consider only direct causal mechanism and ignore any effect due to common causes.\tA. {1} causes {2}.\tB. {2} causes {1}\tC. No causal relationship exists.\t\tLet's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags <Answer>A/B/C</Answer>."""
SINGLE_TEMPLATE2 = "Does there exist a cause-and-effect relationship? Consider only direct causal mechanism and ignore any compensatory or referred effects, or effects due to common causes.\tA. {1} causes {2}.\tB. {2} causes {1}\tC. No causal relationship exists.\t\tLet's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags <Answer>A/B/C</Answer>." 
prefix = ""
prt_template = SINGLE_TEMPLATE1 #EXPLAIN_TEMPLATE

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

def load_graph_true():
    name_dic = get_nam_dic()
    cause = []
    effect = []
    edges = []
    dag = np.zeros((222, 222), dtype=np.int)

    with open('models/bnm.pickle', 'rb') as f:
        model = pickle.load(f)

    for c, e in model.edges():
        edges.append((c,e))

    return edges


import source.CauAcc as acc
dag_GT = acc.load_graph_true_graph()

graph_edges = load_graph_true() 
graph = nx.DiGraph(graph_edges)

print(graph.out_edges("L L4 Radikulopati"))
#sys.exit(0)
# initialize list of lists
# Create the pandas DataFrame
nms = get_var_nms()
nm_dict = get_nam_dic()
print(nms[:30])
print(nms[30:])
df = pd.DataFrame(nms, columns=['names'])
pd.set_option('display.max_rows', 222)
# initialize list of lists
data = [["a", "b", 'Discoligment injury', True]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['cause', 'effect' ,'Query', 'Answer'])
print(nm_dict[nms[0]],  "Yo", nms[0])

for A in nms[30:]:
    for B in nms[30:]:
        if 'DLS' in A:
            A = 'discoligamentous injury' + A[3:]
        if 'DLS' in B:
            B = 'discoligamentous injury' + B[3:]
        Qurey = A + ' causes ' + B +'.'
        if A[:2] == 'L ' or A[:2] == 'R ' or B[:2] == 'L ' or B[:2] == 'R ':
            Qurey += '\"R\" and \"L\" refer to the right and left sides of the body, respectively).'
        Qurey += 'Answer with true or false.' 
        if dag_GT[nm_dict[A]][nm_dict[B]] == 0:
            Answer = False
        else:
            Answer = True
        if "L L4 Radikulopati" in A:
            print(A, nm_dict[A], B, nm_dict[B], Answer)
        new_row = {'cause': A, 'effect': B, 'Query':Qurey, 'Answer':Answer}
        df = df.append(new_row, ignore_index=True)
# the causal queries
df_causal = df.loc[df['Answer'] == True]
print(df_causal.shape)
# the non-causal relationships
df_non_causal = df.loc[df['Answer'] == False]
print(df_non_causal.shape)
# outputting file 1
np.random.seed(7)
rand_index_non_causal = np.random.randint(0,36202,50)
df_non_causal_rand = df_non_causal.iloc[rand_index_non_causal]
df_non_causal_rand.to_csv('result/chatGPT_testfile_false.csv', index=True, header=True)

np.random.seed(8)
rand_index_causal = np.random.randint(0,663,50)
df_causal_rand = df_causal.iloc[rand_index_causal]
df_causal_rand.to_csv('result/chatGPT_testfile_true.csv', index=True, header=True)

np.sum(df['Answer'])
df.to_csv('result/chatGPT_testfile.csv', index=True, header=True)
print(df)

merged = pd.concat([df_causal_rand, df_non_causal_rand], axis=0)
merged = merged.reset_index()
print(merged)
prompts = []
pair_ids = []
pair_answers = []
for i in range(len(merged)):
    s = merged.at[i, "cause"]
    t = merged.at[i, "effect"]
    pair_id_str = "pair" + str(i)
    s = expand_node_text(s)
    t = expand_node_text(t)
    
    prt = prt_template.format(prefix, s, t) 
    ans = "A" if merged.at[i, "Answer"] else "C"
    prompts.append(prt)
    pair_answers.append(ans)
    # writing the groundtruth results
    pair_ids.append(pair_id_str)

merged["prompt"] = prompts
merged["pair_id"] = pair_ids
print(merged)
merged.to_csv("result/prompts.csv", index=False, header=True)

grdf = pd.DataFrame({'pair_id': pair_ids, 'groundtruth': pair_answers})
print(grdf)
grdf.to_csv("result/groundtruth.csv", index=False, header=True)
