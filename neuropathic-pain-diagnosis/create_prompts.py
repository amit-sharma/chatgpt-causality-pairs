import networkx as nx
import pandas as pd
import sys
from source.sampling import get_var_nms,get_nam_dic
import re
import random

import json

SYSTEM_PROMPT = "You are a helpful assistant to a neuropathic pain diagnosis expert."
EXPLAIN_TEMPLATE = "{0} Explain what is {1} and {2}. Then reason whether {1} can cause {2}. Provide your final answer within the tags <Answer>Yes/No</Answer>."
STEP_TEMPLATE = "{0} Let's think step-by-step whether {1} can cause {2}. Then provide your final answer on whether {1} causes {2}, within the tags <Answer>Can cause/Cannot cause</Answer>."

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

# load json data from a file
with open('data.json', 'r') as f:
    data = json.load(f)

# data is now a dict object containing the JSON data
print(len(data["nodes"]))
print(len(data["links"]))
graph = nx.node_link_graph(data, directed=True, multigraph=False)

print(len(graph.nodes()))
print(len(graph.edges()))

print(graph)
print(graph.nodes())
print(graph.edges())

print(graph.nodes['DLS C3-C4'])
print(graph.edges(['R S1']))

 # initialize list of lists
# Create the pandas DataFrame
nms = get_var_nms()
nm_dict = get_nam_dic()

df = pd.DataFrame(nms, columns=['names'])
pd.set_option('display.max_rows', 222)


print(nms[26], list(graph.nodes())[26])
#print(df)
#print(nm_dict)

id_arr = []
prompts = []
answers = []
prefix = SYSTEM_PROMPT
prt_template = STEP_TEMPLATE #EXPLAIN_TEMPLATE
pair_id = 0
pair_ids = []
pair_answers = []
rand_edges = list(graph.edges())
random.shuffle(rand_edges)
for (s,t) in rand_edges:
    pair_id_str = "pair" + str(pair_id)
    id_arr.append(pair_id_str)
    s = expand_node_text(s)
    t = expand_node_text(t)
    
    prt = prt_template.format(prefix, s, t) 
    prompts.append(prt)
    answers.append(1)
    # reverse
    id_arr.append(pair_id_str)
    prt = prt_template.format(prefix, t, s) 
    prompts.append(prt)
    answers.append(0)

    # writing the groundtruth results
    pair_ids.append(pair_id_str)
    pair_answers.append(1)
    pair_id += 1

prdf = pd.DataFrame({'pair_id':id_arr, 'prompt':prompts, 'groundtruth': answers})
print(prdf["prompt"][0])
prdf.to_csv("prompts.csv", index=False, header=True)

grdf = pd.DataFrame({'pair_id': pair_ids, 'groundtruth': pair_answers})
print(grdf)
grdf.to_csv("groundtruth.csv", index=False, header=True)
if False:
    # initialize list of lists
    data = [['Discoligment injury', True]]

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['Query', 'Answer'])

    # print dataframe.
    print(df)

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
            new_row = {'Query':Qurey, 'Answer':Answer}
            df = df.append(new_row, ignore_index=True)

    print(df)
    # the causal queries
    df_causal = df.loc[df['Answer'] == True]
    print(df_causal.shape)
    # the non-causal relationships
    df_non_causal = df.loc[df['Answer'] == False]
    print(df_non_causal.shape)

    import numpy as np
    np.random.seed(7)
    rand_index_non_causal = np.random.randint(0,36202,50)
    df_non_causal_rand = df_non_causal.iloc[rand_index_non_causal]
    df_non_causal_rand.to_csv('result/chatGPT_testfile_false.csv', index=True, header=True)

    np.random.seed(8)
    rand_index_causal = np.random.randint(0,663,50)
    df_causal_rand = df_causal.iloc[rand_index_causal]
    df_causal_rand.to_csv('result/chatGPT_testfile_true.csv', index=True, header=True)

    import numpy as np
    np.sum(df['Answer'])
    df.to_csv('result/chatGPT_testfile.csv', index=True, header=True)
