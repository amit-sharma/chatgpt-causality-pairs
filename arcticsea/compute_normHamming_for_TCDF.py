import numpy as np
import json
import re
from scipy.spatial import distance
import plot_graph
import sys

def dist_between_matrices_normHamming(A,B):
    return distance.hamming(A.flatten(),B.flatten())

#############################################

graph_Yiyi=np.array([[0,0,0,1,1,0,1,1,1,1,1,0],
                      [0,0,0,0,0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,0,0,1,0,0,0],
                      [1,0,0,0,0,1,1,1,1,0,0,1],
                      [1,0,1,0,0,1,0,0,1,1,1,0],
                      [0,0,1,0,1,0,0,0,0,1,1,0],
                      [1,0,0,0,0,0,0,0,1,0,0,0],
                      [1,0,0,0,0,0,0,0,1,0,0,0],
                      [1,1,1,1,0,0,1,1,0,0,0,0],
                      [1,1,1,0,1,1,0,0,0,0,0,0],
                      [1,1,1,0,1,1,0,0,0,0,0,0],
                      [0,0,1,1,0,1,0,0,0,0,0,0]])
print(np.sum(graph_Yiyi))
graph_h0k2=np.array([[0,0,0,0,0,0,0,0,0,0,0,0], # test 1: hidden_layer=0, kernel_size=2
            [0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,1,0,0,0,0],
                  	[0,0,0,0,0,0,1,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0]])
graph_h0k4=np.array([[0,0,0,0,0,0,0,0,0,0,0,0], # test 2: hidden_layer=0, kernel_size=4
            [0,0,0,0,0,0,0,0,0,0,1,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,1,0,0,0,0,0,0,1],
                  	[1,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,1,0,0,0,0],
                  	[1,0,0,0,0,0,1,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,1,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,1,0,0,0,0,0,0,0,0]])
print(np.sum(graph_h0k4))
graph_h0k6=np.array([[0,0,0,0,0,0,0,0,0,0,0,0], # test 3: hidden_layer=0, kernel_size=6
            [0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,1,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0]])
graph_h1k2=np.array([[0,0,0,0,0,0,0,0,0,0,0,0], # test 4: hidden_layer=1, kernel_size=2
            [0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,1,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,1,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,1,0,0,0,0,0,0,0,0,0]])
graph_h1k4=np.array([[0,0,0,0,0,0,0,0,0,0,0,0], # test 5: hidden_layer=1, kernel_size=4
            [0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0]])
graph_h1k6=np.array([[0,0,0,0,0,0,0,0,0,0,0,0], # test 6: hidden_layer=1, kernel_size=6
            [0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0],
                  	[0,0,0,0,0,0,0,0,0,0,0,0]])

graph_chatgpt = np.zeros((12,12))
model_name = "text-davinci-003"
gpt_result_file = "%s_system_results_singleprompt.jsonl" % model_name
accuracyf = open(f"{gpt_result_file}.csv", "w")
acc = 0
pred_st = 0
pred_ts = 0
true_st, true_ts = 0, 0
recall_st, recall_ts = 0,0
prec_st, prec_ts = 0,0
with open(gpt_result_file) as fin:
    cnt = 0
    for line in fin:
        data = json.loads(line)
        pair_id = data["pair_id"]
        s = int(pair_id.split("_")[1])
        t = int(pair_id.split("_")[2])
        if gpt_result_file.find("gpt-3.5-turbo") != -1 or gpt_result_file.find("gpt-4") != -1 or gpt_result_file.find("gpt-35-turbo") != -1:
            pred = data["result"]["choices"][0]["message"]["content"].strip().lower()
        else:
            pred = data["result"]["choices"][0]["text"].strip().lower()
        try:
            ans = re.search(r"<answer>.*</answer>", pred).group(0)
        except AttributeError:
            ans = "Error"
        st_ans = 1 if ">a" in ans or ">c" in ans else 0
        ts_ans = 1 if ">b" in ans or ">c" in ans else 0
        print(s, t, ans, st_ans, ts_ans)
        true_st_ans = graph_Yiyi[s,t]
        true_ts_ans = graph_Yiyi[t,s]
        acc += true_st_ans == st_ans
        acc += true_ts_ans == ts_ans
        if true_st_ans == st_ans and st_ans == 1:
            prec_st += 1
        if true_ts_ans == st_ans and ts_ans == 1:
            prec_ts += 1
        if true_st_ans == st_ans and true_st_ans == 1:
            recall_st += 1
        if true_ts_ans == st_ans and true_ts_ans == 1:
            recall_ts += 1
        pred_st += st_ans
        pred_ts += ts_ans
        true_st += true_st_ans
        true_ts += true_ts_ans
        accuracyf.writelines(f"{s},{t},{ans},{st_ans},{ts_ans},{true_st_ans},{true_ts_ans}\n")
        graph_chatgpt[s,t] = st_ans
        graph_chatgpt[t,s] = ts_ans
        #if numeric_ans == "Error":
        #    correct_cause.append(-1)
        #elif y == numeric_ans:
        #    correct_cause.append(1)
        #else:
        #    correct_cause.append(0)
        cnt += 1
accuracyf.close()
acc = acc/(cnt*2)
print(graph_chatgpt)
print(np.sum(graph_chatgpt), np.sum(graph_Yiyi), acc)
print(dist_between_matrices_normHamming(graph_Yiyi,graph_chatgpt))

print(true_st, true_ts)
print(pred_st, pred_ts)
print(prec_st, "Prec:", prec_st/pred_st, "Recall", recall_st/true_st)
print(prec_ts, "Prec:", prec_ts/pred_ts, "Recall", recall_ts/true_ts)

# Overall precision
prec = (prec_st + prec_ts)/(pred_st+pred_ts)
rec = (recall_st+recall_ts)/(true_st+true_ts)
f1 = 2*(prec*rec)/(prec+rec)
print("precision:", prec, "recall", rec, "f1", f1)
nodes = []
# load json data from a file
with open('variables.txt', 'r') as f:
    for l in f.readlines():
        l = l.strip()
        varname = l.split(":")[1]
        nodes.append(varname.strip())
plot_graph.visualize_static(graph_chatgpt,nodes,"static.png")
plot_graph.visualize_static_unweighted(graph_chatgpt,nodes,"static_unweighted.png")
