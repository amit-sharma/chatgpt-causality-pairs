import os
import openai
import json
import time
import re
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
#SYSTEM_PROMPT = "You are a neuropathic pain diagnosis expert."
NUM_EVALS = 100 # 50
DATADIR = ""
#SYSTEM = "You are a helpful assistant for causal reasoning."
SYSTEM = None #"You are an expert on causal reasoning. Consider the effect of each action on the specified event. Then reason about the effect of only the final action on the specified event, assuming that the previous actions have already been taken. It is important to only use the information provided below, do not assume any other information."


def read_prompts(filename):
    df = pd.read_csv(filename)
    prompts = df[["pair_id", "prompt"]].to_dict('records')
    for i in range(len(prompts)):
        prompts[i]["prompt"] = prompts[i]["prompt"].replace("\t", "\n")
        
    print(prompts[:2])
    return prompts[:NUM_EVALS] if NUM_EVALS is not None else prompts

def query_gpt(prompts, model_name, output_file, system=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with open(output_file, "w") as fout:
        for p in prompts:
            if model_name == "gpt-3.5-turbo":
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": p["prompt"]})
                response = openai.ChatCompletion.create(
                        model=model_name, 
                        messages=messages,
                        temperature=0.7)
            else:
                response = openai.Completion.create(
                        model=model_name,
                        prompt=p["prompt"],
                        temperature=1e-3,
                        max_tokens=512)
                        #top_p=1,
                        #frequency_penalty=0,
                        #presence_penalty=0,
                        #logprobs=5)
            p["result"] = response
            print(response)
            fout.write(json.dumps(p) + "\n")
            time.sleep(0.1)


def generate_accuracy_results(groundtruth_file, gpt_result_file, result_file):
    if NUM_EVALS is None:
        csv_results = pd.read_csv(groundtruth_file)#.loc[50:,:].reset_index()
    else:
        csv_results = pd.read_csv(groundtruth_file).loc[0:(NUM_EVALS-1), :]
    print(csv_results)
    labels = csv_results["groundtruth"]
    preds, correct_cause = [], []
    tp,fp, tn, fn = 0,0,0,0
    numeric_answers = []
    with open(gpt_result_file) as fin:
        cnt = 0
        for line in fin:
            data = json.loads(line)
            if gpt_result_file.find("gpt-3.5-turbo") != -1:
                pred = data["result"]["choices"][0]["message"]["content"].strip().lower()
            else:
                pred = data["result"]["choices"][0]["text"].strip().lower()
            print(pred)
            try:
                ans = re.search(r"<answer>.*</answer>", pred).group(0)
            except AttributeError:
                ans = "Error"
            #ans = ans.strip("<answer>").rstrip("</answer>")
            preds.append(ans)
            #numeric_ans = 1 if "yes" in ans else 0 if "no" in ans else -1
            numeric_ans = "A" if ">a" in ans else ("C" if ">c" in ans else ("B" if ">b" in ans else "Error"))
            numeric_answers.append(numeric_ans)
            y = labels[int(cnt)]
            print(ans, numeric_ans)
            if numeric_ans == "Error":
                correct_cause.append(-1)
            elif y == numeric_ans:
                if y in ["A", "B"]:
                    tp += 1
                else:
                    tn += 1
                correct_cause.append(1)
            else:
                if y in ["A", "B"]:
                    fn += 1
                else:
                    fp += 1
                correct_cause.append(0)
            cnt += 1
    print(correct_cause, sum(correct_cause))
    csv_results["CorrectCause"] = correct_cause
    accuracy = np.mean(correct_cause)
    print("accuracy: ", accuracy)

    filt_csv_res = csv_results[(csv_results["CorrectCause"]!=-1)]
    accuracy = np.mean(filt_csv_res["CorrectCause"].tolist())
    print("accuracy: ", accuracy)
    prec =  tp/(fp+tp)
    rec = tp/(tp + fn)
    f1 = 2 * prec * rec /(prec+rec)
    print("Precision",prec)
    print("recall", rec)
    print("F1", f1)

    # Calculating per-class f1
    #f1_a = f1_score(labels, numeric_answers, "A")
    #f1_b = f1_score(labels, numeric_answers, "B")
    #f1_c = f1_score(labels, numeric_answers, "C")
    #macro_f1 = (f1_a+f1_b+f1_c)/3
    macro_f1 = f1_score(list(labels), numeric_answers, average="macro")
    print("Macro F1", macro_f1)
    csv_results.to_csv(result_file, index=False)

if __name__ == "__main__":
    datadir = DATADIR
    prompts = read_prompts(datadir + "prompts.csv")
    # model_name = "text-davinci-003"
    # for model_name in ["text-davinci-002", "text-davinci-001", "davinci", "ada", "babbage", "text-babbage-001", "text-curie-001", "curie"]:
    for model_name in ["gpt-3.5-turbo"]:# ["text-davinci-003"]: #["gpt-3.5-turbo"]:
        gpt_output_file = datadir + "%s_system_results.jsonl" % model_name
        query_gpt(prompts, model_name, gpt_output_file, system=SYSTEM)
        groundtruth_file = datadir + "groundtruth.csv"
        gpt_result_file = datadir + "%s_system_results.csv" % model_name
        print(model_name)
        generate_accuracy_results(groundtruth_file, gpt_output_file, gpt_result_file)

