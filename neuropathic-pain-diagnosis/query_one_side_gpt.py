import os
import sys
import openai
import json
import time
import re
import pandas as pd
import numpy as np

#SYSTEM_PROMPT = "You are a neuropathic pain diagnosis expert."
NUM_EVALS = None # 10
# DATADIR = "result/"
DATADIR = ""
#SYSTEM = "You are a helpful assistant for causal reasoning."
SYSTEM = "You are an expert on neuropathic pain diagnosis."
SKIP_PROMPTS = 0

def read_prompts(filename, skip_prompts=0):
    df = pd.read_csv(filename)
    prompts = df[["pair_id", "prompt"]].to_dict('records')
    for i in range(len(prompts)):
        prompts[i]["prompt"] = prompts[i]["prompt"].replace("\t", "\n")
        
    # print(prompts[skip_prompts:(skip_prompts+2)])
    return prompts[skip_prompts:NUM_EVALS] if NUM_EVALS is not None else prompts[skip_prompts:]

def query_gpt(prompts, model_name, output_file, system=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with open(output_file, "w") as fout:
        for p in prompts:
            if model_name == "gpt-3.5-turbo":
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": p["prompt"]})
                response = openai.ChatCompletion.create(model=model_name, messages=messages)
            else:
                response = openai.Completion.create(
                        model=model_name,
                        prompt=p["prompt"],
                        temperature=1e-3,
                        max_tokens=3,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=5)
            p["result"] = response
            fout.write(json.dumps(p) + "\n")
            fout.flush()
            time.sleep(0.1)


def generate_accuracy_results(groundtruth_file, gpt_result_file, result_file):
    if NUM_EVALS is None:
        csv_results = pd.read_csv(groundtruth_file)#.loc[50:,:].reset_index()
    else:
        csv_results = pd.read_csv(groundtruth_file).loc[0:(NUM_EVALS-1), :]
    # print(csv_results)
    labels = csv_results["groundtruth"]
    preds, correct_cause = [], []
    with open(gpt_result_file) as fin:
        cnt = 0
        for line in fin:
            data = json.loads(line)
            if gpt_result_file.find("gpt-3.5-turbo") != -1:
                pred = data["result"]["choices"][0]["message"]["content"].strip().lower()
            else:
                pred = data["result"]["choices"][0]["text"].strip().lower()
            # print(pred)
            ans = pred.lower()
            preds.append(ans)
            y = labels[int(cnt)]
            num_ans = 1 if "yes" in pred else 0
            num_y = 1 if y.lower() == "yes" else 0
            if num_ans == num_y:
                correct_cause.append(1)
            else:
                correct_cause.append(0)
            cnt += 1
    # print(correct_cause, sum(correct_cause))
    csv_results["CorrectCause"] = correct_cause
    accuracy = np.mean(correct_cause)
    print("accuracy: ", accuracy)
    csv_results.to_csv(result_file, index=False)

if __name__ == "__main__":
    datadir = DATADIR
    prompts = read_prompts(datadir + "one_side_prompts.csv", skip_prompts=SKIP_PROMPTS)
    # model_name = "text-davinci-003"
    # for model_name in ["text-davinci-003"]:
    # for model_name in ["ada"]:
    # ["text-davinci-002", "text-davinci-001", "davinci", "babbage", "text-babbage-001", "text-curie-001", "curie"]:
    for model_name in ["text-ada-001"]:
        gpt_output_file = datadir + "%s_one_side_results.jsonl" % model_name
        query_gpt(prompts, model_name, gpt_output_file, system=SYSTEM)
        groundtruth_file = datadir + "one_side_groundtruth.csv"
        gpt_result_file = datadir + "%s_one_side_results.csv" % model_name
        print(model_name)
        generate_accuracy_results(groundtruth_file, gpt_output_file, gpt_result_file)
    """
    model_name = "gpt-3.5-turbo"
    gpt_output_file = datadir + "%s_one_side_results_system_none.jsonl" % model_name
    query_gpt(prompts, model_name, gpt_output_file, system=None)
    groundtruth_file = datadir + "one_side_groundtruth.csv"
    gpt_result_file = datadir + "%s_one_side_results_system_none.csv" % model_name
    print(model_name)
    generate_accuracy_results(groundtruth_file, gpt_output_file, gpt_result_file)
    
    model_name = "gpt-3.5-turbo"
    gpt_output_file = datadir + "%s_one_side_results_system_expert.jsonl" % model_name
    query_gpt(prompts, model_name, gpt_output_file, system=SYSTEM)
    groundtruth_file = datadir + "one_side_groundtruth.csv"
    gpt_result_file = datadir + "%s_one_side_results_system_expert.csv" % model_name
    print(model_name)
    generate_accuracy_results(groundtruth_file, gpt_output_file, gpt_result_file)
    """
