import os
import sys
import openai
import json
import time
import re
import pandas as pd
import numpy as np

#SYSTEM_PROMPT = "You are a neuropathic pain diagnosis expert."
NUM_EVALS = None# 50
DATADIR = ""
#DATADIR = ""
SYSTEM = None #"You are a helpful assistant for causal reasoning."
SKIP_PROMPTS = 0

openai.api_type = "azure"
openai.api_version = "2023-03-15-preview" 


def read_prompts(filename, skip_prompts=0):
    df = pd.read_csv(filename)
    prompts = df[["pair_id", "prompt"]].to_dict('records')
    for i in range(len(prompts)):
        prompts[i]["prompt"] = prompts[i]["prompt"].replace("\t", "\n")
        
    print(prompts[skip_prompts:(skip_prompts+2)])
    return prompts[skip_prompts:NUM_EVALS] if NUM_EVALS is not None else prompts[skip_prompts:]

def query_gpt(prompts, model_name, output_file, system=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")  # Your Azure OpenAI resource's endpoint value.

    with open(output_file, "w") as fout:
        i = 0
        while i < len(prompts):
            try:
                if model_name in ["gpt-35-turbo",'gpt-4']:
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompts[i]["prompt"]})
                    response = openai.ChatCompletion.create(
                            engine=model_name, 
                            messages=messages,
                            temperature=0.7)
                else:
                    response = openai.Completion.create(
                            model=model_name,
                            prompt=prompts[i]["prompt"],
                            temperature=1e-3,
                            max_tokens=512)
                            #top_p=1,
                            #frequency_penalty=0,
                            #presence_penalty=0,
                        #logprobs=5)
                prompts[i]["result"] = response
                print(response)
                fout.write(json.dumps(prompts[i]) + "\n")
                i += 1
            except openai.error.RateLimitError:
                print("FACED a rate limit")
                time.sleep(10)
            time.sleep(10)


def generate_accuracy_results(groundtruth_file, gpt_result_file, result_file):
    if NUM_EVALS is None:
        csv_results = pd.read_csv(groundtruth_file)#.loc[50:,:].reset_index()
    else:
        csv_results = pd.read_csv(groundtruth_file).loc[0:(NUM_EVALS-1), :]
    print(csv_results)
    labels = csv_results["groundtruth"]
    preds, correct_cause = [], []
    with open(gpt_result_file) as fin:
        cnt = 0
        for line in fin:
            data = json.loads(line)
            if gpt_result_file.find("gpt-3.5-turbo") != -1 or gpt_result_file.find("gpt-4") != -1 or gpt_result_file.find("gpt-35-turbo") != -1:
                pred = data["result"]["choices"][0]["message"]["content"].strip().lower()
            else:
                pred = data["result"]["choices"][0]["text"].strip().lower()
            print(pred)
            try:
                ans = re.search(r"<answer>.*</answer>", pred).group(0)
            except AttributeError:
                ans = "Error"
            #ans = "Yes" if pred.startswith("yes") else "No" if pred.startswith("no") else "Error"
            preds.append(ans)
            numeric_ans = "Y" if ">yes" in ans else ("N" if ">no" in ans else "Error")
            if numeric_ans == "Error":
                try:
                    ans2 = re.search(r"answer: [abc]", pred).group(0)
                    numeric_ans = ans2[-1].upper()
                except AttributeError:
                    ans2 = "Error"
                    print("STILL CANNOT FIND")
                
            numeric_ans = "Yes" if numeric_ans == "Y" else ("No" if numeric_ans=="N" else "Error")
                
            y = labels[int(cnt)].strip()
            print("Yo", ans, numeric_ans, y)
            if numeric_ans == "Error":
                correct_cause.append(0.5)
            elif y == numeric_ans:
                correct_cause.append(1)
            else:
                correct_cause.append(0)
            cnt += 1
    print(correct_cause, sum(correct_cause))
    csv_results["CorrectCause"] = correct_cause
    accuracy = np.mean(correct_cause)
    print("accuracy: ", accuracy)

    filt_csv_res = csv_results[(csv_results["CorrectCause"]!=-1)]
    accuracy = np.mean(filt_csv_res["CorrectCause"].tolist())
    print("accuracy: ", accuracy)
    csv_results.to_csv(result_file, index=False)

if __name__ == "__main__":
    datadir = DATADIR
    prompts = read_prompts(datadir + "prompts.csv", skip_prompts=SKIP_PROMPTS)
    # model_name = "text-davinci-003"
    # for model_name in ["text-davinci-002", "text-davinci-001", "davinci", "ada", "babbage", "text-babbage-001", "text-curie-001", "curie"]:
    for model_name in ["gpt-4"]:
        gpt_output_file = datadir + "%s_system_results.jsonl" % model_name
        #query_gpt(prompts, model_name, gpt_output_file, system=SYSTEM)
        groundtruth_file = datadir + "groundtruth.csv"
        gpt_result_file = datadir + "%s_system_results.csv" % model_name
        print(model_name)
        generate_accuracy_results(groundtruth_file, gpt_output_file, gpt_result_file)

