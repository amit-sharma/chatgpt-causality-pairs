import os
import openai
import json
import time
import re
import pandas as pd
import numpy as np

#SYSTEM_PROMPT = "You are a neuropathic pain diagnosis expert."
NUM_EVALS = 10
def read_prompts(filename):
    df = pd.read_csv(filename)
    prompts = df[["pair_id", "prompt"]].to_dict('records')
    print(prompts[:2])
    return prompts[:NUM_EVALS]

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
                        temperature=0.1)
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
    csv_results = pd.read_csv(groundtruth_file).loc[0:(NUM_EVALS/2-1), :]
    print(csv_results)
    labels = csv_results["groundtruth"]
    preds, correct_a_cause_b, correct_b_cause_a = [], [], []
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
            numeric_ans = 1 if "can cause" in ans else 0 if "cannot" in ans else -1
            print(ans, numeric_ans)
            y = labels[int( cnt / 2)]
            if cnt % 2 == 0:
                if numeric_ans == -1:
                    correct_a_cause_b.append(-1)
                elif y == numeric_ans:
                    correct_a_cause_b.append(1)
                else:
                    correct_a_cause_b.append(0)
            else:
                if numeric_ans == -1:
                    correct_b_cause_a.append(-1)
                elif y != numeric_ans:
                    correct_b_cause_a.append(1)
                else:
                    correct_b_cause_a.append(0)
            cnt += 1
    csv_results["CorrectACauseB"] = correct_a_cause_b
    csv_results["CorrectBCauseA"] = correct_b_cause_a
    accuracy = np.mean(correct_a_cause_b + correct_b_cause_a)
    print("accuracy: ", accuracy)
    filt_csv_res = csv_results[(csv_results["CorrectACauseB"]!=-1) & (csv_results["CorrectBCauseA"]!=-1)]
    accuracy = np.mean(filt_csv_res["CorrectACauseB"].tolist() + filt_csv_res["CorrectBCauseA"].tolist())
    print("accuracy: ", accuracy)
    csv_results.to_csv(result_file, index=False)

if __name__ == "__main__":
    prompts = read_prompts("prompts.csv")
    # model_name = "text-davinci-003"
    # for model_name in ["text-davinci-002", "text-davinci-001", "davinci", "ada", "babbage", "text-babbage-001", "text-curie-001", "curie"]:
    for model_name in ["gpt-3.5-turbo"]:# ["text-davinci-003"]: #["gpt-3.5-turbo"]:
        gpt_output_file = "%s_system_results.jsonl" % model_name
        query_gpt(prompts, model_name, gpt_output_file, system=None)
        groundtruth_file = "groundtruth.csv"
        gpt_result_file = "%s_system_results.csv" % model_name
        print(model_name)
        generate_accuracy_results(groundtruth_file, gpt_output_file, gpt_result_file)

