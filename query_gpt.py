import os
import openai
import json
import time
import pandas as pd
import numpy as np

def read_prompts(filename):
    prompts = []
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                pos = line.find(",")
                pair_id = line[:pos]
                prompt = line[pos+1:].strip().capitalize().replace("please", "Please")
                prompts.append({"pair_id": pair_id,
                    "prompt": prompt})
    return prompts


def query_gpt(prompts, model_name, output_file):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with open(output_file, "w") as fout:
        for p in prompts:
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
            time.sleep(0.1)


def generate_accuracy_results(groundtruth_file, gpt_result_file, result_file):
    csv_results = pd.read_csv(groundtruth_file)
    labels = [1 if row["groundtruth"].strip() == "->" else 0
            for _, row in csv_results.iterrows()]
    preds, correct_a_cause_b, correct_b_cause_a = [], [], []
    with open(gpt_result_file) as fin:
        cnt = 0
        for line in fin:
            data = json.loads(line)
            pred = data["result"]["choices"][0]["text"].strip().lower()
            pred = 1 if pred == "yes" else 0
            preds.append(pred)
            y = labels[int(cnt / 2)]
            if cnt % 2 == 0:
                if y == pred:
                    correct_a_cause_b.append(1)
                else:
                    correct_a_cause_b.append(0)
            else:
                if y != pred:
                    correct_b_cause_a.append(1)
                else:
                    correct_b_cause_a.append(0)
            cnt += 1
    csv_results["CorrectACauseB"] = correct_a_cause_b
    csv_results["CorrectBCauseA"] = correct_b_cause_a
    accuracy = np.mean(correct_a_cause_b + correct_b_cause_a)
    print("accuracy: ", accuracy)
    csv_results.to_csv(result_file, index=False)

if __name__ == "__main__":
    prompts = read_prompts("prompts.txt")
    # model_name = "text-davinci-003"
    for model_name in ["text-davinci-002", "text-davinci-001", "davinci", "ada", "babbage", "text-babbage-001", "text-curie-001", "curie"]:
        gpt_output_file = "%s_results.jsonl" % model_name
        query_gpt(prompts, model_name, gpt_output_file)
        groundtruth_file = "results.txt"
        gpt_result_file = "%s_results.csv" % model_name
        print(model_name)
        generate_accuracy_results(groundtruth_file, gpt_output_file, gpt_result_file)

