import os
import openai
import json
import time

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

if __name__ == "__main__":
    prompts = read_prompts("prompts.txt")
    model_name = "text-ada-001"
    output_file = "ada_results.jsonl"
    query_gpt(prompts, model_name, output_file)

