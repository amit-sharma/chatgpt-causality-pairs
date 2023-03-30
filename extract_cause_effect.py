import pandas as pd
import re

causes = []
effects = []
prompts = []
pair_ids = []
df = open("prompts.txt")
for l in df.readlines():
    prompt = l[9:].strip() 
    pair_id = l[:9]
    print(pair_id,",", prompt)
    prompts.append(prompt)
    pair_ids.append(pair_id)
    cause = re.search(r'[dD]oes changing (.+) cause', prompt).group(1)
    causes.append(cause)
    effect = re.search(r'cause a change in (.+)\?', prompt).group(1)
    effects.append(effect)
    print(cause, effect, "\n")
    
#print(causes)
df = pd.DataFrame({'pair_id': pair_ids, 'cause': causes, 'effect': effects, 'prompt': prompts})
print(df)
df.to_csv("prompts_aug.csv", index=False)

comp_df = df.groupby('pair_id', as_index=False).agg(
        {'cause': "|".join, 'effect': "|".join})

comp_df[["cause1", "cause2"]] = comp_df["cause"].str.split("|", expand=True)
comp_df[["effect1", "effect2"]] = comp_df["effect"].str.split("|", expand=True)
print(comp_df)
comp_df.to_csv("comp_prompts_aug.csv", index=False)
