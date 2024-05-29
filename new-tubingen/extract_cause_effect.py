import pandas as pd
import re

causes = []
effects = []
prompts = []
pair_ids = []
df = pd.read_csv("Novel-Goettingen-Benchmark.txt", sep="\t", engine="python", encoding='unicode_escape')

fdf = df[~df["var1"].isnull()]
fdf = fdf[~fdf["var2"].isnull()]

fdf = fdf.rename(columns = {'var1': 'cause', 'var2': 'effect', 'causal direction': 'direction'})
print(fdf.columns)

selcols_fdf = fdf[["pair_id", "cause", "effect"]].copy()
#print(causes)
selcols_fdf.to_csv("prompts_aug.csv", index=False)

# groundtruth file
gt_df = fdf[["pair_id", "cause", "effect", "direction"]].copy()
gt_df.to_csv("groundtruth.csv", index=False)

comp_df = selcols_fdf
comp_df["cause1"] = comp_df["cause"]
comp_df["cause2"] = comp_df["effect"]
comp_df["effect1"] = comp_df["effect"]
comp_df["effect2"] = comp_df["cause"]

print(comp_df)
comp_df.to_csv("comp_prompts_aug.csv", index=False)
