import pandas as pd
import numpy as np


df = pd.read_csv("large-scale.csv")

df1 = df.loc[:2119,:].copy().reset_index()
df2 =df.loc[2120:, :].copy().reset_index().loc[:2119]
df2 = df2.add_suffix("_2")
print(df1)
print(df2)

new_df = pd.concat([df1, df2], axis=1)
new_df[["premise", "claim1"]] = new_df["sentence"].str.split(",", expand=True)
new_df[["premise_copy", "claim2"]] = new_df["sentence_2"].str.split(",", expand=True)
print(new_df[:50])
new_df[["condition", "index", "premise", "claim1", "claim2"]].to_csv("cw_dataset.csv", index=False, sep=";")

df3 = df.loc[4240:, :].copy().reset_index().loc[:2119]
df4 = df.loc[6360:, :].copy().reset_index().loc[:2119]
df4 = df4.add_suffix("_2")

new_df2 = pd.concat([df3, df4], axis=1)
new_df2[["premise", "claim1"]] = new_df2["sentence"].str.split(",", expand=True)
new_df2[["premise_copy", "claim2"]] = new_df2["sentence_2"].str.split(",", expand=True)

print(new_df2[:50])
# reversing order because claim2 is correct
new_df2[["condition", "index", "premise", "claim2", "claim1"]].to_csv("rw_dataset.csv", index=False, sep=";")

