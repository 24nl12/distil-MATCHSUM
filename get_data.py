import pandas as pd
import json
from datasets import load_dataset

data = load_dataset("reddit_tifu", "long")
df = pd.DataFrame(data)
df1 = pd.json_normalize(df.train)

df2 = df1[['documents', 'tldr']]
df2.columns = ['text', 'summary']

df_100 = df2.head(100)
df_1K = df2.head(1000)
df_100.to_csv("reddit_100.csv")
df_1K.to_csv("reddit_1K.csv")
df2.to_csv("reddit_data.csv")
df_100.to_json("reddit_100.jsonl", orient='records', lines=True)