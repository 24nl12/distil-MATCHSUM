# Using BERT Summarizer, choose the 5 most important sentences from each document.

from summarizer import Summarizer
import Levenshtein as lev
import nltk.data
import json
import re
import pandas as pd

def load_jsonl(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = load_jsonl("reddit_data/reddit_100.jsonl")
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
bert_model = Summarizer()

for i, entry in enumerate(data):
    entry['text'] = entry['text'].replace('*', '')
    top5 = bert_model(entry['text'], min_length=60, num_sentences=5)
    entry['idx'] = tokenizer.tokenize(top5)  

df_temp = pd.json_normalize(data)
df_temp.drop(columns='text', inplace=True)
df_temp.to_json("reddit_data/reddit_cands_100.jsonl", orient='records', lines=True)

