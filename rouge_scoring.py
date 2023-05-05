from rouge_score import rouge_scorer
from itertools import combinations
import json
import pandas as pd

def load_jsonl(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = load_jsonl("reddit_data/reddit_cands_100.jsonl")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for entry in data:
    cands = list(combinations(entry['idx'], 2))
    cands += list(combinations(entry['idx'], 3))
    cands = [' '.join(tup) for tup in cands]
    scores = []
    for cand in cands:
        rouge_scores = scorer.score(cand, entry['summary'])
        score = sum([s.fmeasure for s in rouge_scores.values()]) / 3
        scores.append((cand, score))
    scores.sort(key=lambda x : x[1], reverse=True)
    entry['scores'] = scores

df_temp = pd.json_normalize(data)
df_temp.to_json("reddit_data/reddit_rouge_100.jsonl", orient='records', lines=True)