import pandas as pd
from datasets import load_dataset

data = load_dataset("reddit_tifu", "long")
data[:100]