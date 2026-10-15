"""Split data/labels.csv (columns: image_path,text) into fixed train/val/test files."""
import pandas as pd

df = pd.read_csv("data/labels.csv").sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
train, val = int(n * 0.8), int(n * 0.9)
df[:train].to_csv("data/train.csv", index=False)
df[train:val].to_csv("data/val.csv", index=False)
df[val:].to_csv("data/test.csv", index=False)
print(f"train={train} val={val - train} test={n - val}")
