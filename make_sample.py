import pandas as pd

src = "US_Accidents_full.csv"
dst = "traffic_project/accidents.csv"

parts = []
for chunk in pd.read_csv(src, chunksize=500000, low_memory=False):
    parts.append(chunk.sample(frac=0.02, random_state=42))
df = pd.concat(parts).sample(n=100000, random_state=42)
df.to_csv(dst, index=False)
print("saved", len(df), "rows to", dst)
print("severe share (Severity 3 or 4):", round((df["Severity"] >= 3).mean(), 3))