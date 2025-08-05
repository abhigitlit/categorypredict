from supabase import create_client, Client
import pandas as pd
url = "https://acxopokqwzzzpuyvluol.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFjeG9wb2txd3p6enB1eXZsdW9sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY0NTcyMjgsImV4cCI6MjA2MjAzMzIyOH0.QrFLVkfmJ0FHApoTuDVU9K0Ls36N3WZ5hagy396seTU"
supabase: Client = create_client(url, key)
batch_size = 1000
start = 0
all_data = []

while True:
    end = start + batch_size - 1

    response = supabase \
        .table("questions") \
        .select("text, category_id, categories(name)") \
        .range(start, end) \
        .execute()

    batch = response.data
    if not batch:
        break  # Stop when no more rows

    all_data.extend(batch)
    start += batch_size
df = pd.DataFrame(all_data)
df["category"] = df["categories"].apply(lambda x: x["name"])
df = df[["text", "category"]]
label2id = {label: idx for idx, label in enumerate(df["category"].unique())}
df["label"] = df["category"].map(label2id)

df.to_csv("train_dataset.csv", index=False)
df.to_pickle("train_dataset.pkl")

import json
with open("label2id.json", "w") as f:
    json.dump(label2id, f)

print("✅ All questions fetched and saved!")