# Script to download and save the Hausa AfriSenti-Twitter dataset splits to CSV
from datasets import load_dataset

# Load Hausa split (code: 'hau')
dataset = load_dataset("HausaNLP/AfriSenti-Twitter", "hau")

# Save each split to CSV
for split in dataset:
    df = dataset[split].to_pandas()
    df.to_csv(f"data/afrisenti_twitter_hausa_{split}.csv", index=False)

print("Saved Hausa AfriSenti-Twitter splits to data/ directory.")
