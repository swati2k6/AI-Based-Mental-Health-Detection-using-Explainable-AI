import pandas as pd

# Load both files
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Combine them
df = pd.concat([train, test], ignore_index=True)

# Keep only needed columns (adjust if needed)
df = df[['text', 'label']]

# Save final dataset
df.to_csv("data/dataset.csv", index=False)

print("Dataset merged successfully!")