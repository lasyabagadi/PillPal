import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

data_dir = "data/processed"
label_col = "label"  # change if your label column name differs

# Load train.csv
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

# Fit label encoder on *train* labels only
le = LabelEncoder()
train_df["label_id"] = le.fit_transform(train_df[label_col].astype(str))

# Save updated train.csv
train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
print(f"Wrote train.csv with label_id column, classes = {len(le.classes_)}")

# Apply to val and test
for split in ["val", "test"]:
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    # Filter out rows with unseen labels (optional: could also throw error)
    df = df[df[label_col].isin(le.classes_)]
    df["label_id"] = le.transform(df[label_col].astype(str))
    df.to_csv(os.path.join(data_dir, f"{split}.csv"), index=False)
    print(f"Wrote {split}.csv with label_id column")