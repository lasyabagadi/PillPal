import pandas as pd
import os

for csv in ["data/processed/train.csv", "data/processed/val.csv", "data/processed/test.csv"]:
    if not os.path.exists(csv):
        print("Missing:", csv)
        continue
    df = pd.read_csv(csv)
    # Some CSVs use 'image_path' column; fallback to 'filepath' or 'images'
    path_col = None
    for c in ("image_path", "filepath", "images"):
        if c in df.columns:
            path_col = c; break
    if path_col is None:
        raise RuntimeError(f"No path column in {csv}")
    df['exists'] = df[path_col].apply(lambda p: os.path.exists(p))
    missing = df[~df['exists']]
    print(f"{csv}: {len(df)} rows, missing files: {len(missing)}")
    df2 = df[df['exists']].drop(columns=['exists'])
    
    df2.to_csv(csv, index=False)
    print(f"Wrote cleaned {csv} with {len(df2)} rows")