from datasets import load_dataset
from pathlib import Path
import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
# 1) Load the dataset
ds = load_dataset("beta3/GridCorpus_9M_Sudoku_Puzzles_Enriched", "gridcorpus")

# 2) Normalize to a single Dataset (handle DatasetDict vs Dataset)
if isinstance(ds, dict) or hasattr(ds, "keys"):
    # Prefer 'train' if present; otherwise take the first split
    if "train" in ds:
        ds = ds["train"]
    else:
        first_split = next(iter(ds))
        ds = ds[first_split]

# 3) Convert to pandas DataFrame
df = ds.to_pandas()  # or: df = ds.to_pandas() if you prefer

# 4) Save as Parquet
parquet_path = Path(__file__).resolve().parent.parent / 'data' / 'NineGrid' / 'ninegrid.parquet'
parquet_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(parquet_path, index=False)

print(f"Wrote Parquet to: {parquet_path}")
print('Done loading ninegrid')