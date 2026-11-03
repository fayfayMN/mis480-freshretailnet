# MIS 480 group project - look at the raw dataset before cutting our subset
# Derrick, data prep
import pandas as pd
from datasets import load_dataset

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train")

print(f"rows: {ds.num_rows}")
print(f"columns: {ds.column_names}")

print("\ncolumn types:")
for name, feat in ds.features.items():
    print(f"  {name}: {feat}")

print("\nsample row:")
for k, v in ds[0].items():
    show = v if not isinstance(v, list) else f"list(len={len(v)}): {v[:6]}..."
    print(f"  {k}: {show}")

print("\ndistinct counts:")
for col in ["city_id", "store_id", "product_id", "dt"]:
    print(f"  {col}: {len(ds.unique(col))}")

idx = ds.select_columns(["city_id", "store_id", "product_id", "dt"]).to_pandas()
idx["dt"] = pd.to_datetime(idx["dt"])

print(f"\ndate range: {idx['dt'].min().date()} to {idx['dt'].max().date()} ({idx['dt'].nunique()} days)")

print("\nstores per city (top 10):")
print(idx.groupby("city_id")["store_id"].nunique().sort_values(ascending=False).head(10).to_string())

print("\nseries per city (top 10):")
print(idx.groupby("city_id").apply(lambda g: g.groupby(["store_id","product_id"]).ngroups).sort_values(ascending=False).head(10).to_string())

days = idx.groupby(["store_id","product_id"])["dt"].nunique()
print(f"\ndays per series: min {days.min()}, max {days.max()}")
print(f"complete (90-day) series: {(days == 90).sum()} of {len(days)}")
