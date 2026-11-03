# MIS 480 group project - build our working subset and run quality checks
# Derrick, data prep
import numpy as np
import pandas as pd
from datasets import load_dataset

# the slice we agreed on in the meeting: 2 cities, 5 stores each, 10 products each
N_CITIES, N_STORES, N_PRODUCTS, N_DAYS = 2, 5, 10, 90

ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train")
idx = ds.select_columns(["city_id", "store_id", "product_id", "dt"]).to_pandas()

# take the two cities with the most stores
city_stores = idx.groupby("city_id")["store_id"].nunique().sort_values(ascending=False)
cities = sorted(city_stores.head(N_CITIES).index.tolist())

# in each city, the 5 stores with the most products, then 10 products from each
pairs, picks = [], []
for c in cities:
    city_rows = idx[idx.city_id == c]
    stores = city_rows.groupby("store_id")["product_id"].nunique().sort_values(ascending=False).head(N_STORES).index
    for s in sorted(stores):
        prods = sorted(city_rows[city_rows.store_id == s]["product_id"].unique())[:N_PRODUCTS]
        pairs += [(s, p) for p in prods]
        picks.append((c, s, prods))

pairs_set = set(pairs)
print(f"cities: {cities}")
print(f"series picked: {len(pairs_set)}")

# grab the full rows (with the hourly arrays) for just those store-product pairs
keep = idx.set_index(["store_id", "product_id"]).index.isin(pairs_set)
sub = ds.select(np.where(keep)[0].tolist()).to_pandas()
sub["dt"] = pd.to_datetime(sub["dt"])
sub = sub.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)

# quality checks
report = []
def note(x):
    print(x); report.append(str(x))

note(f"rows: {len(sub)}")
note(f"series: {sub.groupby(['store_id','product_id']).ngroups}")
note(f"duplicates: {sub.duplicated(['store_id','product_id','dt']).sum()}")
dps = sub.groupby(['store_id','product_id'])['dt'].nunique()
note(f"days per series min/max: {dps.min()}/{dps.max()}")
note(f"dates: {sub['dt'].min().date()} to {sub['dt'].max().date()}")
note("missing values:")
for col in sub.columns:
    if col not in ("hours_sale", "hours_stock_status"):
        note(f"  {col}: {sub[col].isna().sum()}")
note(f"sale_amount min/mean/max: {sub.sale_amount.min():.2f}/{sub.sale_amount.mean():.2f}/{sub.sale_amount.max():.2f}")
note(f"discount min/max: {sub.discount.min():.2f}/{sub.discount.max():.2f}")
note(f"rows with a discount: {(sub.discount < 1.0).sum()}")
note(f"holiday rows: {sub.holiday_flag.sum()}  promo rows: {sub.activity_flag.sum()}")
note(f"rows with a stockout: {(sub.stock_hour6_22_cnt > 0).sum()} ({(sub.stock_hour6_22_cnt>0).mean()*100:.1f}%)")

# save the subset and the documentation
sub.to_parquet("data/subset/freshretail_subset.parquet", index=False)
sub.drop(columns=["hours_sale", "hours_stock_status"]).to_csv("data/subset/freshretail_subset_daily.csv", index=False)

data_dict = [
    ("city_id", "city, encoded"),
    ("store_id", "store, encoded"),
    ("management_group_id", "store management group, encoded"),
    ("first_category_id", "top product category"),
    ("second_category_id", "mid product category"),
    ("third_category_id", "detailed product category"),
    ("product_id", "product, encoded"),
    ("dt", "date of the daily record"),
    ("sale_amount", "daily sales, normalized (not units or revenue) - target"),
    ("hours_sale", "24 hourly normalized sales values"),
    ("stock_hour6_22_cnt", "out-of-stock hours during business hours (6-22)"),
    ("hours_stock_status", "24 hourly flags, 1 = out of stock"),
    ("discount", "price ratio, 1.0 = no discount"),
    ("holiday_flag", "1 if holiday"),
    ("activity_flag", "1 if a promo is running"),
    ("precpt", "precipitation"),
    ("avg_temperature", "average temperature"),
    ("avg_humidity", "average humidity"),
    ("avg_wind_level", "average wind"),
]
pd.DataFrame(data_dict, columns=["variable", "description"]).to_csv("outputs/data_dictionary.csv", index=False)

with open("outputs/subset_selection.txt", "w") as f:
    f.write(f"cities (most stores): {cities}\n")
    f.write(f"{N_STORES} stores per city (most products), {N_PRODUCTS} products per store\n\n")
    for c, s, prods in picks:
        f.write(f"city {c} | store {s} | products: {prods}\n")

with open("outputs/quality_report.txt", "w") as f:
    f.write("\n".join(report))

print("saved subset and output files")
