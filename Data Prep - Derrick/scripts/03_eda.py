# MIS 480 group project - quick EDA on our subset
# Derrick, data prep / exploration
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/eda", exist_ok=True)
df = pd.read_csv("data/subset/freshretail_subset_daily.csv", parse_dates=["dt"])
print(f"loaded {len(df)} rows")

# 1) how daily sales are distributed
plt.figure(figsize=(7,4))
plt.hist(df["sale_amount"], bins=60, color="#4477aa", edgecolor="white")
plt.xlabel("daily sale_amount (normalized)")
plt.ylabel("store-product-days")
plt.title("Distribution of daily sales")
plt.tight_layout(); plt.savefig("outputs/eda/01_sales_distribution.png", dpi=150); plt.close()

# 2) average sales with each condition off vs on
df["discounted"] = (df["discount"] < 1.0).astype(int)
df["stockout"] = (df["stock_hour6_22_cnt"] > 0).astype(int)
conds = {"Promotion":"activity_flag", "Discount":"discounted", "Holiday":"holiday_flag", "Stockout":"stockout"}
off = [df[df[c]==0]["sale_amount"].mean() for c in conds.values()]
on  = [df[df[c]==1]["sale_amount"].mean() for c in conds.values()]
x = np.arange(len(conds)); w = 0.38
plt.figure(figsize=(7,4))
plt.bar(x-w/2, off, w, label="no", color="#bbbbbb")
plt.bar(x+w/2, on, w, label="yes", color="#4477aa")
plt.xticks(x, list(conds.keys())); plt.ylabel("mean sale_amount")
plt.title("Average daily sales by condition"); plt.legend()
plt.tight_layout(); plt.savefig("outputs/eda/02_sales_by_condition.png", dpi=150); plt.close()

# 3) average sales across the 90 days
daily = df.groupby("dt")["sale_amount"].mean()
plt.figure(figsize=(8,4))
plt.plot(daily.index, daily.values, color="#4477aa")
plt.xlabel("date"); plt.ylabel("mean sale_amount")
plt.title("Average daily sales over the 90 days")
plt.tight_layout(); plt.savefig("outputs/eda/03_sales_over_time.png", dpi=150); plt.close()

# 4) sales across temperature ranges
df["temp_bin"] = pd.cut(df["avg_temperature"], bins=6)
temp = df.groupby("temp_bin", observed=True)["sale_amount"].mean()
plt.figure(figsize=(7,4))
plt.bar([str(i) for i in temp.index], temp.values, color="#4477aa")
plt.xticks(rotation=30, ha="right"); plt.ylabel("mean sale_amount")
plt.title("Average sales by temperature range")
plt.tight_layout(); plt.savefig("outputs/eda/04_sales_by_temperature.png", dpi=150); plt.close()

# 5) sales vs how many hours the item was out of stock
bins, names = [-1,0,3,7,11,24], ["0","1-3","4-7","8-11","12+"]
df["so_bucket"] = pd.cut(df["stock_hour6_22_cnt"], bins=bins, labels=names)
so = df.groupby("so_bucket", observed=True)["sale_amount"].mean()
plt.figure(figsize=(7,4))
plt.bar(so.index.astype(str), so.values, color="#cc6677")
plt.xlabel("out-of-stock hours in the day"); plt.ylabel("mean observed sale_amount")
plt.title("Average observed sales by stockout hours")
plt.tight_layout(); plt.savefig("outputs/eda/05_stockout_effect.png", dpi=150); plt.close()

# numbers to talk to in the meeting
print("\nmean sales, condition off vs on:")
for lab, c in conds.items():
    print(f"  {lab}: no={df[df[c]==0].sale_amount.mean():.2f}  yes={df[df[c]==1].sale_amount.mean():.2f}")
print("\nmean sales by stockout hours:")
print(so.round(2).to_string())
print("\nsaved 5 charts to outputs/eda/")
