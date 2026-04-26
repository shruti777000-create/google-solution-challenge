import pandas as pd
import json
import os

# Load all summary files
df = pd.read_csv("data/processed/adult_train_clean.csv")
gender_stats = pd.read_csv("data/summary/gender_stats.csv")
race_stats = pd.read_csv("data/summary/race_stats.csv")
age_stats = pd.read_csv("data/summary/age_stats.csv")

with open("data/summary/overall_summary.json") as f:
    overall = json.load(f)

os.makedirs("data/report", exist_ok=True)

# ── Build the report ──
report = f"""
=====================================
   FAIRNESS AI PROJECT - DATA REPORT
   Person 1 (Data) - Final Summary
=====================================

DATASET OVERVIEW
----------------
Total training records : {overall['total_records']:,}
High income (>50K)     : {overall['high_income_count']:,} ({overall['overall_rate_pct']}%)
Low income (<=50K)     : {overall['total_records'] - overall['high_income_count']:,} ({round(100 - overall['overall_rate_pct'], 1)}%)

⚠ Dataset is IMBALANCED — 75% low income vs 25% high income
  Person 2: Consider using class_weight='balanced' in your model

GENDER BIAS
-----------
Female  →  11.4% earn >50K
Male    →  31.4% earn >50K
GAP     →  {overall['gender_gap_pct']}% difference ⚠ HIGH BIAS

RACE BIAS
---------
Other              →   9.1% earn >50K
Amer-Indian-Eskimo →  11.9% earn >50K
Black              →  13.0% earn >50K
Asian-Pac-Islander →  27.7% earn >50K
White              →  26.4% earn >50K
GAP                →  {overall['race_max_gap_pct']}% difference ⚠ HIGH BIAS

AGE BIAS
--------
18-25  →   2.0% earn >50K
26-35  →  19.1% earn >50K
36-45  →  35.2% earn >50K
46-55  →  40.2% earn >50K  ← PEAK earning age
56-65  →  32.2% earn >50K
65+    →  22.5% earn >50K

EDUCATION BIAS
--------------
Low education  (1-6)   →  0-7% earn >50K
Mid education  (7-12)  →  5-25% earn >50K
High education (13-16) →  42-75% earn >50K
GAP → 74.9% difference ⚠ HIGHEST BIAS

MISSING DATA (removed during cleaning)
---------------------------------------
workclass      : 1,836 rows removed
occupation     : 1,843 rows removed
native_country :   583 rows removed
Total removed  : ~2,400 rows (~7% of data)

FILES DELIVERED BY PERSON 1
-----------------------------
data/processed/adult_train_clean.csv  ← main training data
data/processed/adult_test_clean.csv   ← main test data
data/charts/gender_income_gap.png     ← for dashboard
data/charts/race_income_gap.png       ← for dashboard
data/charts/age_income_distribution.png ← for dashboard
data/summary/gender_stats.csv         ← for Person 3
data/summary/race_stats.csv           ← for Person 3
data/summary/age_stats.csv            ← for Person 3
data/summary/overall_summary.json     ← for Person 4

RECOMMENDATIONS FOR TEAM
--------------------------
Person 2 (Model)    → Use class_weight='balanced' due to imbalanced data
Person 3 (Fairness) → Focus on gender gap (20%) and race gap (18.6%)
Person 4 (Dashboard)→ Highlight gender and race charts prominently
=====================================
"""

print(report)

# Save report as text file
with open("data/report/data_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("Report saved to data/report/data_report.txt")
