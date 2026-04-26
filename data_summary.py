import pandas as pd
import json
import os

# Load clean data
df = pd.read_csv("data/processed/adult_train_clean.csv")

os.makedirs("data/summary", exist_ok=True)

# ── 1. Gender Stats ──
gender_stats = df.groupby('sex').agg(
    total=('income_binary', 'count'),
    high_income=('income_binary', 'sum'),
    rate=('income_binary', 'mean')
).round(4)
gender_stats['rate_pct'] = (gender_stats['rate'] * 100).round(1)
print("=== Gender Stats ===")
print(gender_stats)

# ── 2. Race Stats ──
race_stats = df.groupby('race').agg(
    total=('income_binary', 'count'),
    high_income=('income_binary', 'sum'),
    rate=('income_binary', 'mean')
).round(4)
race_stats['rate_pct'] = (race_stats['rate'] * 100).round(1)
print("\n=== Race Stats ===")
print(race_stats)

# ── 3. Age Stats ──
df['age_group'] = pd.cut(df['age'], 
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])

age_stats = df.groupby('age_group', observed=True).agg(
    total=('income_binary', 'count'),
    rate=('income_binary', 'mean')
).round(4)
age_stats['rate_pct'] = (age_stats['rate'] * 100).round(1)
print("\n=== Age Group Stats ===")
print(age_stats)

# ── 4. Education Stats ──
edu_stats = df.groupby('education_num').agg(
    total=('income_binary', 'count'),
    rate=('income_binary', 'mean')
).round(4)
edu_stats['rate_pct'] = (edu_stats['rate'] * 100).round(1)
print("\n=== Education Stats ===")
print(edu_stats)

# ── 5. Overall Summary ──
summary = {
    "total_records": int(len(df)),
    "high_income_count": int(df['income_binary'].sum()),
    "overall_rate_pct": round(df['income_binary'].mean() * 100, 1),
    "gender_gap_pct": round(
        df[df['sex']=='Male']['income_binary'].mean() * 100 -
        df[df['sex']=='Female']['income_binary'].mean() * 100, 1),
    "race_max_gap_pct": round(
        race_stats['rate_pct'].max() - race_stats['rate_pct'].min(), 1)
}

print("\n=== Overall Summary ===")
for key, val in summary.items():
    print(f"{key}: {val}")

# ── Save everything ──
gender_stats.to_csv("data/summary/gender_stats.csv")
race_stats.to_csv("data/summary/race_stats.csv")
age_stats.to_csv("data/summary/age_stats.csv")
edu_stats.to_csv("data/summary/edu_stats.csv")

with open("data/summary/overall_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ All summary files saved to data/summary/")
print("✅ Person 3 and Person 4 can now import from data/summary/")
