import pandas as pd
import matplotlib.pyplot as plt
import os

# Load clean data
df = pd.read_csv("data/processed/adult_train_clean.csv")

# Create output folder for charts
os.makedirs("data/charts", exist_ok=True)

# ── Chart 1: Income rate by Gender ──
gender_income = df.groupby('sex')['income_binary'].mean() * 100

plt.figure(figsize=(8, 5))
bars = plt.bar(gender_income.index, gender_income.values, color=['steelblue', 'salmon'])
plt.title('Income >50K Rate by Gender', fontsize=14)
plt.ylabel('Percentage (%)')
plt.ylim(0, 50)
for bar, val in zip(bars, gender_income.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig("data/charts/gender_income_gap.png")
plt.show()
print("✅ Chart 1 saved: gender_income_gap.png")

# ── Chart 2: Income rate by Race ──
race_income = df.groupby('race')['income_binary'].mean() * 100

plt.figure(figsize=(10, 5))
bars = plt.bar(race_income.index, race_income.values, color='steelblue')
plt.title('Income >50K Rate by Race', fontsize=14)
plt.ylabel('Percentage (%)')
plt.ylim(0, 50)
plt.xticks(rotation=15)
for bar, val in zip(bars, race_income.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig("data/charts/race_income_gap.png")
plt.show()
print("✅ Chart 2 saved: race_income_gap.png")

# ── Chart 3: Age distribution by Income ──
plt.figure(figsize=(10, 5))
df[df['income_binary'] == 0]['age'].plot(kind='hist', alpha=0.6, bins=30, label='<=50K', color='steelblue')
df[df['income_binary'] == 1]['age'].plot(kind='hist', alpha=0.6, bins=30, label='>50K', color='salmon')
plt.title('Age Distribution by Income Group', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig("data/charts/age_income_distribution.png")
plt.show()
print("✅ Chart 3 saved: age_income_distribution.png")

# ── Statistical Summary for Team ──
print("\n=== BIAS SUMMARY FOR TEAM SYNC ===")
print("\nIncome >50K rate by Gender:")
print(gender_income.round(1))
print("\nIncome >50K rate by Race:")
print(race_income.round(1))
print("\nOverall >50K rate:", round(df['income_binary'].mean() * 100, 1), "%")
