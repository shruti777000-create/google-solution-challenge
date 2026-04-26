import pandas as pd

df_train = pd.read_csv("data/raw/adult_train_raw.csv")

print("=== First 5 rows ===")
print(df_train.head())

print("\n=== Column types ===")
print(df_train.info())

print("\n=== Missing values ===")
print(df_train.isnull().sum())

print("\n=== Income distribution ===")
print(df_train['income'].value_counts())

print("\n=== Gender distribution ===")
print(df_train['sex'].value_counts())

print("\n=== Race distribution ===")
print(df_train['race'].value_counts())
