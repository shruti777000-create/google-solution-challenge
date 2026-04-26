import pandas as pd
import os

df_train = pd.read_csv("data/raw/adult_train_raw.csv")
df_test = pd.read_csv("data/raw/adult_test_raw.csv")

df_train['split'] = 'train'
df_test['split'] = 'test'

# Fix test income labels (they have a trailing dot)
df_test['income'] = df_test['income'].str.replace('.', '', regex=False)

df_all = pd.concat([df_train, df_test], ignore_index=True)

# Drop rows with missing values
df_all.dropna(inplace=True)

# Strip whitespace from text columns
str_cols = df_all.select_dtypes(include='object').columns
df_all[str_cols] = df_all[str_cols].apply(lambda col: col.str.strip())

# Create binary income column: 1 = >50K, 0 = <=50K
df_all['income_binary'] = (df_all['income'] == '>50K').astype(int)

# Split back into train and test
df_clean_train = df_all[df_all['split'] == 'train'].drop(columns='split')
df_clean_test = df_all[df_all['split'] == 'test'].drop(columns='split')

# Save to processed folder
os.makedirs("data/processed", exist_ok=True)
df_clean_train.to_csv("data/processed/adult_train_clean.csv", index=False)
df_clean_test.to_csv("data/processed/adult_test_clean.csv", index=False)

print("Clean train shape:", df_clean_train.shape)
print("Clean test shape:", df_clean_test.shape)
print("\nIncome binary distribution:")
print(df_clean_train['income_binary'].value_counts())
print("\nDone! Files saved to data/processed/")