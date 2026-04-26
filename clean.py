import pandas as pd
import os

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

df_train = pd.read_csv(url_train, names=column_names, skipinitialspace=True, na_values='?')
df_test = pd.read_csv(url_test, names=column_names, skipinitialspace=True, na_values='?', skiprows=1)

os.makedirs("data/raw", exist_ok=True)

df_train.to_csv("data/raw/adult_train_raw.csv", index=False)
df_test.to_csv("data/raw/adult_test_raw.csv", index=False)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)