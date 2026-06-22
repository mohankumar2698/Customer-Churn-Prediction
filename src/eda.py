import pandas as pd

df = pd.read_csv("data/customer_churn.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df.corr(numeric_only=True))
