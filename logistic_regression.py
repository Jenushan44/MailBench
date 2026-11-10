import pandas as pd

df = pd.read_csv("data/phishing_email.csv")

print(df.head())
print(df.shape)

# Checks for any null values in the dataset
print(df.isnull().sum())

# Checks for any duplicate rows in the dataset
print(df.duplicated().sum())

# Drop duplicate rows in the dataset
cleaned_df = df.drop_duplicates()

print(cleaned_df.duplicated().sum())