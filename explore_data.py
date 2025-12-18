import pandas as pd

# Load the dataset (adjust path if needed)
df = pd.read_csv('data/creditcard_fresh.csv')

# Basic exploration
print("First 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nClass distribution (0 = legit, 1 = fraud):")
print(df['Class'].value_counts(normalize=True))