from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parents[1] / 'Data'

df = pd.read_csv(DATA_DIR / '1) iris.csv')

print(df.head())
df.info()

# Missing values: fill numeric columns with the median, text columns with the mode
print('Missing values per column:')
print(df.isnull().sum())
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

print('Duplicates Count: ', df.duplicated().sum())
df = df.drop_duplicates()

df['species'] = df['species'].str.strip()

df.to_csv(BASE_DIR / 'iris_cleaned.csv', index=False)
print('Data cleaned and saved successfully')
