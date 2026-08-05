
import pandas as pd

df = pd.read_csv('1) iris.csv')

df.head()
df.info()

print('Duplicates Count: ',df.duplicated().sum())

df = df.drop_duplicates()

df['species'] = df['species'].str.strip()

df.to_csv('iris_cleaned.csv', index=False)
print('Data cleaned and saved successfully')

