from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parents[1] / 'Data'

df = pd.read_csv(DATA_DIR / '1) iris.csv')
print(df.head())

plt.figure(figsize=(7,5))
sns.barplot(x='species', y='sepal_length', data=df, hue='species',palette='Set2', legend=False)
plt.title('Average Sepal Length by Species', fontsize = 14)
plt.xlabel('Species', fontsize = 12)
plt.ylabel('Sepal Length', fontsize = 12)
plt.savefig(BASE_DIR / 'bar_plot.png', dpi=300,bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 4))
sns.lineplot(data=df['sepal_length'], color='blue', label='Sepal Length')
plt.title('Line Chart of Sepal Length', fontsize = 14)
plt.xlabel('Index', fontsize = 12)
plt.ylabel('Sepal Length', fontsize = 12)
plt.legend()
plt.savefig(BASE_DIR / 'line_chart.png', dpi=300,bbox_inches='tight')
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species',data=df,palette='deep')
plt.title('Sepal Length vs Sepal Width', fontsize = 14)
plt.xlabel('Sepal Length', fontsize = 12)
plt.ylabel('Sepal Width', fontsize = 12)
plt.savefig(BASE_DIR / 'scatter_plot.png', dpi=300,bbox_inches='tight')
plt.show()