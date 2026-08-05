
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('1) iris.csv')

print('Summary Statistics')
print(df.describe())

numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()

print('Correlation Matrix')
print(correlation_matrix)

numeric_df.hist(figsize=(10,8), bins=20, color='skyblue', edgecolor='black')
plt.suptitle('Data Distribution (Histograms)', fontsize=16)
plt.savefig('eda_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,8))
sns.boxplot(data=numeric_df,palette='Set2')
plt.title('Boxplot of Numerical Features', fontsize=14)
plt.xticks(rotation=45)
plt.savefig('eda_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation', fontsize=14)
plt.savefig('eda_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()