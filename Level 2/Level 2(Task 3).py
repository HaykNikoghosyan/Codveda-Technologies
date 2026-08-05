
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('1) iris.csv')

X = df.drop(columns=['species'])

scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7,4))
plt.plot(range(1,11),wcss, marker='o', linestyle='--')
plt.xlabel('Number of clusters (k)', fontsize=12)
plt.ylabel('WCSS (Inertia)', fontsize=12)
plt.savefig('elbow_method.png')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['cluster'] = kmeans.fit_transform(X_scaled).argmax(axis=1)


plt.figure(figsize=(7,5))
sns.scatterplot(x=df.iloc[:,0],y=df.iloc[:,1],hue=df['cluster'],palette='viridis',s=70)
plt.title('K-means Clustering Result (K=3)', fontsize=14)
plt.savefig('kmeans_clusters.png',dpi=300)
plt.show()

