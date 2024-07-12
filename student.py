import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage

# Sample data: students' interests
data = {
    'Student': ['Riya', 'Bob', 'Anjali', 'David', 'Cadence', 'Frank', 'Piya', 'Hannah', 'Ivy', 'Jack'],
    'Math': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    'Science': [1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    'Art': [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    'Music': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'Sports': [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print("Student interests data:")
print(df)

# Extracting the feature matrix (only interests, excluding student names)
X = df.drop('Student', axis=1).values

# Performing hierarchical/agglomerative clustering
Z = linkage(X, method='ward')

# Calculating the optimal number of clusters to satisfy group size constraints
max_group_size = 6
min_group_size = 4
max_clusters = len(X) // min_group_size

# Creating clusters
labels = fcluster(Z, max_clusters, criterion='maxclust')

# Ensuring group size constraints
cluster_sizes = np.bincount(labels)[1:]  # Cluster sizes (ignoring cluster 0 which doesn't exist)

while np.any(cluster_sizes > max_group_size):
    max_clusters += 1
    labels = fcluster(Z, max_clusters, criterion='maxclust')
    cluster_sizes = np.bincount(labels)[1:]

while np.any(cluster_sizes < min_group_size):
    max_clusters -= 1
    labels = fcluster(Z, max_clusters, criterion='maxclust')
    cluster_sizes = np.bincount(labels)[1:]

# Adding cluster labels to the original DataFrame
df['Cluster'] = labels
print("\nStudents grouped by interests:")
print(df)

# Visualizing the clusters
sns.pairplot(df, hue='Cluster', diag_kind='kde', markers=["o", "s", "^", "D", "v", "P", "X", "*", "+", "1"])
plt.suptitle('Agglomerative Clustering of Students Based on Interests', y=1.02)
plt.show()
