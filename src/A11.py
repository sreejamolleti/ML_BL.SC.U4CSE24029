import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

df = pd.read_excel(file_path, sheet_name="marketing_campaign")

data = df.select_dtypes(include=np.number)
data = data.fillna(data.mean())

X = data.iloc[:, :2].values

k = 3

centroids = X[np.random.choice(len(X), k, replace=False)]

for _ in range(100):

    clusters = []

    for point in X:
        distances = []
        for centroid in centroids:
            distance = np.sqrt(np.sum((point - centroid) ** 2))
            distances.append(distance)

        clusters.append(np.argmin(distances))

    new_centroids = []

    for i in range(k):
        cluster_points = X[np.array(clusters) == i]

        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(centroids[i])

    new_centroids = np.array(new_centroids)

    if np.allclose(centroids, new_centroids):
        break

    centroids = new_centroids

print("Final Centroids:\n")
print(centroids)

plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="viridis")
plt.scatter(centroids[:, 0], centroids[:, 1], color="red", marker="X", s=200)

plt.title("K-Means Clustering")
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])

plt.show()