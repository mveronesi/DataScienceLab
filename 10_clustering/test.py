import numpy as np
import pandas as pd
from KMeans import KMeans
from silhouette import silhouette_samples, silhouette_score

# loading gauss cluster dataset
df1 = pd.read_csv("../datasets/2D_gauss_clusters.txt")
x1 = df1["x"].values.reshape(-1, 1)
y1 = df1["y"].values.reshape(-1, 1)

# clustering
model = KMeans(n_clusters=15, max_iter=3)
labels = model.fit_predict(np.hstack((x1, y1)), plot_clusters=False)

silhouette = silhouette_samples(np.hstack((x1, y1,)), labels)
print(silhouette)
