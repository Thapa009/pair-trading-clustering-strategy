import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt

# 1. Define tickers
tickers = ['AAPL', 'NVDA', 'INTC', 'AMD', 'TSM', 'ASML', 'QCOM', 'TXN', 'MU']

# 2. Download data with auto-adjustment (Close prices are adjusted)
df = yf.download(tickers, start="2015-01-01", end="2025-01-01", auto_adjust=True)['Close']
print(f"Historical data retrieved: {df.shape}")

# 3. Drop rows with any missing values
df.dropna(inplace=True)

# 4. Calculate daily returns
returns = df.pct_change().dropna()
print(f"Returns calculated: {returns.shape}")

# 5. Transpose for clustering on assets instead of time
returns_T = returns.T

# 6. KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(returns_T)
print(f"K-Means Clustering: {dict(zip(np.unique(kmeans_labels), np.bincount(kmeans_labels)))}")
# 7. DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan_labels = dbscan.fit_predict(returns_T)

# Remove noise (-1 label) before counting occurrences
unique_labels = np.unique(dbscan_labels)
cluster_counts = {label: np.sum(dbscan_labels == label) for label in unique_labels if label != -1}

print(f"DBSCAN Clustering (excluding noise): {cluster_counts}")


# 8. Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(returns_T)
print(f"Agglomerative Clustering: {dict(zip(np.unique(agglo_labels), np.bincount(agglo_labels)))}")

# 9. Create summary DataFrame
summary = pd.DataFrame({
    'Ticker': returns_T.index,
    'KMeans': kmeans_labels,
    'DBSCAN': dbscan_labels,
    'Agglomerative': agglo_labels
})
print("\nCluster Assignments:")
print(summary)

# Optional: save to Excel or CSV
# summary.to_csv("cluster_assignments.csv", index=False)
