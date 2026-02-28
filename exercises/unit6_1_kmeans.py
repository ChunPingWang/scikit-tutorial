"""
單元 6-1：K-Means 分群與手肘法
目標：學會非監督式學習的基本流程
"""
import matplotlib
matplotlib.use('Agg')
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 生成模擬資料（3 群）
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# 手肘法：測試 K=1~10
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 圖 1: 手肘法
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].axvline(x=3, color='r', linestyle='--', alpha=0.5, label='K=3')
axes[0].legend()

# 圖 2: K=3 分群結果
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km3.fit_predict(X)
for i in range(3):
    mask = labels == i
    axes[1].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Cluster {i}')
axes[1].scatter(km3.cluster_centers_[:, 0], km3.cluster_centers_[:, 1],
               c='red', marker='X', s=200, label='Centers')
axes[1].set_title('K-Means Clustering (K=3)')
axes[1].legend()

# 圖 3: 真實標籤（對照）
for i in range(3):
    mask = y_true == i
    axes[2].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'True {i}')
axes[2].set_title('True Labels (for comparison)')
axes[2].legend()

plt.tight_layout()
plt.savefig('exercises/unit6_kmeans.png', dpi=100)
print("K-Means 圖表已儲存")

# 驗證手肘法
print(f"\nInertia 值:")
for k, inertia in zip(K_range, inertias):
    print(f"  K={k}: {inertia:.1f}")
print(f"\nK=2→3 下降: {inertias[1]-inertias[2]:.1f}")
print(f"K=3→4 下降: {inertias[2]-inertias[3]:.1f}")
print(f"K=3 處有明顯轉折（下降量大幅減少）")
