"""
單元 6-2：PCA 降維
目標：將高維資料降到 2D 視覺化
"""
import matplotlib
matplotlib.use('Agg')
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 手寫數字資料集：64 維（8x8 像素）
digits = load_digits()
print(f"原始維度: {digits.data.shape[1]}（8x8 像素）")
print(f"樣本數: {digits.data.shape[0]}")

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(digits.data)

# PCA 降到 2 維
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print(f"降維後維度: {X_2d.shape[1]}")
print(f"保留的變異比例: {pca.explained_variance_ratio_.sum():.2%}")

# 視覺化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                      c=digits.target, cmap='tab10',
                      alpha=0.5, s=10)
plt.colorbar(scatter, label='Digit')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA: 64D → 2D Visualization of Handwritten Digits')
plt.savefig('unit6_pca.png', dpi=100)
print("PCA 圖表已儲存")

# 累積變異比例
pca_full = PCA().fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_90 = np.argmax(cumvar >= 0.90) + 1
print(f"\n要保留 90% 資訊，需要 {n_90} 個主成分（原本 64 維）")
