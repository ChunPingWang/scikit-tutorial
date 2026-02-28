"""
單元 3-3：K 值對 KNN 的影響
目標：理解超參數調整的概念
"""
import matplotlib
matplotlib.use('Agg')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 測試不同的 K 值
k_range = range(1, 21)
train_scores = []
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    train_scores.append(knn.score(X_train_s, y_train))
    cv_score = cross_val_score(knn, X_train_s, y_train, cv=5).mean()
    cv_scores.append(cv_score)

best_k = list(k_range)[cv_scores.index(max(cv_scores))]
print(f"最佳 K 值: {best_k}，交叉驗證準確率: {max(cv_scores):.2%}")
print(f"K=1 時訓練準確率: {train_scores[0]:.2%}（預期 100%，過擬合）")

# 視覺化
plt.figure(figsize=(10, 5))
plt.plot(k_range, train_scores, 'o-', label='Training Accuracy')
plt.plot(k_range, cv_scores, 's-', label='Cross-Validation Accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN: Effect of K on Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.savefig('exercises/unit3_knn_k_selection.png', dpi=100)
print("圖表已儲存")
