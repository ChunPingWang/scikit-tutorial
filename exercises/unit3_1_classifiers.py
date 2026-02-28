"""
單元 3-1：KNN vs 決策樹 vs 隨機森林
目標：學會訓練模型、預測、評估準確率
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 準備資料
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 特徵縮放（KNN 對縮放敏感，樹模型其實不需要，但統一處理）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定義三個模型
models = {
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    '決策樹': DecisionTreeClassifier(random_state=42),
    '隨機森林': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 訓練與評估
print("=" * 50)
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n【{name}】準確率: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
