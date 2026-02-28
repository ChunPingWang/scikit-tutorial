"""
單元 2-2：使用 Pipeline 串聯前處理
目標：學會用 Pipeline 讓程式更簡潔、避免資料洩漏
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 載入資料
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 建立 Pipeline：填補 → 縮放 → 模型
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# 一行搞定訓練
pipe.fit(X_train, y_train)

# 一行搞定預測
accuracy = pipe.score(X_test, y_test)
print(f"Pipeline 測試準確率: {accuracy:.2%}")
print(f"Pipeline 步驟: {[step[0] for step in pipe.steps]}")
