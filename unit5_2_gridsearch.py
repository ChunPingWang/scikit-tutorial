"""
單元 5-2：GridSearchCV 自動尋找最佳參數
目標：學會系統性地調參
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 定義要搜尋的參數組合
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

print(f"共要測試 {3*4*3} = 36 種組合")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_s, y_train)

print(f"\n最佳參數: {grid_search.best_params_}")
print(f"最佳 CV 準確率: {grid_search.best_score_:.4f}")
print(f"測試集準確率: {grid_search.score(X_test_s, y_test):.4f}")

# 顯示 Top 5 結果
results = pd.DataFrame(grid_search.cv_results_)
top5 = results.nsmallest(5, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
print(f"\n=== Top 5 參數組合 ===")
print(top5.to_string(index=False))
