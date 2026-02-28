"""
綜合專案：鐵達尼號生還預測
目標：整合所有學到的技能，完成一個端到端的 ML 專案
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# ============================================================
# Step 1: 載入與探索資料
# ============================================================
# 使用 seaborn 內建的 Titanic 資料集
df = sns.load_dataset('titanic')
print("=== 資料概覽 ===")
print(f"形狀: {df.shape}")
print(f"\n欄位:\n{df.dtypes}")
print(f"\n缺漏值:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
survival_rate = df['survived'].mean()
print(f"\n生還率: {survival_rate:.2%}")
print(f"驗證: 生還率約 38%? {'是' if 0.35 <= survival_rate <= 0.42 else '否'}")

# ============================================================
# Step 2: 特徵工程
# ============================================================
# 選擇有用的特徵
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

df_clean = df[features + [target]].copy()

# 編碼類別特徵
df_clean['sex'] = LabelEncoder().fit_transform(df_clean['sex'])
df_clean['embarked'] = df_clean['embarked'].fillna('S')
df_clean['embarked'] = LabelEncoder().fit_transform(df_clean['embarked'])

# 建立新特徵
df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)

print("\n=== 處理後的特徵 ===")
print(df_clean.head())

# ============================================================
# Step 3: 前處理 + 切分資料
# ============================================================
feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
                'embarked', 'family_size', 'is_alone']

X = df_clean[feature_cols]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: 填補 → 縮放
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ============================================================
# Step 4: 模型訓練與比較
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

print("\n=== 模型比較 (5-Fold CV) ===")
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_processed, y_train, cv=5)
    results[name] = cv_scores
    print(f"{name:25s}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# Step 5: 最佳模型調參
# ============================================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train_processed, y_train)

test_acc = grid.score(X_test_processed, y_test)
print(f"\n=== 最佳模型 ===")
print(f"參數: {grid.best_params_}")
print(f"CV 準確率: {grid.best_score_:.4f}")
print(f"測試準確率: {test_acc:.4f}")
print(f"驗證: 測試準確率 > 78%? {'是' if test_acc > 0.78 else '否'}")

# ============================================================
# Step 6: 最終評估
# ============================================================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_processed)

print(f"\n=== 最終分類報告 ===")
print(classification_report(y_test, y_pred, target_names=['Dead', 'Survived']))

# 混淆矩陣
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Dead', 'Survived'],
    ax=axes[0], cmap='Blues'
)
axes[0].set_title('Confusion Matrix')

# 特徵重要性
importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)
axes[1].barh(range(len(sorted_idx)), importances[sorted_idx])
axes[1].set_yticks(range(len(sorted_idx)))
axes[1].set_yticklabels([feature_cols[i] for i in sorted_idx])
axes[1].set_title('Feature Importance')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('unit7_titanic_final.png', dpi=100)

print("\n=== 專案完成！===")
print("你已經完成了一個完整的機器學習專案流程：")
print("  資料探索 → 特徵工程 → 前處理 → 模型訓練 → 調參 → 評估")
