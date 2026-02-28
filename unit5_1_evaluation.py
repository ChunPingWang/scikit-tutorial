"""
單元 5-1：交叉驗證與混淆矩陣
目標：學會用多種指標全面評估模型
"""
import matplotlib
matplotlib.use('Agg')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import numpy as np

# 乳癌資料集（二元分類：良性 vs 惡性）
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, model) in enumerate(models.items()):
    # 交叉驗證
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    print(f"\n【{name}】")
    print(f"  5-Fold CV 準確率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 訓練 + 測試
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    test_acc = model.score(X_test_s, y_test)
    print(f"  測試準確率: {test_acc:.4f}")
    print(f"  驗證: 準確率 > 93%? {'是' if test_acc > 0.93 else '否'}")
    print(f"\n{classification_report(y_test, y_pred, target_names=cancer.target_names)}")

    # 混淆矩陣
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=cancer.target_names,
        ax=axes[idx],
        cmap='Blues'
    )
    axes[idx].set_title(f'{name}\nConfusion Matrix')

plt.tight_layout()
plt.savefig('unit5_confusion_matrix.png', dpi=100)
print("混淆矩陣圖已儲存")
