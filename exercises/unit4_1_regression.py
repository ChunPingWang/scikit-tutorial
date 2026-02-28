"""
單元 4-1：線性迴歸實作
目標：預測房價，理解迴歸的輸出與評估
"""
import matplotlib
matplotlib.use('Agg')
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 載入加州房價資料
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # 單位：十萬美元

print("=== 資料概覽 ===")
print(f"樣本數: {df.shape[0]}，特徵數: {df.shape[1] - 1}")
print(f"\n特徵說明:")
print("MedInc: 收入中位數 / HouseAge: 屋齡 / AveRooms: 平均房間數")
print("AveBedrms: 平均臥室數 / Population: 人口 / AveOccup: 平均住戶數")
print("Latitude: 緯度 / Longitude: 經度")

# 前處理
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 訓練
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)

# 評估
r2 = r2_score(y_test, y_pred)
print("\n=== 模型評估 ===")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f} (十萬美元)")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} (十萬美元)")
print(f"R²:   {r2:.4f}")
print(f"\n驗證: R² 在 0.5~0.7 之間? {'是' if 0.5 <= r2 <= 0.7 else '否'}")

# 各特徵的權重
print("\n=== 特徵權重（影響力）===")
for name, coef in sorted(zip(housing.feature_names, lr.coef_),
                          key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:12s}: {coef:+.4f}")

# 驗證 MedInc 權重最大
max_feature = max(zip(housing.feature_names, lr.coef_), key=lambda x: abs(x[1]))
print(f"\n權重最大特徵: {max_feature[0]} (預期 MedInc)")

# 預測 vs 實際
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
plt.plot([0, 5], [0, 5], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Price (100k USD)')
plt.ylabel('Predicted Price (100k USD)')
plt.title('Linear Regression: Predicted vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('exercises/unit4_regression.png', dpi=100)
print("圖表已儲存")
