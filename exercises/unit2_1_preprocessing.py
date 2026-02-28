"""
單元 2-1：資料前處理實作
目標：學會處理缺漏值、特徵縮放、切分資料
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 模擬一個有缺漏值的學生成績資料
np.random.seed(42)
data = {
    '數學': [85, 90, np.nan, 70, 95, 60, np.nan, 80, 75, 88,
             92, 78, 65, np.nan, 82, 91, 73, 86, 69, 94],
    '英文': [78, 85, 92, np.nan, 88, 72, 80, np.nan, 70, 82,
             90, 75, 68, 84, np.nan, 87, 76, 83, 71, 89],
    '讀書時數': [5, 7, 8, 3, 9, 2, 6, 4, 3, 7,
                8, 5, 2, 6, 4, 8, 3, 7, 2, 9],
    '及格': ['是','是','是','否','是','否','是','是','否','是',
            '是','是','否','是','是','是','否','是','否','是']
}
df = pd.DataFrame(data)

print("=== 原始資料 ===")
print(df.head(10))
print(f"\n缺漏值統計:\n{df.isnull().sum()}")

# Step 1: 處理缺漏值 — 用中位數填補
imputer = SimpleImputer(strategy='median')
df[['數學', '英文']] = imputer.fit_transform(df[['數學', '英文']])
print(f"\n填補後缺漏值: {df.isnull().sum().sum()}")

# Step 2: 類別編碼
le = LabelEncoder()
df['及格_encoded'] = le.fit_transform(df['及格'])
print(f"\n類別對應: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Step 3: 特徵縮放
features = df[['數學', '英文', '讀書時數']].values
target = df['及格_encoded'].values

scaler_std = StandardScaler()
features_std = scaler_std.fit_transform(features)

scaler_mm = MinMaxScaler()
features_mm = scaler_mm.fit_transform(features)

print("\n=== 標準化後（前 3 筆）===")
print(pd.DataFrame(features_std[:3], columns=['數學', '英文', '讀書時數']).round(3))

print("\n=== 正規化後（前 3 筆）===")
print(pd.DataFrame(features_mm[:3], columns=['數學', '英文', '讀書時數']).round(3))

# 驗證 StandardScaler
print(f"\n=== 驗證 StandardScaler ===")
print(f"各特徵平均值: {features_std.mean(axis=0).round(6)}")
print(f"各特徵標準差: {features_std.std(axis=0).round(6)}")

# 驗證 MinMaxScaler
print(f"\n=== 驗證 MinMaxScaler ===")
print(f"各特徵最小值: {features_mm.min(axis=0)}")
print(f"各特徵最大值: {features_mm.max(axis=0)}")

# Step 4: 切分訓練 / 測試集
X_train, X_test, y_train, y_test = train_test_split(
    features_std, target, test_size=0.2, random_state=42
)
print(f"\n訓練集: {X_train.shape[0]} 筆 / 測試集: {X_test.shape[0]} 筆")
