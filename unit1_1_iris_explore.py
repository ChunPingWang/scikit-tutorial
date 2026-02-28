"""
單元 1-1：探索 Iris（鳶尾花）資料集
目標：了解資料集的結構，學會觀察資料
"""
from sklearn.datasets import load_iris
import pandas as pd

# 載入資料
iris = load_iris()

# 轉成 DataFrame 方便觀察
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['品種'] = [iris.target_names[t] for t in iris.target]

# 基本觀察
print("=== 資料基本資訊 ===")
print(f"共有 {df.shape[0]} 筆資料，{df.shape[1] - 1} 個特徵")
print(f"特徵名稱: {iris.feature_names}")
print(f"品種類別: {list(iris.target_names)}")
print()
print("=== 前 5 筆資料 ===")
print(df.head())
print()
print("=== 統計摘要 ===")
print(df.describe().round(2))
print()
print("=== 每個品種各幾筆？ ===")
print(df['品種'].value_counts())
