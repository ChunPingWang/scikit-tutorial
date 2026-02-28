# Scikit-Learn 學習計劃：從零開始的機器學習之旅

> **適用對象**：高中程度初學者（具備基礎 Python 語法即可）
> **學習時長**：約 8～10 週，每週 6～8 小時
> **驗證方式**：每個單元皆附 Claude Code 可執行的實作練習

---

## 前置準備

### 環境建置

```bash
# 建立虛擬環境
python -m venv sklearn-lab
source sklearn-lab/bin/activate  # Windows: sklearn-lab\Scripts\activate

# 安裝必要套件
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```

### 驗證安裝

```python
import sklearn
print(f"scikit-learn 版本: {sklearn.__version__}")
```

### 先備知識自我檢查

在開始之前，請確認你能回答以下問題：

- 什麼是變數、迴圈、函式？
- 能否讀懂簡單的 Python list 與 dict 操作？
- 知道什麼是「平均數」與「比例」嗎？

如果以上有不確定的地方，建議先花一週補齊 Python 基礎。

---

## 第一單元：什麼是機器學習？（第 1 週）

### 觀念

機器學習的核心概念可以用一個比喻來理解：你在準備考試時，會從歷屆考古題中「學會規律」，然後在真正考試時「預測答案」。機器學習做的事情完全一樣，只是學習者從「人」變成了「電腦程式」。

**三大類型**：

- **監督式學習（Supervised Learning）**：有標準答案的學習。給電腦看很多「題目＋答案」的配對，讓它學會規律。例如：看房屋特徵預測房價。
- **非監督式學習（Unsupervised Learning）**：沒有標準答案，讓電腦自己找規律。例如：把顧客按消費習慣自動分群。
- **強化學習（Reinforcement Learning）**：透過嘗試與回饋來學習，類似打電動越打越好。scikit-learn 不涵蓋此類型。

**重要術語**：

| 術語 | 說明 | 生活比喻 |
|------|------|----------|
| 特徵（Feature） | 用來做預測的輸入資料 | 考題的題目 |
| 標籤（Label / Target） | 要預測的答案 | 考題的正確答案 |
| 訓練集（Training Set） | 拿來學習的資料 | 考古題 |
| 測試集（Test Set） | 拿來驗證的資料 | 正式考試 |
| 模型（Model） | 從資料學到的規律 | 你腦中的解題策略 |

### 實作 1-1：第一次接觸 scikit-learn 的內建資料集

```python
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
```

### 實作 1-2：資料視覺化

```python
"""
單元 1-2：用圖表觀察資料
目標：學會用視覺化發現資料中的規律
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[t] for t in iris.target]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 散佈圖：花瓣長度 vs 花瓣寬度
for species in iris.target_names:
    subset = df[df['species'] == species]
    axes[0].scatter(subset['petal length (cm)'], subset['petal width (cm)'], label=species, alpha=0.7)
axes[0].set_xlabel('Petal Length (cm)')
axes[0].set_ylabel('Petal Width (cm)')
axes[0].set_title('Petal Length vs Width by Species')
axes[0].legend()

# 箱型圖：各品種的花萼長度分佈
df.boxplot(column='sepal length (cm)', by='species', ax=axes[1])
axes[1].set_title('Sepal Length Distribution by Species')
axes[1].set_ylabel('Sepal Length (cm)')

plt.tight_layout()
plt.savefig('unit1_visualization.png', dpi=100)
plt.show()
print("圖表已儲存為 unit1_visualization.png")
```

### Claude Code 驗證指引

```
請用 Claude Code 執行以上兩段程式，確認：
1. Iris 資料集有 150 筆資料、4 個特徵、3 個品種
2. 每個品種各 50 筆（平衡資料集）
3. 視覺化圖表能正確產出
```

---

## 第二單元：資料前處理（第 2 週）

### 觀念

真實世界的資料通常很「髒」：有缺漏值、數值範圍差異很大、或者有些是文字而非數字。在餵給模型之前，必須先做好「資料清潔」。

**常見前處理步驟**：

1. **處理缺漏值**：補值（平均數、中位數、眾數）或刪除
2. **特徵縮放（Feature Scaling）**：讓所有特徵的數值範圍相近
   - **標準化（Standardization）**：轉成平均 0、標準差 1（`StandardScaler`）
   - **正規化（Normalization）**：轉成 0～1 範圍（`MinMaxScaler`）
3. **類別編碼**：把文字類別轉成數字（`LabelEncoder`、`OneHotEncoder`）
4. **切分訓練 / 測試集**：通常 80% 訓練、20% 測試

**為什麼要做特徵縮放？** 假設「年齡」範圍是 0～100，「年收入」是 0～10,000,000，如果不縮放，模型會以為「年收入」比「年齡」重要很多倍，但其實只是單位不同。

### 實作 2-1：完整的前處理流程

```python
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

# Step 4: 切分訓練 / 測試集
X_train, X_test, y_train, y_test = train_test_split(
    features_std, target, test_size=0.2, random_state=42
)
print(f"\n訓練集: {X_train.shape[0]} 筆 / 測試集: {X_test.shape[0]} 筆")
```

### 實作 2-2：Pipeline — 串聯前處理步驟

```python
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
```

### Claude Code 驗證指引

```
請用 Claude Code 驗證：
1. 缺漏值填補後所有 NaN 都消失
2. StandardScaler 後各特徵平均接近 0、標準差接近 1
3. MinMaxScaler 後所有值介於 0~1
4. Pipeline 能正確執行並輸出準確率
```

---

## 第三單元：分類演算法（第 3～4 週）

### 觀念

分類（Classification）就是預測「類別」，像是：這封信是不是垃圾郵件？這張照片是貓還是狗？

**本單元介紹三種經典分類器**：

**K-最近鄰（KNN）**：找到最近的 K 個鄰居，看他們大多數是什麼類別，就預測為那個類別。像是「物以類聚」——你的朋友們大多是理科生，那你也很可能是理科生。

**決策樹（Decision Tree）**：用一連串的「是非題」把資料分開。像是：「身高 > 170 嗎？」→「體重 > 60 嗎？」→ 結論。直覺、好解釋，但容易過擬合。

**隨機森林（Random Forest）**：建很多棵不同的決策樹，讓它們投票決定結果。「三個臭皮匠勝過一個諸葛亮」的概念。

### 實作 3-1：三種分類器的比較

```python
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
```

### 實作 3-2：決策樹視覺化

```python
"""
單元 3-2：決策樹視覺化
目標：理解決策樹如何做出判斷
"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
# 限制深度，讓樹不會太複雜
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(iris.data, iris.target)

# 文字版的樹結構
print("=== 決策樹規則（文字版）===")
print(export_text(clf, feature_names=iris.feature_names))

# 圖形版
plt.figure(figsize=(16, 8))
plot_tree(clf,
          feature_names=iris.feature_names,
          class_names=list(iris.target_names),
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree for Iris Classification (max_depth=3)')
plt.tight_layout()
plt.savefig('unit3_decision_tree.png', dpi=120)
plt.show()
print("決策樹圖已儲存為 unit3_decision_tree.png")
```

### 實作 3-3：KNN 的 K 值選擇

```python
"""
單元 3-3：K 值對 KNN 的影響
目標：理解超參數調整的概念
"""
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
plt.savefig('unit3_knn_k_selection.png', dpi=100)
plt.show()
print("圖表已儲存")
```

### Claude Code 驗證指引

```
請用 Claude Code 驗證：
1. 三個分類器都能在 Iris 資料集上達到 90% 以上準確率
2. 決策樹能正確視覺化
3. KNN 的 K 值曲線顯示：K=1 時訓練準確率 100%（過擬合），
   隨著 K 增大，訓練/驗證準確率逐漸收斂
```

---

## 第四單元：迴歸演算法（第 5 週）

### 觀念

迴歸（Regression）是預測「連續數值」，像是：根據房屋面積預測房價、根據氣溫預測冰品銷量。

**線性迴歸（Linear Regression）**：找一條「最佳擬合線」，讓這條線與所有數據點的距離總和最小。公式：`y = w₁x₁ + w₂x₂ + ... + b`，其中 `w` 是權重，`b` 是偏差。

**評估指標**：

| 指標 | 意義 | 直覺理解 |
|------|------|----------|
| MAE | 平均絕對誤差 | 平均每次預測差多少 |
| MSE | 均方誤差 | 對大誤差的懲罰更重 |
| RMSE | 均方根誤差 | 與原始單位相同，更好解讀 |
| R² | 決定係數 | 模型能解釋多少比例的變異（越接近 1 越好） |

### 實作 4-1：線性迴歸預測房價

```python
"""
單元 4-1：線性迴歸實作
目標：預測房價，理解迴歸的輸出與評估
"""
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
print("\n=== 模型評估 ===")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f} (十萬美元)")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} (十萬美元)")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

# 各特徵的權重
print("\n=== 特徵權重（影響力）===")
for name, coef in sorted(zip(housing.feature_names, lr.coef_),
                          key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:12s}: {coef:+.4f}")

# 預測 vs 實際
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
plt.plot([0, 5], [0, 5], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Price (100k USD)')
plt.ylabel('Predicted Price (100k USD)')
plt.title('Linear Regression: Predicted vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('unit4_regression.png', dpi=100)
plt.show()
```

### 實作 4-2：多項式迴歸與過擬合

```python
"""
單元 4-2：過擬合 vs 欠擬合的視覺化
目標：理解模型複雜度與泛化能力的關係
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 生成模擬資料：y = sin(x) + 雜訊
np.random.seed(42)
X = np.sort(np.random.uniform(0, 6, 30)).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 30)
X_plot = np.linspace(0, 6, 200).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
degrees = [1, 4, 15]
titles = ['Underfitting (degree=1)', 'Good Fit (degree=4)', 'Overfitting (degree=15)']

for ax, degree, title in zip(axes, degrees, titles):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X_plot)
    train_mse = mean_squared_error(y, model.predict(X))

    ax.scatter(X, y, color='steelblue', s=30, label='Data')
    ax.plot(X_plot, y_pred, color='tomato', linewidth=2, label=f'Model')
    ax.plot(X_plot, np.sin(X_plot), 'g--', alpha=0.5, label='True: sin(x)')
    ax.set_title(f'{title}\nTrain MSE: {train_mse:.4f}')
    ax.legend(fontsize=8)
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('unit4_overfitting.png', dpi=100)
plt.show()
print("過擬合示意圖已儲存")
```

### Claude Code 驗證指引

```
請用 Claude Code 驗證：
1. 線性迴歸 R² 應在 0.5~0.7 之間（加州房價的合理範圍）
2. MedInc（收入中位數）的權重絕對值應最大
3. 多項式迴歸圖中，degree=1 欠擬合、degree=15 過擬合明顯
```

---

## 第五單元：模型評估與選擇（第 6 週）

### 觀念

「模型選完就結束了嗎？」不是的，好的機器學習實踐者會用嚴謹的方法來評估模型好壞。

**交叉驗證（Cross-Validation）**：把資料分成 K 份，每次拿其中 1 份當測試、其餘當訓練，重複 K 次取平均。這樣能更公平地評估模型能力。

**混淆矩陣（Confusion Matrix）**：一個表格，顯示模型分對分錯的細節。

```
                預測：正    預測：負
實際：正         TP          FN
實際：負         FP          TN
```

**精確率 vs 召回率**：精確率（Precision）= 在我說「是」的裡面，真的有多少是「是」。召回率（Recall）= 在真正是「是」的裡面，我抓到了多少。兩者通常是 trade-off。

### 實作 5-1：完整的模型評估流程

```python
"""
單元 5-1：交叉驗證與混淆矩陣
目標：學會用多種指標全面評估模型
"""
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

    print(f"  測試準確率: {model.score(X_test_s, y_test):.4f}")
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
plt.show()
```

### 實作 5-2：超參數搜尋

```python
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
```

### Claude Code 驗證指引

```
請用 Claude Code 驗證：
1. 兩個模型在乳癌資料集的準確率都應 > 93%
2. 混淆矩陣能正確產出並顯示 TP/FP/TN/FN
3. GridSearchCV 能找到最佳參數組合
4. 醫療場景中，Recall（召回率）比 Precision 更重要——
   思考：漏診（FN）和誤診（FP）哪個後果更嚴重？
```

---

## 第六單元：非監督式學習（第 7 週）

### 觀念

非監督式學習沒有「標準答案」，模型自行從資料中發現結構。最常用的是**分群（Clustering）**。

**K-Means 演算法**：
1. 隨機放 K 個「中心點」
2. 每個資料點歸到最近的中心點
3. 重新計算各群的中心
4. 重複 2-3 直到穩定

**如何決定 K？** 用「手肘法（Elbow Method）」：嘗試不同 K 值，畫出 inertia（群內距離總和）的曲線，找到「彎曲」的位置。

**降維（Dimensionality Reduction）**：把高維度資料壓縮到 2D 或 3D 以便視覺化。PCA（主成分分析）是最常用的方法。

### 實作 6-1：K-Means 分群

```python
"""
單元 6-1：K-Means 分群與手肘法
目標：學會非監督式學習的基本流程
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 生成模擬資料（3 群）
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# 手肘法：測試 K=1~10
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 圖 1: 手肘法
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].axvline(x=3, color='r', linestyle='--', alpha=0.5, label='K=3')
axes[0].legend()

# 圖 2: K=3 分群結果
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km3.fit_predict(X)
for i in range(3):
    mask = labels == i
    axes[1].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Cluster {i}')
axes[1].scatter(km3.cluster_centers_[:, 0], km3.cluster_centers_[:, 1],
               c='red', marker='X', s=200, label='Centers')
axes[1].set_title('K-Means Clustering (K=3)')
axes[1].legend()

# 圖 3: 真實標籤（對照）
for i in range(3):
    mask = y_true == i
    axes[2].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'True {i}')
axes[2].set_title('True Labels (for comparison)')
axes[2].legend()

plt.tight_layout()
plt.savefig('unit6_kmeans.png', dpi=100)
plt.show()
```

### 實作 6-2：PCA 降維視覺化

```python
"""
單元 6-2：PCA 降維
目標：將高維資料降到 2D 視覺化
"""
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
plt.show()

# 累積變異比例
pca_full = PCA().fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_90 = np.argmax(cumvar >= 0.90) + 1
print(f"\n要保留 90% 資訊，需要 {n_90} 個主成分（原本 64 維）")
```

### Claude Code 驗證指引

```
請用 Claude Code 驗證：
1. 手肘法圖表在 K=3 處有明顯轉折
2. K-Means 分群結果與真實標籤高度吻合
3. PCA 降到 2 維後，不同數字在散佈圖中能大致分開
4. 保留 90% 變異大約需要 20~25 個主成分
```

---

## 第七單元：綜合專案（第 8～10 週）

### 專案：鐵達尼號生還預測

這是一個經典的機器學習入門專案。你將運用前面學到的所有技能，從頭到尾完成一個完整的 ML 工作流程。

```python
"""
綜合專案：鐵達尼號生還預測
目標：整合所有學到的技能，完成一個端到端的 ML 專案
"""
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
print(f"\n生還率: {df['survived'].mean():.2%}")

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

print(f"\n=== 最佳模型 ===")
print(f"參數: {grid.best_params_}")
print(f"CV 準確率: {grid.best_score_:.4f}")
print(f"測試準確率: {grid.score(X_test_processed, y_test):.4f}")

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
plt.show()

print("\n=== 專案完成！===")
print("你已經完成了一個完整的機器學習專案流程：")
print("  資料探索 → 特徵工程 → 前處理 → 模型訓練 → 調參 → 評估")
```

### Claude Code 驗證指引

```
請用 Claude Code 完整執行本專案，驗證：
1. 資料載入正確，生還率約 38%
2. 特徵工程成功建立 family_size 和 is_alone
3. 三個模型的 CV 準確率都在 78%~83% 之間
4. GridSearchCV 找到的最佳模型測試準確率 > 78%
5. 特徵重要性中，sex 和 fare 應排在前列
6. 所有圖表正確產出

進階挑戰：
- 嘗試加入更多特徵工程（例如從 Name 欄位提取 Title）
- 比較是否使用 StandardScaler 對樹模型的影響
- 用 RandomizedSearchCV 替代 GridSearchCV 看效率差異
```

---

## 學習路線圖總覽

```
第 1 週  ➜  什麼是機器學習？資料集探索
第 2 週  ➜  資料前處理：缺漏值、縮放、Pipeline
第 3-4 週 ➜  分類：KNN、決策樹、隨機森林
第 5 週  ➜  迴歸：線性迴歸、過擬合與欠擬合
第 6 週  ➜  模型評估：交叉驗證、混淆矩陣、調參
第 7 週  ➜  非監督式學習：K-Means、PCA
第 8-10 週 ➜  綜合專案：鐵達尼號生還預測
```

## 延伸學習建議

完成本計劃後，建議的下一步學習方向：

- **深入 scikit-learn**：SVM、XGBoost、Stacking Ensemble
- **深度學習入門**：PyTorch 或 TensorFlow / Keras
- **實戰平台**：Kaggle 競賽（從 Getting Started 類型開始）
- **數學基礎**：線性代數、微積分、機率統計（配合 3Blue1Brown 影片）
- **MLOps 入門**：MLflow、模型部署

## 附錄：常見錯誤與除錯

| 錯誤訊息 | 原因 | 解法 |
|----------|------|------|
| `ValueError: could not convert string to float` | 有未編碼的文字特徵 | 用 LabelEncoder 或 OneHotEncoder |
| `ValueError: Input contains NaN` | 資料有缺漏值 | 用 SimpleImputer 填補 |
| `ConvergenceWarning` | 模型未收斂 | 增加 `max_iter` 參數 |
| `train score 高但 test score 低` | 過擬合 | 降低模型複雜度、增加資料、正則化 |

---

> **提示**：每完成一個單元，建議用自己的話寫一段「我學到了什麼」的筆記。教別人是最好的學習方式！
