"""
單元 4-2：過擬合 vs 欠擬合的視覺化
目標：理解模型複雜度與泛化能力的關係
"""
import matplotlib
matplotlib.use('Agg')
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
print("過擬合示意圖已儲存")
