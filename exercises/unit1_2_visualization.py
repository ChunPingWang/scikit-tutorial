"""
單元 1-2：用圖表觀察資料
目標：學會用視覺化發現資料中的規律
"""
import matplotlib
matplotlib.use('Agg')
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
plt.savefig('exercises/unit1_visualization.png', dpi=100)
print("圖表已儲存為 exercises/unit1_visualization.png")
