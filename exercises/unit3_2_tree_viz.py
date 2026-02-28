"""
單元 3-2：決策樹視覺化
目標：理解決策樹如何做出判斷
"""
import matplotlib
matplotlib.use('Agg')
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
plt.savefig('exercises/unit3_decision_tree.png', dpi=120)
print("決策樹圖已儲存為 exercises/unit3_decision_tree.png")
