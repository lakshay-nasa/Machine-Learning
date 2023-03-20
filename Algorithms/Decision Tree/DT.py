import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

df_iris = pd.read_csv("Iris.csv")

# print(df_iris)

df_iris_X = df_iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

df_iris_y = df_iris.Species


tree_clf = DecisionTreeClassifier(max_depth=2, random_state=36)
tree_clf.fit(df_iris_X, df_iris_y)

# print(tree_clf.fit(df_iris_X, df_iris_y))