import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

df_iris = pd.read_csv("Iris.csv")

# print(df_iris)


df_iris.drop('Id', axis=1, inplace=True)

df_iris_X = df_iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

df_iris_y = df_iris.Species


tree_clf = DecisionTreeClassifier(max_depth=2, random_state=36)
tree_clf.fit(df_iris_X, df_iris_y)

# print(tree_clf.fit(df_iris_X, df_iris_y))


# Visualization --> DT.py

# from sklearn.tree import export_graphviz
# from sklearn import tree
# from IPython.display import SVG
# from graphviz import Source
# from IPython.display import display

# labels = df_iris_X.columns

# graph = Source(tree.export_graphviz(tree_clf, feature_names = labels, class_names = df_iris_y.unique(), max_depth = 2, filled = True))
# display(SVG(graph.pipe(format='svg')))