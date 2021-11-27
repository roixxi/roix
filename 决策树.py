from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


def decison_iris():
    """
        用决策树对鸢尾花进行分类
    """
    # 1、获取数据集
    iris = load_iris()

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3、决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    # criterion=用什么方法进行决策(默认基尼系数）

    estimator.fit(x_train, y_train)

    # 4、模型评估
    y_pred = estimator.predict(x_test)
    print(y_test == y_test)
    score = estimator.score(x_test, y_test)
    print(score)

    # 5、可视化决策树
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)


if __name__ == '__main__':
    decison_iris()

    