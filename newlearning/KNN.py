"""
    1、获取数据
    2、数据集划分
    3、特征工程（标准化
    4、KNN预估器流程
    5、模型评估
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knn_iris():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    y_pred = estimator.predict(x_test)
    print(y_test == y_test)

    score = estimator.score(x_test, y_test)
    print(score)


def knn_iris_gscv():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier()
    # 加入网格搜索与交叉验证
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9]}
    # estimator为实例化的转换器，param_grid为需手动改变的参数，cv为要折叠几次
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    # 最佳参数：best_params_
    print(estimator.best_params_)
    # 最佳结果：best_score_
    # 最佳估计器：best_estimator_
    # 交叉验证结果：cv_results_

    y_pred = estimator.predict(x_test)
    print(y_test == y_test)

    score = estimator.score(x_test, y_test)
    print(score)


if __name__ == '__main__':
    knn_iris()









































