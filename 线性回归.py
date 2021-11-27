from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  # 正规方程
from sklearn.linear_model import Ridge  # 岭回归
from sklearn.linear_model import SGDRegressor  # 梯度下降
from sklearn.metrics import mean_squared_error  # 均方误差
import joblib  # 模型保存与加载


def linear1():
    """
        正规方程的优化方法对波士顿房价进行预测
    """
    # 1、获取数据
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3、标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5、得出模型
    print("正规方程权重系数为：\n", estimator.coef_)
    print("正规方程偏置为：\n", estimator.intercept_)

    # 6、模型评估
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程均方误差为：\n", error)

    return None


def linear2():
    """
        梯度下降的优化方法对波士顿房价进行预测
    """
    # 1、获取数据
    boston = load_boston()

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3、标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # # 4、预估器
    # estimator = SGDRegressor()
    # estimator.fit(x_train, y_train)
    # 
    # # 保存模型
    # joblib.dump(estimator, "my_ridge.pkl")
    # 加载模型
    estimator = joblib.load("my_ridge.pkl")

    # 5、得出模型
    print("梯度下降权重系数为：\n", estimator.coef_)
    print("梯度下降偏置为：\n", estimator.intercept_)

    # 6、模型评估
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降均方误差为：\n", error)

    return None


def linear3():
    """
        岭回归的优化方法对波士顿房价进行预测
    """
    # 1、获取数据
    boston = load_boston()

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3、标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、预估器
    estimator = Ridge()
    estimator.fit(x_train, y_train)

    # 5、得出模型
    print("岭回归权重系数为：\n", estimator.coef_)
    print("岭回归偏置为：\n", estimator.intercept_)

    # 6、模型评估
    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归均方误差为：\n", error)

    return None


if __name__ == '__main__':
    linear1()
    linear2()
    linear3()
