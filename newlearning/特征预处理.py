import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


def minmax_demo():
    """
        归一化处理
    """
    # 1、获取数据
    data = pd.read_csv("data.txt")
    data = data.iloc[:, :3]     # 取前三列
    # 2、实例化一个转换器类
    transfer = MinMaxScaler()
    # MinMaxScaler(feature_range=[0,1]) 默认值为0-1，可以修改，例如[2,3]
    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)


def stand_demo():
    """
        标准化处理（使其均值为0，标准差为1）
        x' = (x-平均值)/标准差
    """
    # 1、获取数据
    data = pd.read_csv("data.txt")
    data = data.iloc[:, :3]     # 取前三列
    # 2、实例化一个转换器类
    transfer = StandardScaler()
    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)


def variance_demo():
    """
        过滤低方差特征
    """
    # 1、获取数据
    data = pd.read_csv("factor_returns.csv")
    data = data.iloc[:, 1:-2]
    # 2、实例化一个转换器类
    transfer = VarianceThreshold()
    # (threshold = int)设置域值
    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new, data_new.shape)

    # 计算两个变量之间的相关系数
    r1 = pearsonr(data['pe_ratio'], data['pb_ratio'])
    print("相关系数：\n", r1)
    r2 = pearsonr(data['revenue'], data['total_expense'])
    print("相关系数：\n", r2)


def pca_demo():
    """
        PCA降维
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    # 1、实例化一个转化器类
    transfer = PCA(n_components=2)
    # n_components = int/float 当int=n时，为保留几个特征；当float=a时，为保留百分之多少的有效数据（num/2！=0，5）
    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)


if __name__ == '__main__':
    # variance_demo()
    pca_demo()


















