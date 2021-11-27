import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba


def datasets_demo():
    #获取数据集
    iris=load_iris()
    # print(iris)
    # print(iris["DESCR"])
    # print(iris.data.shape)
    # print(iris.target)

    # 数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data, iris.target, random_state=22, test_size=0.2)
    print("训练集的特征值：\n", x_train, x_train.shape)


def dic_demo():
    """
    字典特征抽取
    """
    data=[{"city":"北京","temperature": 100},{"city":"上海","temperature": 60},{"city":"深圳","temperature": 30}]
    # 1、实例化一个转换器类
    transfer=DictVectorizer()
    # DictVectorizer(sparse=True)为默认值，返回一个sparse矩阵（稀疏矩阵），将非零值按位置显示出来
    # (0, 1) 1.0
    # (0, 3) 100.0
    # (1, 0) 1.0
    # (1, 3) 60.0
    # (2, 2) 1.0
    # (2, 3) 30.0
    # 当sparse=False时
    # [[0.   1.   0. 100.]
    #  [1.   0.   0.  60.]
    #  [0.   0.   1.  30.]]

    # 2、调用fit_transform
    data_new=transfer.fit_transform(data)
    print(data_new)
    print(transfer.get_feature_names())     # 获取每一列的特征名


def count_demo():
    """
        文本特征抽取：CountVecotrizer
    """
    data = ["life is short, i like like python","life is too long,i dislike python"]
    # 1、实例化一个转换器类
    transfer=CountVectorizer()
    # 2、调用fit_transform
    data_new=transfer.fit_transform(data)
    # CountVectorizer中没有设置sparse=False关键字；若需得到矩阵 data_new.toarry即可
    print(transfer.get_feature_names())
    print(data_new)


def cut_word(text):
    """
        进行中文分词(对字符串进行）
    """
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo():
    """
        中文文本特征提取
    """
    data=["最初的加速膨胀被称为暴胀时期，之后已知的四个基本力分离",
          "宇宙逐渐冷却并继续膨胀，允许第一个亚原子粒子和简单的原子形成",
          "暗物质逐渐聚集，在引力作用下形成泡沫一样的结构，大尺度纤维状结构和宇宙空洞"]
    # 将中文文本进行分词
    data_new=[]
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_final=["宇宙","的"])
    # stop_final=[]中可以放入需要去除的特征值，比如”的“；”你“；”我“之类的
    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print(transfer.get_feature_names())
    print(data_final)


def tfidf_demo():
    """
        用TF-IDF方法进行文本特征抽取
        TF 为词频：该词出现次数/该文章总词数
        IDF 为逆向文档频率 log10（文章总数/出现该词的文章数）
        TF-IDF=IF*IDF
    """
    data = ["最初的加速膨胀被称为暴胀时期，之后已知的四个基本力分离",
            "宇宙逐渐冷却并继续膨胀，允许第一个亚原子粒子和简单的原子形成",
            "暗物质逐渐聚集，在引力作用下形成泡沫一样的结构，大尺度纤维状结构和宇宙空洞"]
    # 将中文文本进行分词
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = TfidfVectorizer()
    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print(transfer.get_feature_names())
    print(data_final)


if __name__ == '__main__':
    # datasets_demo()
    # dic_demo()
    # count_demo()
    # count_chinese_demo()
    tfidf_demo()

# load和fetch返回类型datasets.base.Bunch(字典格式)
# data：特征值数据数组，是[n_samples * n_features]的二维numpy.ndarry数组
# target:标签数组，是n_sample的一维numpy.ndarry数组
# DESCR:数据描述
# feature_names:特征名，新闻数据，手写数字，回归数据集没有
# target_name:标签名


"""
    sklearn.model_selectin.train_test_split(array,*options)
    x数据集的特征值
    y数据集的标签值
    test_size数据集的大小，一般为float
    random_state 随机数种子，不同的种子会造成不同的随机采样结果。相同种子的采样结果相同
    return训练集特征值，测试特征值，训练目标值，测试目标集
            x_train   x_test   y_train  y_test
    

"""



















