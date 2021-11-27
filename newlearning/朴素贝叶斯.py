from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def nb_news():
    """
        用朴素贝叶斯对新闻进行分类
    """
    # 1、获取数据
    news = fetch_20newsgroups(subset='all')

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

    # 3、特征工程（文本特征抽取
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、朴素贝叶斯算法估计器流程
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 5、模型评估
    y_pred = estimator.predict(x_test)
    print(y_pred == y_test)

    score = estimator.score(x_test, y_test)
    print(score)


if __name__ == '__main__':
    nb_news()



























