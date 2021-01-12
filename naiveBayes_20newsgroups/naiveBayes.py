import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold


#数据加载 & 处理
def create_data():
    corpus_path = './naiveBayes'
    sample_cate = ['alt.atheism', 'soc.religion.christian']
    X = fetch_20newsgroups(data_home=corpus_path, subset='all', categories=sample_cate,
                           remove=('headers', 'footers', 'quotes'))
    return X

# 向量化处理
def vectorizer_data(data):
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_vector = vectorizer.fit_transform(data.data)
    X_vector = X_vector.todense()  # 将稀疏矩阵转换成一般矩阵
    # print(X_vector.shape)
    return X_vector

X = create_data()
X_vector = vectorizer_data(X)

class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(x):
        return sum(x)/float(len(x))

    # 方差
    def std(self, x):
        avg = self.mean(x)
        return np.sqrt(sum(np.power(x_i-avg, 2) for x_i in x)/float(len(x)))

    # 概率密度函数
    def gaussian_prob(self, x, mean, std):
        exp = np.exp(-1*(np.power(x-mean, 2))/(2*np.power(std, 2)+(1e-5)))
        return (1/(np.sqrt(2*np.pi*math.pow(std, 2))+(1e-5)))*exp

    # 计算训练的均值和方差
    def mean_and_std(self, x):
        mean_and_std = [(self.mean(i), self.std(i)) for i in zip(*x)]
        return mean_and_std

    # 分别求出数学期望和标准差
    def fit(self, x, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(x, y):
            data[label].append(f)
        self.model = {label: self.mean_and_std(value) for label,value in data.items()}
        return "GaussianNB train Done!"

    # 计算概率
    def prob(self, data):
        probability = {}
        for label, value in self.model.items():
            probability[label] = 1
            # print(range(len(value)))
            for i in range(len(value)):
                mean, std = value[i]
                probability[label] *= self.gaussian_prob(data[i], mean, std)
        return probability

    # 类别
    def predict(self, x_test):
        label = sorted(self.prob(x_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    # 精确度
    def score(self, x_test, y_test):
        right = 0
        for x, y in zip(x_test, y_test):
            label = self.predict(x)
            if label == y:
                right += 1
        return right / float(len(x_test))


kf = KFold(n_splits=5, random_state=1, shuffle=True)
k = 0
for Xtrain, Xtest in kf.split(X_vector):
    X_train = np.array(X_vector)[Xtrain, :]
    X_test = np.array(X_vector)[Xtest, :]
    y_train = np.array(X.target)[Xtrain]
    y_test = np.array(X.target)[Xtest]

    model = NaiveBayes()
    model.fit(X_train, y_train)

    k = k + 1
    predictRes = 0
    scoreRes = 0
    for j in range(len(Xtest)): # 此处变量j 和54行循环的变量i 命名要区分开
        predictRes += model.predict(X_test[j, :])
        scoreRes += model.score([X_test[j, :].tolist()], [y_test[j].tolist()]) # 需要转化成列表
    print("第" + str(k) + "次交叉验证的预测结果为：" ,predictRes/len(Xtest))
    print("第" + str(k) + "次交叉验证测试集的精确度为：" ,scoreRes/len(Xtest))
    print("第" + str(k) + '次交叉验证的测试集的score为: ', model.score(X_test, y_test))


