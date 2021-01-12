import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 数据加载 & 处理
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names) # 直接读到pandas的数据框架中

    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
    df['label'] = iris.target # 样本的类型，[0,1,2]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data.shape)
    return data[:, :2], data[:, -1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

class SVM:
    def __init__(self, C=1.0,max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self.kernel_type = kernel
        self.C = C # 松弛变量

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.

        self.alpha = np.ones(self.m)
        self.E_i = [self.E(i) for i in range(self.m)]

    # g(x) 预测值，输入
    def g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j]*self.Y[j]*self.kernel(self.X[i],self.X[j])
        return r

    def KKT(self, i):
        Y_g = self.g(i)*self.Y[i]
        if self.alpha[i] == 0:
            return Y_g >= 1
        elif 0 < self.alpha[i] and self.alpha[i] < self.C:
            return Y_g == 1
        else:
            return Y_g <= 1

    def kernel(self, x1, x2):
        if self.kernel_type == 'linear':
            return sum([x1[k]*x2[k] for k in range(self.n)])
        elif self.kernel_type == 'poly':
            return (sum([x1[k]*x2[k] for k in range(self.n)]) +1)**2
        return 0

    def E(self, i):
        return self.g(i) - self.Y[i]

    def init_alpha(self):
        # 外层循环先遍历所有满足 0<a<c 的样本点，检验是否满足 KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)

        for i in index_list:
            if self.KKT(i):
                continue
            E1 = self.E_i[i]

            # 如果 E1 是正的，选择最小的，如果 E1 是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x:self.E_i[x])
            else:
                j = max(range(self.m), key=lambda x:self.E_i[x])
            return i,j

    def compare(self, alpha, L, H):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

    def fit(self, features, labels):
        self.init_args(features, labels)

        for t in range(self.max_iter):
            # train
            i1, i2 = self.init_alpha()

            # boundary
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C , self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E_i[i1]
            E2 = self.E_i[i2]
            # eta = K11 +K22 -2*K12
            eta = self.kernel(self.X[i1],self.X[i1]) + self.kernel(self.X[i2],self.X[i2]) - 2*self.kernel(self.X[i1],self.X[i2])
            if eta <= 0:
                continue

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E2-E1)/eta
            alpha2_new = self.compare(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2]*(self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1])\
                 - self.Y[i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1])\
                 - self.Y[i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new)/2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E_i[i1] = self.E(i1)
            self.E_i[i2] = self.E(i2)

        return "Done!"

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count/len(X_test)

    def weight(self):
        # Linear model
        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w

    def plotFeature(self, X_train, y_train):
        xcord1 = []; ycord1 = []
        xcord2 = []; ycord2 = []
        for i in range(len(X_train)):
            if(y_train[i] == 1):
                xcord1.append(X_train[i, 0])
                ycord1.append(X_train[i, 1])
            else:
                xcord2.append(X_train[i, 0])
                ycord2.append(X_train[i, 1])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red')
        ax.scatter(xcord2, ycord2, s=30, c='green')
        x = np.arange(2, 7.0, 0.1)
        y = (-self.b-self.weight()[0] * x)/self.weight()[1]
        #y = (self.weight() * x) + self.b
        ax.plot(x, y)
        plt.xlabel('X1'); plt.ylabel('X2')
        plt.show()


svm = SVM(C=0.8, max_iter=100)
max_train_score = 0.0; max_test_score = 0.0
for i in range(10):
    svm.fit(X_train, y_train)
    if(svm.score(X_train, y_train) > max_train_score):
        max_train_score = svm.score(X_train, y_train)
    if(svm.score(X_test, y_test) > max_test_score):
        max_test_score = svm.score(X_test, y_test)

# Test on Training data
print("训练集的精确度为：" + str(max_train_score))

# Test on Testing data
print("测试集的精确度为：" + str(max_test_score))
print(svm.weight())

# ploting
svm.plotFeature(X_train, y_train)
svm.plotFeature(X_test, y_test)