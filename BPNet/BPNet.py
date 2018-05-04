import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap



class Perceptron(object):
    """
    Parameters
    ------------
    eta : float
        学习率 (between 0.0 and 1.0)
    n_iter : int
        迭代次数
    Attributes
    -----------
    w_ : 1d-array
        权重
    errors_ : list
        误差
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def load_data(file_name):
    '''导入数据
    input:  file_name(string):文件的存储位置
    output: feature_data(mat):特征
            label_data(mat):标签
            n_class(int):类别的个数
    '''
    # 1、获取特征
    f = open(file_name)  # 打开文件
    feature_data = []
    label_tmp = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(int(lines[-1]))      
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    
    # 2、获取标签
    m = len(label_tmp)
    n_class = len(set(label_tmp))  # 得到类别的个数
    print("n_class", n_class)
    
    label_data = np.mat(np.zeros((m, n_class)))
    for i in range(m):
        label_data[i, label_tmp[i]] = 1
    
    return np.mat(feature_data), label_data, n_class

def plot_decision_regions(X, y, classifier, resolution=0.01):
    """
    可视化分类器
    :param X:  样本特征向量
    :param y:  样本标签向量
    :param classifier: 分类器
    :param resolution: 残差
    :return:
    """

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx2.min(), xx2.max())

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


feature, label, n_class = load_data("data.txt")
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(feature, label)

# 使用散点图可视化样本
# plt.scatter(feature[:50, 0], feature[:50,1], color='red', marker='o', label='setosa')
# plt.scatter(feature[50:100, 0], feature[50:100, 1], color='blue', marker='x', label='versicolor')
# plt.xlabel('petal length')
# plt.ylabel('sepal length')
# plt.legend(loc='upper left')
# plt.show
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of misclassifications')
# plt.show()

# 调用可视化分类器函数
plot_decision_regions(feature, label, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()