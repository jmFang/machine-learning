# coding:UTF-8
'''
Date:20180324
@author:jmfang
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def load_data(data_file):
    '''
    input:file_name(string) location of training data
    output:fearture_data(mat) features
           label_data(mat) labels
    '''
    feature = []
    label = []
    f = open(data_file)
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        # feature_tmp.append(1)
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        if lines[-1] == '+':
            label.append(1)
        elif lines[-1] == '-':
            label.append(0)
    f.close()
    return np.mat(feature), np.mat(label).T

# 画出散点图和超平面     
def showLogRegression(weights, train_x, train_y):
    numSamples, numFeartures = np.shape(train_x)
    if numFeartures != 3:
        print('sorry numfeature is not 3')
        return 1
    for i in range(numSamples):
        if train_y[i,0] == '-':
            plt.plot(train_x[i,0], train_x[i,1],'or')
        elif train_y[i,0] == '+':
            plt.plot(train_x[i,0],train_x[i,1],'xb')

    # min_x = min(train_x[:,1])[0,0]
    # max_x = max(train_x[:,1])[0,0]
    # weights = weights.getA()
    # y_min_x = float(-weights[0] - weights[1]*min_x)/weights[2]
    # y_max_x = float(-weights[0] - weights[1]*max_x)/weights[2]
    # plt.plot([min_x,max_x],[y_min_x,y_max_x],'-g')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def main():
    feature, label = load_data("data.txt")
    print(feature)
    print(label)
    clf = svm.SVC(kernel='linear')
    clf.fit(feature, label)
    w = clf.coef_[0]

    # dis = w.dot(w.T)
    dis = np.mat(w) * np.mat(w).T
    print("dis:",2/dis)

    a = -w[0]/w[1] #斜率  
    #画图划线  
    xx = np.linspace(-1,3) #(-5,5)之间x的值  
    print("clf.intercept_[0]: ",clf.intercept_)
    # 
    yy = a*xx - clf.intercept_[0]/w[1] #xx带入y，截距  
    # w = np.ones((np.shape(label)[0], 1))
    # showLogRegression(w, feature, label)
    #画出与点相切的线  
    b = clf.support_vectors_[0]  
    print("b：",b)
    yy_down = a*xx + (b[1] - a*b[0])  
    b = clf.support_vectors_[-2]  
    yy_up = a*xx + (b[1] - a*b[0])
    print("weights:", w)  
    print("a:", a) 
    print("clf.n_support_c: ", clf.n_support_) 
    print("support_vectors_:"+"\n",clf.support_vectors_)  
    print("clf.coef_:",clf.coef_)  
      
    plt.figure(figsize=(4,5))  
    # 超平面
    plt.plot(xx,yy, 'k-', label='super plain')  
    plt.plot(xx,yy_down, label='down')  
    plt.plot(xx,yy_up, label='up')  
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(clf.support_vectors_[0,0],clf.support_vectors_[0,1],s=60, c='red') 
    plt.scatter(clf.support_vectors_[2,0],clf.support_vectors_[2,1],s=60, c='blue') 
    for i in range(np.shape(feature)[0]):
         if label[i,0] == 1: 
            plt.plot(feature[i,0],feature[i,1],'+b')
         elif label[i,0] == 0:
            plt.plot(feature[i,0],feature[i,1],'or') 
    plt.axis('tight')  
    plt.legend()
    plt.show()


main()