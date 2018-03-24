# coding:UTF-8
'''
Date:20180324
@author:jmfang
'''
import numpy as np 
import matplotlib.pyplot as plt

def load_data(file_name):
    '''
    input:file_name(string) location of training data
    output:fearture_data(mat) features
           label_data(mat) labels
    '''
    f = open(file_name)
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = []
        label_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  #偏置项
        # feartures
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(float(lines[-1]))

        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    f.close() #close file
    # list 转 矩阵
    return np.mat(feature_data), np.mat(label_data)

def sig(x):
    ''' Sigmoid function
    input: x(mat),fearture * weights
    output: sigmoid(x) (mat), Sigmoid value
    '''
    return 1.0 / (1 + np.exp(-x))

def train_process(feature, label, maxCycle, alpha):
    '''use gradient desent algorithm to train LR model
    input: feature (mat) feartures
           label (mat) labels
           maxCycle (int) maximum of looping times
           alpha (float) learning rate
    output: w(mat) weights
    '''
    # number of colume (number of fearture)
    n = np.shape(feature)[1]
    # initialize the weights
    w = np.mat(np.ones((n,1)))
    i = 0
    while i <= maxCycle:
        i += 1
        h = sig(feature * w)
        err = label - h
        if i % 100 == 0:
            print('\t-----------iter= '+str(i)+\
                ',train error rate= ' + str(error_rate(h,label)))
            # update weights
            w = w + alpha * feature.T * err
    return w

def error_rate(h, label):
    '''compute current velue of cost function
    input: h(mat): prediction
           label (mat) : real value
    output: err/m (float) error rate
    '''
    m = np.shape(h)[0]
    sum_err = 0.0
    for i in range(m):
        if h[i,0] > 0 and (1 - h[i,0]) > 0:
            sum_err -= (label[i,0] * np.log(h[i, 0])+ 
                (1 - label[i,0]) * np.log(1 - h[i,0]))
        else:
            sum_err -= 0
    return sum_err / m

def save_model(file_name, w):
    ''' save final model
    input: file_name (string): file's name to save model
           w (mat) : weights of LR model
    '''
    m = np.shape(w)[0]
    f_w = open(file_name, 'w')
    w_array = []
    for i in range(m):
        w_array.append(str(w[i,0]))

    f_w.write("\t".join(w_array))
    f_w.close()

# 从矩阵中获取列向量
# def getCol(feature, index):
#     t = feature[:,index]
#     t1 = t.tolist()
#     res = []
#     for item in t1:
#         res.append(item[0])
#     return res
# 从矩阵中，根据分类label，分别获取列向量
# def classFied(feature, index,label):
#     t = feature[:,index]
#     t1 = t.tolist()
#     res1 = []
#     res2 = []
#     for i in range(len(t1)):
#         if label[i] == 0:
#             res1.append(t1[i][0])
#         else:
#             res2.append(t1[i][0])
#     return res1, res2
# 画出散点图和超平面     
def showLogRegression(weights, train_x, train_y):
    numSamples, numFeartures = np.shape(train_x)
    if numFeartures != 3:
        print('sorry numfeature is not 3')
        return 1
    for i in range(numSamples):
        if int(train_y[i,0]) == 0:
            plt.plot(train_x[i,1], train_x[i,2],'or')
        elif int(train_y[i,0]) == 1:
            plt.plot(train_x[i,1],train_x[i,2],'ob')

    min_x = min(train_x[:,1])[0,0]
    max_x = max(train_x[:,1])[0,0]
    weights = weights.getA()
    y_min_x = float(-weights[0] - weights[1]*min_x)/weights[2]
    y_max_x = float(-weights[0] - weights[1]*max_x)/weights[2]
    plt.plot([min_x,max_x],[y_min_x,y_max_x],'-g')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ == "__main__":
    # 1. load data
    print("-------------1.loading data-------------")
    feature, label = load_data("data.txt")
    # 2. training LR model
    print("-------------2.training model-----------")
    w = train_process(feature, label, 10000, 0.01)
    # save final model
    print("-------------3.save model---------------")
    save_model("weights.txt", w)
    # plot figure
    print("-------------4.plot figure--------------")
    showLogRegression(w, feature, label)