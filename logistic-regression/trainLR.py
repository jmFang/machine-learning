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
        for i in range(len(lines)):
            if i == 1:
                label_tmp.append(float(lines[i]))
            elif i != 0:
                feature_tmp.append(float(lines[i]))
        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    f.close() #close file
    # list 转 矩阵
    # feature_t,_,_= featureNormal(np.mat(feature_data))
    # label_t,_,_ = featureNormal(np.mat(label_data))
    # return feature_t, label_t
    return np.mat(feature_data), np.mat(label_data)

def sig(x):
    ''' Sigmoid function
    input: x(mat),fearture * weights
    output: sigmoid(x) (mat), Sigmoid value
    '''
    return 1.0 / (1 + np.exp(-x))

def train_process(feature, label, maxCycle, alpha, lam = 1):
    '''use gradient desent algorithm to train LR model
    input: feature (mat) feartures
           label (mat) labels
           maxCycle (int) maximum of looping times
           alpha (float) learning rate
    output: w(mat) weights
    '''
    # number of colume (number of fearture)
    n = np.shape(feature)[1]
    m = np.shape(label)[0]
    # initialize the weights
    w = np.mat(np.ones((n,1)))
    cost_histroy = np.zeros((maxCycle+1, 1))
    weight_his = np.zeros((maxCycle+2,n))
    i = 0
    convergence_rate = 0.001
    while i <= maxCycle:
        h = sig(feature * w)
        err = h - label
        cost_histroy[i] = error_rate(h,label)
        if i % 100 == 0:
            print('\t-----------iter= '+str(i)+\
                ',train error rate= ' + str(error_rate(h,label)))
        if i > 0 and cost_histroy[i-1,0] - cost_histroy[i,0] > 0 and cost_histroy[i-1,0] - cost_histroy[i, 0] <= convergence_rate:
            print("convergenced with rate :" + str(cost_histroy[i-1,0] - cost_histroy[i,0]) +\
                "at iter " + str(i))
            # update weights
        w = w - alpha * ((1/m) * (feature.T * err + lam*w))
        i += 1
        weight_his[i,:] = w.T
    for i in range(len(w)):
        w[i] = np.round(w[i],2)
    return w, cost_histroy, weight_his

def featureNormal(X):
    X_normal = X
    n = X.shape[1]
    mu = np.zeros((1, n)) 
    sigma = np.zeros((1,n)) 
    for i in range(n):
        mu[0,i] = np.mean(X[:,i])
        max = np.max(X[:,i])
        min = np.min(X[:,i])
        sigma[0, i] = np.std(X[:,i])
        for j in range(X.shape[0]):
            if sigma[0,i] != 0:
                X_normal[j,i] = round(X_normal[j,i] - mu[0,i] /sigma[0,i],2)
    # X_normal = (X - mu) / sigma
    if n > 1:
        for j in range(X.shape[0]):
            X_normal[j:0] = 1
    return X_normal, mu, sigma

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
    iters = 300
    learning_rate = 3
    # 1. load data
    print("-------------1.loading data-------------")
    feature, label = load_data("medical.txt")
    print(feature)
    print(label)
    # 2. training LR model
    print("-------------2.training model-----------")
    w, cost, wh = train_process(feature, label, iters, learning_rate)
    print(w)
    # save final model
    print("-------------3.save model---------------")
    save_model("weights.txt", w)
    # plot figure
    print("-------------4.plot figure--------------")
    # showLogRegression(w, feature, label)
    plt.plot(cost,label='learning rate:' + str(learning_rate), color='red')
    plt.title("cost function")
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.legend()
    plt.show()
    # plt.plot(wh[:,0], label='theta(0)')
    # plt.plot(wh[:,1], label='theta(1)')
    # plt.plot(wh[:,2], label='theta(2)')
    # plt.plot(wh[:,3], label='theta(3)')
    # plt.plot(wh[:,4], label='theta(4)')
    # plt.xlabel('iteration')
    # plt.ylabel('theta(i)')
    # plt.title('learning rate: ' + str(learning_rate))
    # plt.legend()
    # plt.show()