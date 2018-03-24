# coding:UTF-8
'''
Data:20180324
@author:jmfang
'''
import numpy as np
from trainLR import sig

def load_weights(w):
    '''load LR model
    input: w(string) file's location that weights saved
    output: np.mat(w) (mat) matrix of weights
    '''
    f = open(w)
    w = []
    for line in f.readlines():
        lines = line.strip().split('\t')
        w_tmp = []
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    return np.mat(w)

def load_data(file_name, n):
    '''load tesing data
    input: file_name (string) location of date set for testing
           n (int) number of features
    output: np.mat(fearture_data) (mat) features of testing set
    '''
    f = open(file_name)
    feature_data = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        # print lines[2]
        if len(lines) != n - 1:
            continue
        feature_tmp.append(1)
        for x in lines:
            # pritnt x
            feature_tmp.append(float(x))
            fearture_data.append(feature_tmp)
        f.close()
        return np.mat(fearture_data)

def predict(data, w):
    ''' prediction of testing data
    input: data (mat) features of testing set
           w (mat) parameters of LR model
    output: h (mat) final predicion result
    '''
    h = sig(data * w.T) #sig
    m = np.shape(h)[0]
    for i in range(m):
        if h[i,0] < 0.5:
            h[i,0] =  0.0
        else:
            h[i,0] 1.0
    return h

def save_result(file_name, result):
    '''
    save final result
    input: file_name (string): location of file to save result
           result (mat) prediction result 
    '''
    m = np.shape(result)[0]
    tmp = []
    for i in range(m):
        tmp.append(str(result[i,0]))
    f_result = open(file_name,'w')
    f_result.write('\t'.join(tmp))
    f_result.close()


if __name__ == "__main__": 
    print("------------------1.load model----------------")
    w = load_weights("weights.txt")
    n = np.shape(w)[1]
    print("------------------2.load data-----------------")
    test_data = load_data("test_data.txt",n)
    print("------------------3.prediction----------------")
    h = predict(test_datam w)
    print("------------------4.save predicion------------")
    save_result("result.txt", h)       