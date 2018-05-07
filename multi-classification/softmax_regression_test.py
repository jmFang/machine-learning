# coding:UTF-8

import numpy as np
import random as rd
import pandas as pd
import math
import csv
def load_weights(weights_path):
    '''导入训练好的Softmax模型
    input:  weights_path(string)权重的存储位置
    output: weights(mat)将权重存到矩阵中
            m(int)权重的行数
            n(int)权重的列数
    '''
    f = open(weights_path)
    w = []
    for line in f.readlines():
        w_tmp = []
        lines = line.strip().split("\t")
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    weights = np.mat(w)
    m, n = np.shape(weights)
    return weights, m, n

def loaddatafromCSV(filename):
    feature_data =[]
    with open(filename) as f:
        reader = csv.reader(f)
        for index,row in enumerate(reader):
            feature_tmp = []
            feature_tmp.append(1)
            for i in range(len(row)-1):
                feature_tmp.append(float(row[i+1]))
            feature_data.append(feature_tmp)
    feature_ = np.mat(feature_data)
    n, m = np.shape(feature_)
    m_mean = np.mean(feature_[:,1:m], axis=0)
    m_max = np.max(feature_[:,1:m],axis=0)
    m_min = np.min(feature_[:,1:m], axis=0)
    m_mean = np.tile(m_mean,(n,1))
    m_min = np.tile(m_min, (n,1))
    m_max = np.tile(m_max, (n,1))
    feature_[:,1:m] = (feature_[:,1:m] - m_mean[:,0:m]) / (m_max[:,0:m] - m_min[:,0:m])
    return feature_

def predict(test_data, weights):
    '''利用训练好的Softmax模型对测试数据进行预测
    input:  test_data(mat)测试数据的特征
            weights(mat)模型的权重
    output: h.argmax(axis=1)所属的类别
    '''
    h = test_data * weights
    return h.argmax(axis=1)+1#获得所属的类别

def save_result(file_name, result):
    '''保存最终的预测结果
    input:  file_name(string):保存最终结果的文件名
            result(mat):最终的预测结果
    '''
    with open(file_name) as csvFile:
        rows = csv.reader(csvFile)
        with open("file.csv",'w', newline='') as f:
            writer = csv.writer(f)
            for index,row in enumerate(rows):
                tmp_row = row
                if index >= 1:
                    tmp_row[-1] = "cls_" + str(result[index-1,0])
                writer.writerow(tmp_row)
    # f_result = open(file_name, "w")
    # m = np.shape(result)[0]
    # for i in range(m):
    #     f_result.write(str(result[i, 0]) + "\n")
    # f_result.close()

if __name__ == "__main__":
    # 1、导入Softmax模型
    print ("---------- 1.load model ------------")
    w, m , n = load_weights("weights")
    # 2、导入测试数据
    print ("---------- 2.load data ------------")
    test_data = loaddatafromCSV("test_raw.csv")
    # 3、利用训练好的Softmax模型对测试数据进行预测
    print( "---------- 3.get Prediction ------------")
    result = predict(test_data, w)
    # 4、统计预测的正确率
    # acc = countAcc(result, label)
    # 5、保存最终的预测结果
    filepath = "sampleSubmission.csv"

    print( "---------- 4.save prediction ------------")
    save_result(filepath, result)
