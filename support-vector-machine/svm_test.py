#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

# from numpy import *
import numpy as np
import svm_train as SVM

################## test svm #####################
## step 1: load data
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
            label.append(-1)
    f.close()
    return np.mat(feature), np.mat(label).T

print ("step 1: load data...")
dataSet, labels = load_data('data.txt')
train_x = dataSet
train_y = labels
test_x = dataSet
test_y = labels

## step 2: training...
print ("step 2: training...")
C = 0.001
toler = 0.001
maxIter = 500
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))

print("b: ", svmClassifier.b)
## step 3: testing
print ("step 3: testing...")
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)

## step 4: show the result
print ("step 4: show the result...")
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))
SVM.showSVM(svmClassifier)