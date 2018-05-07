from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import csv

def loaddatafromCSV(filename):
    feature_data =[]; label_data=[]
    with open(filename) as f:
        reader = csv.reader(f)
        for index,row in enumerate(reader):
            feature_tmp = []
            for i in range(len(row)-2):
                feature_tmp.append(float(row[i+1]))
            label_data.append(int(row[-1]))
            feature_data.append(feature_tmp)
    return feature_data, label_data

def loadtestdatafromCSV(filename):
    feature_data =[]
    with open(filename) as f:
        reader = csv.reader(f)
        for index,row in enumerate(reader):
            feature_tmp = []
            for i in range(len(row)-1):
                feature_tmp.append(float(row[i+1]))
            feature_data.append(feature_tmp)
    return feature_data

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



file_train = 'train.csv'
file_test = 'test_raw.csv'
file_out = 'sampleSubmission.csv'

X, Y = loaddatafromCSV(file_train)
test_data = loadtestdatafromCSV(file_test)
clf = RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(X, Y)

p = clf.predict(test_data)
save_result(file_out,np.mat(p).T)
