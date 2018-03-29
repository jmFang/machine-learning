import numpy as np
import matplotlib.pyplot as plt 
def least_square(feature, label):
    ''' least square (normal function)
    input: feature (mat) : features
           label (mat) : labels
    ouput: w (mat) regression coefficient
    '''
    w = (feature.T * feature).I * feature.T * label
    for i in range(len(w)):
        w[i,0] = round(w[i,0],2)
    return w

def ridge_regression(feature, label, lam): 
    '''
    input: feature (mat)
           label (mat)
           lam (mat) balance coefficient
    '''
    n = np.shape(feature)[1]
    w = (feature.T * feature + lam * np.mat(np.eye(n))).I * feature.T * label
    for i in range(len(w)):
        w[i,0] = round(w[i,0],2)
    return w

def load_data(file_name):
    '''
    input: file_name (string) 
    output: feature(mat)
            label (mat)
    '''
    f = open(file_name)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    f.close() 
    print(np.mat(feature))
    print(np.mat(label).T)
    feature_mat,_,_ = featureNormal(np.mat(feature))
    label_mat,_,_ = featureNormal(np.mat(label).T)
    print("--------normalized-------")
    # feature_norm, xx, xx1 = featureNormal(feature_mat)
    # label_norm, xx2,xx3 = featureNormal(label_mat)
    # return feature_norm, label_norm
    return feature_mat, label_mat
    # return np.mat(feature), np.mat(label).T
def getMean(X):
    sum = 0
    n = X.shape[0]
    print(X)
    for i in range(n):
        sum += int(X[i,0])
    return sum/n

def featureNormal(X):
    X_normal = X
    n = X.shape[1]
    mu = np.zeros((1, n)) 
    sigma = np.zeros((1,n)) 
    for i in range(n):
        mu[0,i] = np.mean(X[:,i])
        max = np.max(X[:,i])
        min = np.min(X[:,i])
        sigma[0, i] = max - min
        for j in range(X.shape[0]):
            if sigma[0,i] != 0:
                X_normal[j,i] = round(X_normal[j,i] - mu[0,i] /sigma[0,i],2)
    # X_normal = (X - mu) / sigma
    if n > 1:
        for j in range(X.shape[0]):
            X_normal[j:0] = 1
    return X_normal, mu, sigma

def train_GD(feature, label, maxCycle, alpha):
    ''' gradient desent algorithm to find coefficients
        input: feature (mat)
               label (mat)
               maxCycle (int) times of iteration
               alpha (float) learning rate
    '''
    # number of colume (number of features)
    n = np.shape(feature)[1]
    m = label.shape[0]
    convergence_rate = 0.001
    # initialize the coefficients vector
    theta = np.mat(np.ones((n,1)))
    print(theta)
    cost_history = np.zeros((maxCycle+2,1))
    cost_history[0] = cost(feature, label, theta)
    # print("J(theta_0) = " + str(cost_history[0]))
    for iter in range(maxCycle):
        theta = theta - (alpha/m)*(feature.T.dot(feature.dot(theta) - label))
        cost_history[iter+1] = cost(feature, label, theta)
        # print("J(theta_"+str(iter+1) + ") = " + str(cost_history[iter+1]))
        # declare convergence if J(theta) decreases by less than 10^-3
        if (iter > 0) and (abs(cost_history[iter-1,0] - cost_history[iter,0]) < convergence_rate):
            print("at No. "+ str(iter)+ " iterate, convergenced with: " + str(cost_history[iter,0]))
    return theta, cost_history

def cost(feature, label, w):
    '''compute error
        input: feature (mat)
               label (mat)
               w (mat)
        output: cost function value
    '''
    m = label.shape[0]
    err = feature.dot(w) - label
    res = (err.T.dot(err))/2*m
    return res

def predict(feature, theta):
    # print(feature)
    return feature * theta

def save_model(file_name, w):
    ''' save final model
        input: file_name (string)
               w (mat)
    '''
    m,n = np.shape(w)
    f_w = open(file_name, "w")
    w_array = []
    for i in range(m):
        w_tmp = []
        for j in range(n):
            w_tmp.append(str(round(w[i,j],2)))
        f_w.write("\t".join(w_tmp)+"\n")
    f_w.close()

def computeCost(feature,labels, theta):
    label_tmp = feature * theta
    err = (labels - label_tmp).T * (labels - label_tmp)

    plt.plot(labels, label='orignal', color='green')
    plt.plot(label_tmp, label='model',color='red')
    plt.legend()  
    plt.show()
    return np.sum(err)

def main():
    test_feature = [1,88,73,87,92]
    lam = 1
    learning_rate = 1
    iterNum = 100
    print("----------------1.load data----------------")
    feature, label = load_data("data.txt")
    print(feature)
    print(label)
    print("----------------2.train model--------------") 
    # print("---------------gradient desent-------------")
    # theta, cost = train_GD(feature, label, iterNum, learning_rate)
    # print(theta)

    # print("---------------normal equation-------------")
    # theta = least_square(feature, label)
    # print(theta)
    print("---------------ridge_regression--------------")
    theta = ridge_regression(feature, label, lam)
    print(theta)
    print("---------------compute cost----------------")
    costSum = computeCost(feature,label, theta)
    print("cost value :" + str(costSum))
    print("-----------------3.save model--------------")
    save_model("coefficients.txt", theta)
    print("---------------predict----------------------")
    pre = np.mat(test_feature)
    print("features for predicttion: " + str(pre))

    res = predict(pre, theta)
    print("predicttion result: " + str(res))
    print("-------------4.plot figure--------------")
    # showLogRegression(w, feature, label)
    # plt.plot(cost)
    # plt.show()

main()  