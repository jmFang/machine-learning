import numpy as np 
import matplotlib.pyplot as plt

def main():
    feature = [[0,0], [1,0], [0,1], [-1,0], [1,-1]]
    label = ['+', '+', '-', '-', '-']

    feature_mat = np.mat(feature)
    label_mat = np.mat(label).T
    print(feature_mat)
    print(label_mat)
    numSamples, numFeartures = np.shape(feature_mat)
    for i in range(numSamples):
        if label_mat[i,0] == '-':
            plt.plot(feature_mat[i,0], feature_mat[i,1],'or')
        elif label_mat[i,0] == '+':
            plt.plot(feature_mat[i,0], feature_mat[i,1],'ob')
    plt.show()
main()