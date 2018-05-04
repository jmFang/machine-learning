import numpy as np
import matplotlib.pyplot as plt

def threshold():
    score = [0.73, 0.69, 0.67, 0.55, 0.47, 0.45, 0.44, 0.35, 0.15, 0.08]
    label = ['+', '+', '-', '-', '+', '+', '-', '-', '-', '-']
    p = 4;n = 6
    t = 1
    neg = []
    sens = [];spe = []
    max = 0; index = 1
    while t > 0:
        pos = 0;count = 0
        for i in range(len(score)):
            if score[i] > t:
                pos = pos + 1
                if label[i] == '+':
                    count = count + 1
        tmp = round(count/p, 3)
        sens.append(tmp)
        spe.append(round((pos- count)/n, 3))
        if tmp > max:
            index = t
            max = tmp
        t = t - 0.01

    print('t:', index)
    plt.plot(spe, sens, color='r',markerfacecolor='blue',marker='o')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    for a, b in zip(spe, sens):
        plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
    plt.show()  
        
threshold()

            


