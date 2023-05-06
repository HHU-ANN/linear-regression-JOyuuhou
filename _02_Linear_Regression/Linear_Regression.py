# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y=read_data(path='./data/exp02')
    loss = np.dot(np.linalg.inv(np.dot(x.T,x)+0.5*np.identity(6,dtype=int)),np.dot(x.T,y))
    return data @ loss
    
def lasso(data):
    x,y=read_data(path='./data/exp02')
    loss = np.dot(np.linalg.inv(np.dot(x.T,x)),np.dot(x.T,y)+0.5*np.ones(6))
    return loss @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

def main():
    x,y=read_data(path='./data/exp02/')
    data=np.array([1,2,3,4,5,6])
    print (ridge(data))
    print (lasso(data))
