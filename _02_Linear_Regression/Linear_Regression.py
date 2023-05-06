# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
from sklearn.linear_model import Ridge

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(X,Y,data):
    
    ridge_reg=Ridge(alpha=0.4, solver='sag')
    ridge_reg.fit(X,Y)
    return ridge_reg.predict(data)
    
def lasso(X,Y,data):
    lasso_reg=Lasso(alpha=0.01, max_iter=30000)
    lasso_reg.fit(X,Y)
    return lasso_reg.predict(data)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

def main():
    x,y=read_data(path='./data/exp02/')
    data=np.array([1,2,3,4,5,6])
    print (ridge(x,y,data))
    print (lasso(x,y,data))
