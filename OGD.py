

import random
import numpy as np
import tensorflow as tf
import math

# logistic regression
class LR(object):

    @staticmethod
    def fn(w, x):
        ''' sigmod function '''
        return 1.0 / (1.0 + np.exp(-w.dot(x)))

    @staticmethod
    def loss(y, y_hat):
        '''cross-entropy loss function'''
        return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1-y)*np.log(1-y_hat)))

    @staticmethod
    def grad(y, y_hat, x):
        '''gradient function'''
        return (y_hat - y) * x

class OGD(object):

    def __init__(self,dim,alpha,decisionFunc=LR):
        self.alpha = alpha
        self.w = np.zeros(dim)
        self.dim = dim
        self.decisionFunc = decisionFunc

    def predict(self, x):
        return self.decisionFunc.fn(self.w, x)

    def update(self, x, y,step):
        y_hat = self.predict(x)
        g = self.decisionFunc.grad(y, y_hat, x)
        learning_rate = self.alpha / np.sqrt(step + 1)  # damping step size
        # SGD Update rule theta = theta - learning_rate * gradient
        self.w = self.w - learning_rate * g
        return self.decisionFunc.loss(y,y_hat)

    def training(self, X,Y, max_itr=10000):
        n = 0
        while True:
            for x, y in zip(X, Y):
                loss = self.update(x, y,n)
                n += 1
                if n > max_itr:
                    return

    def OGD_(self, X, Y):
        m,n=np.shape(X)
        ogd=OGD(n,alpha=0.05)
        ogd.training(X, Y,  max_itr=10000)
        w = ogd.w
        # 整理矩阵系数为信任度，返回start
        dim = w.shape[0]
        trustValue = []

        for i in range(0, dim):
            value = math.exp(w[i])  # exp() 方法返回x的指数,ex。
            trustValue.append(value)
        return trustValue
    # 整理矩阵系数为信任度，返回end

