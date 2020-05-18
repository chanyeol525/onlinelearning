

import random
import numpy as np
import tensorflow as tf
import math

# logistic regression
from alipy import ToolBox

from sklearn.utils import shuffle


class AC(object):

    def AC_(self, X, y):

       # X, y = shuffle(X, Y)
       # y = y.astype('int')
        alibox = ToolBox(X=X, y=y, query_type='AllLabels')

        alibox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

        model = alibox.get_default_model()

        # stopping_criterion = alibox.get_stopping_criterion('num_of_queries',50)

        model.fit(X, y)
        pred = model.predict(X)

        # 整理矩阵系数为信任度，返回start
        w = model.class_weight
        dim = w.shape[0]
        trustValue = []

        for i in range(0, dim):
            value = math.exp(w[i])  # exp() 方法返回x的指数,ex。
            trustValue.append(value)
        return trustValue
    # 整理矩阵系数为信任度，返回end

