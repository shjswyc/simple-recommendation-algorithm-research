# 相似度算法和其他一些封装的函数
from Package import *


# 1.余弦相似度
def Cosine(x, y):
    sum_xy = 0.0
    normX = 0.0
    normY = 0.0
    for a, b in zip(x, y):
        sum_xy += a * b
        normX += a ** 2
        normY += b ** 2
    if normX == 0.0 or normY == 0.0:
        return 0
    else:
        return sum_xy / ((normX * normY) ** 0.5)


# 2.皮尔逊相关系数
def Pearson(x, y):
    sum_XY = 0.0
    sum_X = 0.0
    sum_Y = 0.0
    normX = 0.0
    normY = 0.0
    count = 0
    for a, b in zip(x, y):
        count += 1
        sum_XY += a * b
        sum_X += a
        sum_Y += b
        normX += a ** 2
        normY += b ** 2
    if count == 0:
        return 0
    # denominator part
    denominator = (normX - sum_X ** 2 / count) ** 0.5 * (normY - sum_Y ** 2 / count) ** 0.5
    if denominator == 0:
        return 0
    return (sum_XY - (sum_X * sum_Y) / count) / denominator


# 3.相关系数
def Jaccard(x, y):
    M = 0
    N = 0
    for a, b in zip(x, y):
        if a == b:
            if a == 1:
                M += 1
        else:
            N += 1
    if M + N == 0:
        return 0
    else:
        return M / (M + N)


# 4.信息增益
def calc_entropy(x):
    """
    计算信息熵
    """
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def calc_condition_entropy(x, y):
    """
    计算条件信息熵
    """
    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_entropy(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent
    return ent


def Calc_entropy_grap(x, y):
    """
    信息增益
    """
    base_ent = calc_entropy(y)
    condition_ent = calc_condition_entropy(x, y)
    ent_grap = base_ent - condition_ent
    return ent_grap


# 5.欧式距离
def Euclidean(x, y):
    d = 0
    for a, b in zip(x, y):
        d += (a - b) ** 2
    return d ** 0.5


# 6.曼哈顿距离
def Manhattann(a, b):
    distance = 0
    for i in range(len(a)):
        distance += np.abs(a[i] - b[i])
    return distance


# 7.切比雪夫距离
def Chebyshevn(a, b):
    distance = 0
    for i in range(len(a)):
        if (abs(a[i] - b[i]) > distance):
            distance = abs(a[i] - b[i])
    return distance


# 8.汉明距离
def Hamming(a, b):
    sumnum = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            sumnum += 1
    return sumnum


# 最大最小归一化
def MaxMinNormalization(array):
    l = len(array)
    if l != 0:
        max = np.max(array)
        min = np.min(array)
        for j in range(0, l):
            array[j] = MaxMinCalc(array[j], min, max)
    return array


# 最大最小归一化
def MaxMinCalc(x, min, max):
    if max == min:
        return x
    else:
        x = (x - min) / (max - min)
        return x


def calc_rmse(test, pred):
    MSE = np.sum(np.power((test - pred), 2)) / len(test)
    RMSE = np.sqrt(MSE)
    return RMSE
