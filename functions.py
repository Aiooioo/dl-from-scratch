import numpy as np

def sin(x):
    return np.sin(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.array(x > 0, dtype=np.int8)

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 均方误差损失函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 交叉熵误差损失函数
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def numeric_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
