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


def cross_entropy_error_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numeric_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        temp_val = x[idx]

        x[idx] = float(temp_val) + h
        fxh1 = f(x)

        x[idx] = float(temp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = temp_val

        it.iternext()

    return grad

def numeric_gradient_batch(f, x):
    if x.ndim == 1:
        return numeric_gradient(f, x)
    else:
        grad = np.zeros_like(x)
        for idx, x in enumerate(x):
            grad[idx] = numeric_gradient(f, x)
        
        return grad


def function_quadratic_sum(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)