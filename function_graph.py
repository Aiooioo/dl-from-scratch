import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from functions import *

jet_colormap = mpl.colors.Colormap('jet')


# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y = sin(x)

x1 = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x1)

y2 = step_function(x1)
y3 = relu(x1)
y4 = softmax(x1)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

def graph(fs, title='', xlabel='x', ylabel='y'):
    # 绘制图形
    for [f, x] in fs:
        plt.plot(x, f)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def graph3D():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)
    z = x**2 + y**2


    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_title('f(x0, x1)')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()    

def quiver():
    x = np.arange(-2, 2.5, 0.25)
    y = np.arange(-2, 2.5, 0.25)
    x, y = np.meshgrid(x, y)

    x = x.flatten()
    y = y.flatten()

    grad = numeric_gradient_batch(function_quadratic_sum, np.array([x, y]))
    figure, ax = plt.subplots()
    
    ax.quiver(x, y, -grad[0], -grad[1], angles='xy', color='#666666')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    ax.grid()
    ax.legend()
    ax.set_aspect(1.0)
    plt.show()

if __name__ == '__main__':
    # graph([[y, x], [y1, x1], [y2, x1], [y3, x1], [y4, x1]])
    quiver()