#######################################
# pyGPGO examples
# gif_gen: generates a gif (the one in paper.md) showing how the BO framework
# works on the Franke function, step by step.
#######################################

import numpy as np
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import glob
import imageio

import os
os.chdir("/Users/travis/Documents/Education/Barcelona GSE/Quarter 3/topics II/hrvoje/fashionMNIST-bayesian-optimization")

from CNN_test_model import test_model

def f(x, y):
    # Franke's function (https://www.mathworks.com/help/curvefit/franke.html)
    test_model(dropout = x, learning_rate = y)

def plotFranke():
    x = np.linspace(0, 1, num=1000)
    y = np.linspace(0, 1, num=1000)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title('Original function')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)


def plotPred(gpgo, num=100):
    X = np.linspace(0, 1, num=num)
    Y = np.linspace(0, 1, num=num)
    U = np.zeros((num**2, 2))
    i = 0
    for x in X:
        for y in Y:
            U[i, :] = [x, y]
            i += 1
    z = gpgo.GP.predict(U)[0]
    Z = z.reshape((num, num))
    X, Y = np.meshgrid(X, Y)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.set_title('Gaussian Process surrogate')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    best = gpgo.best
    ax.scatter([best[0]], [best[1]], s=40, marker='x', c='r', label='Sampled point')
    plt.legend(bbox_to_anchor=(1.3, 0.05))
    ax.set_xlabel('dropout')
    ax.set_ylabel('learning rate')
    #plt.show()
    return Z

cov = matern32() # other kernel types: sqdExp, matern, matern52, gammaExp, rationalQuadratic
gp = GaussianProcess(cov)
acq = Acquisition(mode='ExpectedImprovement')  # other modes: UCB, ProbabilityImprovement, loads more
param = {'dropout': ('cont', [0, 1]),
         'learning_rate': ('cont', [0.1,1])}

np.random.seed(1337)
gpgo = GPGO(gp, acq, test_model, param)
gpgo.run(max_iter=1)

gpgo.run(max_iter = 1, resume = True)

gpgo.history

dir(gpgo)

if __name__ == '__main__':
    n_iter = 10
    cov = matern32() # other kernel types: sqdExp, matern, matern52, gammaExp, rationalQuadratic
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')  # other modes: UCB, ProbabilityImprovement, loads more
    param = {'dropout': ('cont', [0, 1]),
             'learning_rate': ('cont', [0.1,1])}


    np.random.seed(1337)
    gpgo = GPGO(gp, acq, test_model, param)
    gpgo.run(max_iter=1)

    for i in range(n_iter):
        #fig = plt.figure(figsize=plt.figaspect(0.5))
        fig = plt.figure()
        fig.suptitle("CNN Hyperparameter Tuning (Iteration {})".format(i+1))
        gpgo.run(max_iter=1, resume=True)

        plotPred(gpgo)
        #plt.show()
        plt.savefig('img/gif/{}.png'.format(i), dpi=300)
        plt.close()

images = []
filenames = sorted(glob.glob('img/gif/*.png'))


for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('img/CNN_BayesianOptimization.gif', images, duration = 1)
