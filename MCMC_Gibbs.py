import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
def add_gaussian_noise(im,prop,varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im).astype('float')
    im2[index] += e[index]
    return im2

def add_saltnpeppar_noise(im,prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2

def neighbours(i,j,M,N,size=4):
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]

        return n

    if size==8:
        print('Not yet implemented\n')
    return -1

def L(x, y):
    if y == 0:
        y = -1
    return x * y

def E(w, x, i, j):
    sum = 0
    for neighbour in neighbours(i, j, x.shape[0], x.shape[1]):
        sum += w[i][j] * x[i][j] * x[neighbour[0]][neighbour[1]]

def MCMC_Gibbs(x, y, w = np.ones(1), iteration = 10):
    xt = np.copy(x)
    w = np.ones(x.shape)


    for tau in range (0, iteration):
        for i in range (xt.shape[0]):
            for j in range (xt.shape[1]):
                xt[i][j] = 1
                nominator = np.exp(L(xt[i][j], y[i][j])) * np.exp(E(w, xt, i, j))

                denom_left = np.exp(L(xt[i][j], y[i][j])) * np.exp(E(w, xt, i, j))
                xt[i][j] = -1
                denom_right = np.exp(L(xt[i][j], y[i][j])) * np.exp(E(w, xt, i, j))
                denominator = denom_left + denom_right
                p_i = nominator / denominator
                xt = np.random.uniform(0,1)
                if p_i > t:
                    xt[i][j] = 1
                else:
                    xt[i][j] = -1
    return(xt)

# proportion of pixels to alter
prop = 0.7
varSigma = 0.1
im = imread('resized_gray_pug.png')
im = im/255
fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(im,cmap='gray')
im2 = add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(132)
ax2.imshow(im2,cmap='gray')
im2 = add_saltnpeppar_noise(im,prop)
ax3 = fig.add_subplot(133)
ax3.imshow(im2,cmap='gray')

y = np.random.choice((-1, 1), im2.shape)

MCMC_Gibbs(im2, y)
