import numpy as np
import pandas as pd
import sys
import math
def initialize_parameters(layer_dims):
    W = {}
    b = {}
    l = len(layer_dims)
    for i in range(1,l):
        W[i] = np.zeros((layer_dims[i-1],layer_dims[i]))
        b[i] = np.zeros((1,layer_dims[i]))
    return W,b

def activation(mode,x):
    if mode == 'sigmoid':
        return 1/(1 + np.exp(-1 * x))
    if mode == 'tanh':
        return np.tanh(x)
    if mode == 'relu':
        return np.maximum(0,x)

def computeSoft(Z):
    #A1 = (A @ W) + b
    A1 = np.exp(Z)
    sum1 = np.sum(A1,axis=1).reshape(A1.shape[0],1)
    return A1/sum1
    
def forward(X,W,b,modes):
    L = len(W) + 1
    A = X
    caches = {}
    #caches[1] = (0,A,W[1],b[1])
    Z = np.zeros(1)
    for i in range(1,L):
        caches[i] = (Z,A,W[i],b[i])
       # print(i,W[i].shape)
       # print(A.shape,W[i].shape)
        if i<L-1:
            Z = A@W[i] + b[i]
            A = activation(modes[i+1],Z)
        elif i==L-1:
            Z = A@W[i] + b[i]
            A = computeSoft(Z)
            #print(A.shape)
    return caches,A

def derivatives(mode,x):
    if mode=='sigmoid':
        a = activation('sigmoid',x)
        return a*(1-a)
    if mode=='tanh':
        a = activation('tanh',x)
        return 1-(a**2)
    if mode == 'relu':
        return (x>0)*1
    
def linear_backward(dZ,caches):
    (Z,A,W,b) = caches
    #print(A.T.shape)
    #print(dZ.shape)
    dW = A.T @ dZ
    db = np.sum(dZ,axis=0)
    #print('v',dZ)
    #print('y',W.T.shape,dZ.shape)
    dA = dZ@(W.T)
    return dW,db,dA

def linear_activation(mode,dA,Z):
    dZ = dA * derivatives(mode,Z)
    return dZ

def backward(AL,Y,caches,modes):
    n = Y.shape[0]
    dZ = ((AL - Y)*(1/n))
   # print('x',dZ.shape)
    dW = {}
    db = {}
    L = len(caches) +1
    for k in reversed(range(1,L)):
        Z,A,W,b = caches[k]
        #print(k,W.shape)
        dW[k],db[k],dA = linear_backward(dZ,caches[k])
        if k>1:
            dZ = linear_activation(modes[k],dA,Z)
    return dW,db

def update_parameters(W,b,dW,db,learning_rate):
    L = len(W) + 1
    for i in range(1,L):
        W[i] = W[i] - learning_rate*dW[i]
        b[i] = b[i] - learning_rate*db[i]
    return W,b

def gradientDescent(X,Y,modes,learning_rate,W,b):
    #W,b = initialize_parameters(layers_dims) 
#     for i in range(iterations):
    caches,AL = forward(X,W,b,modes)
    dW,db = backward(AL,Y,caches,modes)
    W,b = update_parameters(W,b,dW,db,learning_rate)
    return W,b

def miniBatch(train,param,weight):
    train = pd.read_csv(train,header=None)
    X_train = train.values[:,:].copy()
    X_train = X_train[:,0:X_train.shape[1]-1].copy()
    Y_train = train.values[:,X_train.shape[1]].copy()
    Y_ohe = np.zeros((Y_train.shape[0],10))
    for i in range(Y_train.shape[0]):
        Y_ohe[i,Y_train[i]] = 1
    hparams = []
    hparams1=[] 
    f = open(param) 
    for word in f.read().split(','):
        hparams1.append((word))
    for word in hparams1:
        for w in word.split():
            hparams.append(float(w))
    print(hparams)
    layers_dims = []
    layers_dims.append(X_train.shape[1])
    for i in range(4,len(hparams)):
        layers_dims.append(int(hparams[i]))
    layers_dims.append(10)
    L = len(layers_dims)
    print(L)
    modes = ['x','x']
    for i in range(2,L):
        modes.append('sigmoid')
    W,b =  initialize_parameters(layers_dims)
    i=0
    k=0
    while i<hparams[2]:
        for j in range(X_train.shape[0]//int(hparams[3])):
            if hparams[0] == 1:
                alpha = hparams[1]
            else:
                alpha = hparams[1]/math.sqrt(i+1)
            X_train1 = X_train[j*int(hparams[3]):(j+1)*int(hparams[3]),:]
            Y_train1 = Y_ohe[j*int(hparams[3]):(j+1)*int(hparams[3])]
            W,b = gradientDescent(X_train1,Y_train1,modes,alpha,W,b)
            i = i+1
            if i>=hparams[2]:
                break
        k = k+1
    print(W)
    for i in range(1,L):
        b[i] = b[i].reshape(b[i].shape[1]*b[i].shape[0],1)
        for param in b[i]:
            print(np.asscalar(param),file=open(weight, "a"))
        W[i] = W[i].reshape(W[i].shape[1]*W[i].shape[0],1)
        for param in W[i]:
            print(np.asscalar(param),file=open(weight, "a"))
if __name__ == '__main__':
    miniBatch(*sys.argv[1:])
            
