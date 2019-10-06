import numpy as np
import pandas as pd
import sys
import math
from scipy.fftpack import dct
def initialize_parameters(layer_dims):
    W = {}
    b = {}
    l = len(layer_dims)
    for i in range(1,l):
        W[i] = np.random.randn(layer_dims[i-1],layer_dims[i])*0.01
        b[i] = np.zeros((1,layer_dims[i]))
    return W,b
    
def initialize_parameters_better(layer_dims):
    W = {}
    b = {}
    l = len(layer_dims)
    for i in range(1,l):
        W[i] = np.random.randn(layer_dims[i-1],layer_dims[i])*(np.sqrt(2/layer_dims[i]))
        b[i] = np.zeros((1,layer_dims[i]))
    return W,b

def activation(mode,x):
    if mode == 'sigmoid':
        return 1/(1 + np.exp(-1 * x))
    if mode == 'tanh':
        return np.tanh(x)
    if mode == 'relu':
        return np.maximum(0,x)
    if mode == 'softplus':
        return np.log(1 + np.exp(x))
    if mode == 'linear':
        return x
    if mode == 'leakyRelu':
        return ((x>=0) * x) + ((x<0) *0.01*x)

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
    if mode == 'softplus':
        return activation('sigmoid',x)
    if mode == 'linear':
        return 1
    if mode == 'leakyRelu':
        return (x>0)*1 + (x<=0)*0.01 
    
    
def linear_backward(dZ,caches,lambd=0):
    (Z,A,W,b) = caches
    #print(A.T.shape)
    #print(dZ.shape)
    dW = (A.T @ dZ) + (lambd/dZ.shape[0])*W
    
    db = np.sum(dZ,axis=0)
    #print('v',dZ)
    #print('y',W.T.shape,dZ.shape)
    dA = dZ@(W.T)
    return dW,db,dA

def linear_activation(mode,dA,Z):
    dZ = dA * derivatives(mode,Z)
    return dZ

def backward(AL,Y,caches,modes,lambd = 0):
    n = Y.shape[0]
    dZ = ((AL - Y)*(1/n))
   # print('x',dZ.shape)
    dW = {}
    db = {}
    L = len(caches) +1
    for k in reversed(range(1,L)):
        Z,A,W,b = caches[k]
        #print(k,W.shape)
        dW[k],db[k],dA = linear_backward(dZ,caches[k],lambd)
        if k>1:
            dZ = linear_activation(modes[k],dA,Z)
    return dW,db
    

def update_parameters(W,b,dW,db,learning_rate):
    L = len(W) + 1
    for i in range(1,L):
        W[i] = W[i] - learning_rate*dW[i]
        b[i] = b[i] - learning_rate*db[i]
    return W,b

def gradientDescent(X,Y,modes,learning_rate,W,b,lambd=0):
    #W,b = initialize_parameters(layers_dims) 
#     for i in range(iterations):
    caches,AL = forward(X,W,b,modes)
    #print(AL)
    dW,db = backward(AL,Y,caches,modes,lambd)
    W,b = update_parameters(W,b,dW,db,learning_rate)
    return W,b
    

def miniBatch(train,test,weight):
    train = pd.read_csv(train,header=None)
    test = pd.read_csv(test,header=None)
    X_train = train.values[:,:].copy()
    X_train = X_train[:,0:X_train.shape[1]-1].copy()
    #X_train = X_train/255
    Mean1 = np.mean(X_train,axis=0,keepdims=True)
    Stddev = np.std(X_train,axis=0,keepdims=True)
    #X_train = (X_train-Mean1)/Stddev
    Y_train = train.values[:,X_train.shape[1]].copy()
    Y_ohe = np.zeros((Y_train.shape[0],10))
    for i in range(Y_train.shape[0]):
        Y_ohe[i,Y_train[i]] = 1
    X_test = test.values[:,:].copy()
    X_test = X_test[:,0:X_test.shape[1]-1].copy()
    #X_test = (X_test-Mean1)/Stddev
    #X_test = X_test/255
    layers_dims = [1024,512,512,256,10]
    modes = ['x','x','leakyRelu','leakyRelu','leakyRelu']
    W,b =  initialize_parameters(layers_dims)
    i=0
    k=0
    while i<4500:
        for j in range(X_train.shape[0]//500):
            X_train1 = X_train[j*int(500):(j+1)*int(500),:]
            Y_train1 = Y_ohe[j*int(500):(j+1)*int(500)]
            W,b = gradientDescent(X_train1,Y_train1,modes,0.1/math.sqrt((i+1)),W,b)
            print(W)
            i = i+1
            print(i)
            if i>=4500:
                break
        k = k+1
    print(W)
    x,AL = forward(X_test,W,b,modes)
    pred = np.argmax(AL,axis=1)
    print(pred)
    #for i in range(1,L):
    for param in pred:
        print(np.asscalar(param),file=open(weight, "a"))
if __name__ == '__main__':
    miniBatch(*sys.argv[1:])
            

