import numpy as np
#Module activations.py

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

def sigmoid(Z, derivative=False):
    S = 1/(1+np.exp(-Z))
    return np.multiply(S, (1-S)) if derivative else S

def crossentropy(Y_prim, Y):
    return -np.sum(np.sum(Y * np.log(Y_prim), axis=1))/Y.shape[0]