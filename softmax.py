import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    L = np.true_divide(np.exp(L), np.sum(np.exp(L)))
    return L

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
# Trying for Y=[1,0,1,1] and P=[0.4,0.6,0.1,0.5].
def cross_entropy(Y, P):
    print(P)
    print(Y)
    pass

# print(softmax([1,2,3,4,5]))
# a = np.arange(25).reshape(5,5)
# a = [1,2,3,4,5]

Y=[1,0,1,1]
P=[0.4,0.6,0.1,0.5]
print('Trying for Y=[1,0,1,1] and P=[0.4,0.6,0.1,0.5].')
YP = np.subtract(np.einsum('i,j->i', Y, np.log(P)), np.einsum('i,j->i', np.subtract(1, Y), np.log(P)))
# YP1 = np.einsum('i,j->i', np.subtract(1, Y), np.log(P))
print((YP))
print(np.sum(YP))