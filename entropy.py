import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    R = np.multiply(Y, np.log(P)) + np.multiply(np.subtract(1,Y), np.log(np.subtract(1,P)))
    pass
    return np.sum(-R)

P = [0.4, 0.6, 0.1, 0.5]
Y = [1, 0, 1, 1]
print(cross_entropy(Y, P))