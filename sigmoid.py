# Use the numpy library
import numpy as np

# Activation (sigmoid) function
def sigmoid(x):
    pass
    return np.sum(np.divide(1, (1 + np.exp(-x))))

def testWeights(w1, w2, b):
    return sigmoid(w1*0.4 + w2*0.6 + b)

print(testWeights(2, 6, -2))
print(testWeights(3, 5, -2.2))
print(testWeights(5, 4, -3))