import numpy as np

E = np.random.randn(10, 4)

def embed(indices):
    return(E[indices])

one_hot = np.zeros(10)
one_hot[5] = 1
print(one_hot)
print(one_hot @ E)
print(E[5])