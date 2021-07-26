import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

N_DIMS = 8  # DIM size
DNA_SIZE = N_DIMS            # DNA (real number)
DNA_BOUND = [0, N_DIMS + 1]       # solution upper and lower bounds
N_GENERATIONS = 800
POP_SIZE = 100           # population size
N_KID = 20               # n kids per generation

res= sum(np.random.binomial(9, 0., 20000) == 0)/20000
print(res)
print(np.random.binomial(9, 0., 5))

n1 = 6
num = 10000
bi = np.random.binomial(n=n1, p=0.51, size=num)
n = np.random.normal(n1*0.5, sqrt(n1*0.5*0.5), size=num)

mut_strength=np.random.rand(POP_SIZE, DNA_SIZE)

plt.hist(bi, bins=20);
plt.hist(n, alpha=0.5, bins=20);
plt.show();