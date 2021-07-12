import numpy as np
import matplotlib.pyplot as plt

N_DIMS = 5  # DNA size
DNA_SIZE = 1             # DNA (real number)
DNA_BOUND = [0, 5]       # solution upper and lower bounds
N_GENERATIONS = 200
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation


def MakePnt():
   return np.random.rand(N_DIMS, 2)


def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function

def FXY(xy):
   return F(xy[:0]) + F(xy[:1])

class SmartDim(object):
   def __init__(self):
      self.pop = dict(DNA=5 * np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),   # initialize the pop DNA values
           mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values

   def Myplotting(self):
      plt.cla()
      plt.scatter(self.city_pos[:, 0].T, self.city_pos[:, 1].T, s=100, c='k')
      plt.xlim((-0.1, 1.1))
      plt.ylim((-0.1, 1.1))
      plt.pause(0.01)

sd =SmartDim()