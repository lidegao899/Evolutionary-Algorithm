import numpy as np
import matplotlib.pyplot as plt

N_DIMS = 2  # DIM size
DNA_SIZE = N_DIMS * 2             # DNA (real number)
DNA_BOUND = [0, 40]       # solution upper and lower bounds
N_GENERATIONS = 300
POP_SIZE = 2           # population size
N_KID = 60               # n kids per generation

TargePos = np.array([[-2, 20], [-2, 20]])
def MakeTarList():
    xl = TargePos[..., 0].repeat(N_DIMS/2)
    yl = TargePos[..., 1].repeat(N_DIMS/2)
    return xl, yl


txl, tyl = MakeTarList()

tmp = np.array([np.random.rand(1, N_DIMS),[tyl]]).reshape(1,DNA_SIZE).repeat(POP_SIZE, axis=0)
print(tmp)

tmpvar =np.random.rand(1,N_DIMS * POP_SIZE).reshape(POP_SIZE,N_DIMS)
print(tmpvar)
djska= tyl.repeat(POP_SIZE, axis=0).reshape(POP_SIZE,N_DIMS)
endvar = np.concatenate((tmpvar, tyl.repeat(POP_SIZE, axis=0).reshape(POP_SIZE,N_DIMS)), axis=1)
print(endvar)
# noRepeat = np.array([np.random.rand(1, N_DIMS) for i in range(POP_SIZE)])
# print(noRepeat)

# # 数组元素一维拼接
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6]])
# # print(np.concatenate((a, b), axis=0))


# # 数组二维拼接
# print(np.concatenate((a, b.repeat(2, axis=0)), axis=1))

# print(np.concatenate((a, b), axis=None))
x = np.array([3, 1, 2])
print(np.argsort(x))