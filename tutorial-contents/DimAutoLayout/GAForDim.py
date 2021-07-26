"""
Visualize Genetic Algorithm to find the shortest path for travel sales problem.

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import matplotlib.pyplot as plt
import numpy as np
import MakeAndPlotDim as ploter
import AutoDimUtil as util


N_DIMS = 6  # DNA size
DNA_SIZE = N_DIMS            # DNA (real number)
CROSS_RATE = 0.2
MUTATE_RATE = 1 / N_DIMS
POP_SIZE = 100
N_GENERATIONS = 200
DNA_BOUND = [1,  N_DIMS]       # solution upper and lower bounds

curGenIndex = 0
bstFitnessList = np.empty(N_GENERATIONS)
xAis = np.arange(N_GENERATIONS)

bstFitness = 0
bstDNA = ''

dt = util.DimUtil(DNA_SIZE)
MaxFitness = N_DIMS * DNA_BOUND[1] * np.max(dt.dimWeight)

# def MakeTarList():
#     keys = np.random.randint(1, 40, N_DIMS*2).reshape(N_DIMS, 2)
#     keys = np.sort(keys)
#     # 计算长度
#     lens = np.array([key[1] - key[0] for key in keys])
#     index = np.argsort(-lens)
#     keys = keys[index]
#     weight = np.exp(2 * np.array(range(1, len(keys)+1)))
#     lens = np.array([key[1] - key[0] for key in keys])

#     # 生成顺序
#     return keys, lens, weight


# TargePos, dimVal, dimWeight = MakeTarList()
# MaxFitness = N_DIMS * DNA_BOUND[1] * np.max(dimWeight)




# def isOverLap(dimA, dimB):
#     if(dimA[0] > dimB[0] and dimA[0] < dimB[1]):
#         return True
#     if(dimA[0] < dimB[0] and dimA[1] > dimB[0]):
#         return True
#     if(dimA[0] == dimB[0]):
#         return True
#     return False


# def getOverlapNum():
#     XOverlapNum = np.zeros(40)
#     for dimRandge in TargePos:
#         for index in range(dimRandge[0], dimRandge[1]):
#             XOverlapNum[index] += 1
#     return np.max(XOverlapNum)


# NUM_OVERLAP = getOverlapNum()
# VALID_DIM_RANGE = NUM_OVERLAP
# # DNA_BOUND[1] = NUM_OVERLAP

# def GetFitness(lens):
#     return MaxFitness - np.array(lens)


# # 获取所有样本的长度


# def GetLen(yList):
#     # 样本所有点到（0,0）的距离
#     sum = []
#     for y in yList:
#         # for xy,val in zip(xys,dimVal):

#         lenList = y * dimWeight
#         len = np.sum(lenList)
#         sum.append(len)
#     return sum

# # 计算DNA内最近点的距离


# def getMinDisToOther(DNAS):
#     # 有重叠的标注，高度一样则非法，mindis = 0.01
#     sum = []
#     for DNA in DNAS:
#         minDis = 1
#         # 计算所有点的距离，取最小值
#         for i in range(N_DIMS):
#             for j in range(i + 1, N_DIMS):
#                 if(isOverLap(TargePos[i], TargePos[j])):
#                     if(DNA[i] == DNA[j]):
#                         # 范围重叠时，判断高度是否一致
#                         minDis *= 0.5
#         sum.append(min(minDis, 1))
#     return sum


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.vstack([np.random.permutation(np.arange(1,N_DIMS + 1)) for _ in range(pop_size)])
        # self.pop = np.vstack([np.random.randint(1, VALID_DIM_RANGE, N_DIMS) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):     # get cities' coord in order
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        # if np.random.rand() < self.cross_rate:
        #     i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
        #     cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
        #     keep_city = parent[~cross_points]                                       # find the city number
        #     swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
        #     parent[:] = np.concatenate((keep_city, swap_city))
        # return parent

        if np.random.rand() < CROSS_RATE:
            i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
            cross_points = np.random.randint(0, 2, size=self.DNA_size).astype(np.bool)   # choose crossover points
            parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.randint(DNA_BOUND[0], DNA_BOUND[1])
        return child
        # for point in range(self.DNA_size):
        #     if np.random.rand() < self.mutate_rate:
        #         swap_point = np.random.randint(0, self.DNA_size)
        #         swapA, swapB = child[point], child[swap_point]
        #         child[point], child[swap_point] = swapB, swapA
        # return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


ga = GA(DNA_size=N_DIMS, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
bstDNA = ga.pop[0]

# env = TravelSalesPerson(N_DIMS)
for generation in range(N_GENERATIONS):
    # lx, ly = ga.translateDNA(ga.pop, env.city_position)
    # fitness, total_distance = ga.get_fitness(lx, ly)

    # 获取所有适应度
    lens = dt.GetLen(ga.pop)
    fitness = dt.GetFitness(MaxFitness, lens)      # calculate global fitness
    minDis = dt.getMinDisToOther(ga.pop)
    fitness = fitness * minDis

    # update fitness
    curBestFit = np.max(fitness)
    if(curBestFit > bstFitness):
        bstFitness = curBestFit
        best_idx = np.argmax(fitness)
        bstDNA = ga.pop[best_idx]

    bstFitnessList[curGenIndex] = bstFitness
    print('Gen:', generation, '| best fit: %.2f' % bstFitness,)

    curGenIndex += 1
    # curBstDNA = np.concatenate(
    #     (TargePos, np.array([bstDNA]).T), axis=1)
    # ploter.plotDNA(curBstDNA)
    # ploter.plotFitness(xAis[:curGenIndex],
    #                 bstFitnessList[:curGenIndex] - bstFitnessList[0])
    print('max fit', bstFitness)
    
    ga.evolve(fitness)
curBstDNA = np.concatenate(
    (dt.targetPos, np.array([bstDNA]).T), axis=1)
ploter.plotDNA(curBstDNA)
ploter.plotFitness(xAis[:curGenIndex],
                bstFitnessList[:curGenIndex] - bstFitnessList[0])
plt.pause(100)
plt.show()
plt.ioff()
