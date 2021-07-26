import numpy as np
import matplotlib.pyplot as plt
import MakeAndPlotDim as ploter
import AutoDimUtil as util

N_DIMS = 10  # DIM size
DNA_SIZE = N_DIMS            # DNA (real number)
DNA_BOUND = [1, N_DIMS + 1]       # solution upper and lower bounds
N_GENERATIONS = 40
POP_SIZE = 100           # population size
N_KID = 20               # n kids per generation

dt = util.DimUtil(DNA_SIZE)
bstFitness = []
xAis = np.arange(N_GENERATIONS)

# TargePos, dimVal= dt.MakeTarList()
MaxFitness = N_DIMS * DNA_BOUND[1] * np.max(dt.dimWeight)
# 生小孩
def make_kid(pop, n_kid):
    # generate empty kid holder
    kids = {'DNA': np.empty((n_kid, DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        # crossover (roughly half p1 and half p2)
        # 选父母
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)
        # 交叉点
        cp = np.random.randint(
            0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
        # 分别选择父母的部分DNA
        # kv = pop['DNA'][p1]
        # ks = pop['mut_strength'][p1]

        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        # 分别选择父母的变异强度
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]

        # 正态分布标准差
        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(
            ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
        # 正态分布
        # kv += ks * np.random.rand(*kv.shape)
        tmp = np.ceil(ks * np.random.rand(*kv.shape) - DNA_BOUND[1]/2)
        kv += tmp
        # kv += (np.random.binomial(n=DNA_BOUND[1],
        #        p=0.5, size = N_DIMS) - np.ceil(DNA_BOUND[1]/2))

        if(np.min(kv) < 0):
            a = 1
        # muteIdx= np.random.randint(0,N_DIMS-1)
        # kv[muteIdx]=np.random.randint(0,DNA_BOUND)
        # 限制范围
        kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value
        # kv[:] = np.ceil(kv)
        if(np.min(kv) < 0):
            a = 1
    return kids

# 移除不好样本


def kill_bad(pop, kids):
    # 新老合并
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    # 获取所有适应度
    lens = dt.GetLen(pop['DNA'])
    fitness = dt.GetFitness(MaxFitness, lens)      # calculate global fitness
    minDis = dt.getMinDisToOther(pop['DNA'])

    fitness = fitness * minDis
    bestFit = np.max(fitness)
    print('max fit', bestFit)

    bstFitness.append(bestFit)
    
    idx = np.arange(pop['DNA'].shape[0])
    # 递增排列，取后POP_SIZE位
    # selected by fitness ranking (not value)
    good_idx = idx[np.argsort(fitness)][-POP_SIZE:]
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop


class SmartDim(object):
    def __init__(self):
        # self.pop = dict(DNA=np.random.randint(1, VALID_DIM_RANGE, N_DIMS * POP_SIZE).reshape(POP_SIZE, N_DIMS),
        #                 mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values
        # self.pop = dict(DNA=np.array([np.random.permutation(np.arange(1,N_DIMS + 1)) for _ in range(POP_SIZE
        #                                                                                             )]),
        #                         mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values
        self.pop = dict(DNA=np.array([np.random.permutation(np.arange(1, N_DIMS + 1)) for _ in range(POP_SIZE
                                                                                                     )]),
                        mut_strength=np.random.randint(1, DNA_SIZE + 1, POP_SIZE * DNA_SIZE).reshape(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values


def getEvoRst():
    sd = SmartDim()
    for i in range(N_GENERATIONS):
        kids = make_kid(sd.pop, N_KID)
        sd.pop = kill_bad(sd.pop, kids)
        print('min dis is ', np.min(dt.getMinDisToOther(sd.pop['DNA'])))
        # bstDNA = np.concatenate(
        #     (TargePos, np.array([sd.pop['DNA'][-1]]).T), axis=1)
        # ploter.plotDNA(bstDNA)
        # ploter.plotFitness(xAis[:curGenIndex], bstFitness[:curGenIndex] - bstFitness[0])
    bstDNA = np.concatenate(
        (dt.targetPos, np.array([sd.pop['DNA'][-1]]).T), axis=1)
    return bstDNA,bstFitness

def runLocal():
    sd = SmartDim()
    for i in range(N_GENERATIONS):
        kids = make_kid(sd.pop, N_KID)
        sd.pop = kill_bad(sd.pop, kids)
        print('min dis is ', np.min(dt.getMinDisToOther(sd.pop['DNA'])))
        # bstDNA = np.concatenate(
        #     (TargePos, np.array([sd.pop['DNA'][-1]]).T), axis=1)
        # ploter.plotDNA(bstDNA)
        # ploter.plotFitness(xAis[:curGenIndex], bstFitness[:curGenIndex] - bstFitness[0])
    bstDNA = np.concatenate(
        (dt.targetPos, np.array([sd.pop['DNA'][-1]]).T), axis=1)

    ploter.plotDNA(bstDNA)
    ploter.plotFitness(bstFitness - bstFitness[0])
    plt.pause(100)
    plt.show()
    plt.ioff()

runLocal()

# sd = SmartDim()
# plt.ion()
# # plotDNA(sd.pop['DNA'][-1])

# for i in range(N_GENERATIONS):
#     kids = make_kid(sd.pop, N_KID)
#     sd.pop = kill_bad(sd.pop, kids)
#     print('min dis is ', np.min(dt.getMinDisToOther(sd.pop['DNA'])))
#     # bstDNA = np.concatenate(
#     #     (TargePos, np.array([sd.pop['DNA'][-1]]).T), axis=1)
#     # ploter.plotDNA(bstDNA)
#     # ploter.plotFitness(xAis[:curGenIndex], bstFitness[:curGenIndex] - bstFitness[0])
# bstDNA = np.concatenate(
#     (dt.targetPos, np.array([sd.pop['DNA'][-1]]).T), axis=1)
# ploter.plotDNA(bstDNA)
# ploter.plotFitness(xAis,
#                    bstFitness - bstFitness[0])

# plt.pause(100)
# plt.show()
# plt.ioff()


