import numpy as np
import matplotlib.pyplot as plt
import MakeAndPlotDim as ploter

N_DIMS = 8  # DIM size
DNA_SIZE = N_DIMS            # DNA (real number)
DNA_BOUND = [1, N_DIMS + 1]       # solution upper and lower bounds
N_GENERATIONS = 300
POP_SIZE = 200           # population size
N_KID = 80               # n kids per generation

curGenIndex = 0
bstFitness = np.empty(N_GENERATIONS)
xAis = np.arange(N_GENERATIONS)


def MakeTarList():
    keys = np.random.randint(1, 40, N_DIMS*2).reshape(N_DIMS, 2)
    keys = np.sort(keys)
    # 计算长度
    lens = np.array([key[1] - key[0] for key in keys])
    index = np.argsort(-lens)
    keys = keys[index]
    weight = np.exp(2 * np.array(range(1, len(keys)+1)))
    lens = np.array([key[1] - key[0] for key in keys])

    # 生成顺序
    return keys, lens, weight


TargePos, dimVal, dimWeight = MakeTarList()
MaxFitness = N_DIMS * DNA_BOUND[1] * np.max(dimWeight)


def isOverLap(dimA, dimB):
    if(dimA[0] > dimB[0] and dimA[0] < dimB[1]):
        return True
    if(dimA[0] < dimB[0] and dimA[1] > dimB[0]):
        return True
    if(dimA[0] == dimB[0]):
        return True
    return False


def getOverlapNum():
    XOverlapNum = np.zeros(40)
    for dimRandge in TargePos:
        for index in range(dimRandge[0], dimRandge[1]):
            XOverlapNum[index] += 1
    return np.max(XOverlapNum)


NUM_OVERLAP = getOverlapNum()
VALID_DIM_RANGE = NUM_OVERLAP
DNA_BOUND[1] = NUM_OVERLAP

def GetFitness(lens):
    return MaxFitness - np.array(lens)


# 获取所有样本的长度


def GetLen(yList):
    # 样本所有点到（0,0）的距离
    sum = []
    for y in yList:
        # for xy,val in zip(xys,dimVal):

        lenList = y * dimWeight
        len = np.sum(lenList)
        sum.append(len)
    return sum

# 计算DNA内最近点的距离


def getMinDisToOther(DNAS):
    # 有重叠的标注，高度一样则非法，mindis = 0.01
    sum = []
    for DNA in DNAS:
        minDis = 1
        # 计算所有点的距离，取最小值
        for i in range(N_DIMS):
            for j in range(i + 1, N_DIMS):
                if(isOverLap(TargePos[i], TargePos[j])):
                    if(DNA[i] == DNA[j]):
                        # 范围重叠时，判断高度是否一致
                        minDis *= 0.5
        sum.append(min(minDis, 1))
    return sum


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
        kv += (np.random.binomial(n=VALID_DIM_RANGE,
               p=0.5, size = N_DIMS) - np.ceil(VALID_DIM_RANGE/2))

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
    lens = GetLen(pop['DNA'])
    fitness = GetFitness(lens)      # calculate global fitness
    minDis1 = getMinDisToOther(pop['DNA'])

    # 新老合并
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    # 获取所有适应度
    lens = GetLen(pop['DNA'])
    fitness = GetFitness(lens)      # calculate global fitness
    minDis = getMinDisToOther(pop['DNA'])

    fitness = fitness * minDis
    bestFit = np.max(fitness)
    print('max fit', bestFit)

    bstFitness[curGenIndex] = bestFit
    if(curGenIndex == 0):
        bstFitness[:] = bstFitness[0]

    idx = np.arange(pop['DNA'].shape[0])
    # 递增排列，取后POP_SIZE位
    # selected by fitness ranking (not value)
    good_idx = idx[np.argsort(fitness)][-POP_SIZE:]
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop


class SmartDim(object):
    def __init__(self):
        self.pop = dict(DNA=np.random.randint(1, VALID_DIM_RANGE, N_DIMS * POP_SIZE).reshape(POP_SIZE, N_DIMS),
                        mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values


sd = SmartDim()
plt.ion()
# plotDNA(sd.pop['DNA'][-1])

for i in range(N_GENERATIONS):
    kids = make_kid(sd.pop, N_KID)
    sd.pop = kill_bad(sd.pop, kids)
    print('min dis is ', np.min(getMinDisToOther(sd.pop['DNA'])))
    curGenIndex += 1
    bstDNA = np.concatenate(
        (TargePos, np.array([sd.pop['DNA'][-1]]).T), axis=1)
    # ploter.plotDNA(bstDNA)
    # ploter.plotFitness(xAis[:curGenIndex], bstFitness[:curGenIndex] - bstFitness[0])

ploter.plotDNA(bstDNA)
ploter.plotFitness(xAis[:curGenIndex],
                   bstFitness[:curGenIndex] - bstFitness[0])

plt.pause(100)
plt.show()
plt.ioff()
