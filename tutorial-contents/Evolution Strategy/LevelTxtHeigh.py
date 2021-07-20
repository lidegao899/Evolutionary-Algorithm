import numpy as np
import matplotlib.pyplot as plt
import MakeAndPlotDim as ploter

N_DIMS = 4  # DIM size
DNA_SIZE = N_DIMS            # DNA (real number)
DNA_BOUND = [0, 20]       # solution upper and lower bounds
N_GENERATIONS = 200
POP_SIZE = 2           # population size
N_KID = 1               # n kids per generation

curGenIndex = 0
TargePos = [[1, 6], [4, 9], [1, 15], [12, 37]]

bstFitness = np.empty(N_GENERATIONS)
xAis = np.arange(N_GENERATIONS)

def MackDic():
    keys = []
    values = []


def isOverLap(dimA, dimB):
    if(dimA[0] > dimB[0] and dimA[0] < dimB[1]):
        return True
    if(dimA[0] < dimB[0] and dimA[1] > dimB[0]):
        return True
    if(dimA[0] == dimB[0]):
        return True
    return False


def MakeTarList():
    # xl = TargePos[..., 0].repeat(N_DIMS/2)
    # yl = TargePos[..., 1].repeat(N_DIMS/2)

    dimVal = [5, 5, 14, 25]
    dimWeight = [4, 3, 2, 1]

    # dimVal=np.array(range(1, N_DIMS + 1))
    dimWeight = np.exp(dimWeight)
    return dimVal, dimWeight


dimVal, dimWeight = MakeTarList()
MaxFitness = N_DIMS * DNA_BOUND[1] * np.max(dimWeight)


def GetFitness(lens):
    # arr = []
    # for len in lens:
    #     # arr.append(100/abs(len-2))
    #     arr.append(MaxFitness-len)
    # if(np.min(arr) < 0):
    #     print('error')
    return  MaxFitness - np.array(lens)


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
    minDis = 10000
    for DNA in DNAS:
        # 计算所有点的距离，取最小值
        for i in range(N_DIMS):
            for j in range(i + 1, N_DIMS):
                if(isOverLap(TargePos[i],TargePos[j])):
                    if(DNA[i] == DNA[j]):
                    # 范围重叠时，判断高度是否一致
                        minDis = 0.01
                    # break
        sum.append(min((minDis, 1)))
    return sum


# def plotDim(posCur, posTar, dimLen):
#     arrow_params = {'head_width':1, 'head_length':1
#                     , 'length_includes_head': True}
#     axs[0].arrow(posCur[0], posCur[1], 0, dimLen/2, **arrow_params)
#     axs[0].arrow(posCur[0], posTar[1], 0, -dimLen/2, **arrow_params)

#     # 标注指向目标的方向向量
#     dimVec = np.subtract(posCur, posTar)
#     dir = np.cross([dimVec[0],dimVec[1],0],[0,0,1])
#     # 辅助法向量，用于画标注距离线
#     norV = (dir / np.linalg.norm(dir))[:2]
#     lBorder = [posCur,posTar] + norV.repeat(1) * dimLen/2
#     rBorder = [posCur,posTar] - norV.repeat(1) * dimLen/2
#     # 绘制标注边界线
#     axs[0].plot(lBorder[...,0], lBorder[...,1], color='black')
#     axs[0].plot(rBorder[...,0], rBorder[...,1], color='black')

# def plotDNA(DNA):
#     # 清空区域
#     axs[0].cla()
#     axs[1].cla()

#     # drawDNA
#     xl, yl = DNA.reshape((2, N_DIMS))
#     axs[0].scatter(txl, tyl, s=200, lw=0, c='black', alpha=0.5)
#     axs[0].scatter(xl, yl, s=200, lw=0, c='red', alpha=0.5)
#     axs[0].set_title('best dimension graph')

#     for i in range(len(xl)):
#         axs[0].plot([xl[i], txl[i]], [yl[i], tyl[i]], 'r-')
#         # plot text
#         axs[0].text(xl[i], yl[i], str(dimVal[i]), size=15, va="center", ha="center")
#         plotDim([xl[i],yl[i]],[txl[i],tyl[i]],dimVal[i])

#     # draw best fitness
#     axs[1].set_title('best fitness')
#     axs[1].plot(xAis[:curGenIndex], bstFitness[:curGenIndex] - bstFitness[0])
#     fig.tight_layout()
#     plt.pause(0.2)


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
        kv += ks * np.random.rand(*kv.shape)
        if(np.min(kv)<0):
            a=1
        # muteIdx= np.random.randint(0,N_DIMS-1)
        # kv[muteIdx]=np.random.randint(0,DNA_BOUND)
        # 限制范围
        kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value
        kv[:] = np.ceil(kv)
        if(np.min(kv)<0):
            a=1
    return kids

# 移除不好样本


def kill_bad(pop, kids):
    # 新老合并
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    # 获取所有适应度
    lens = GetLen(pop['DNA'])
    fitness = GetFitness(lens)      # calculate global fitness
    minDis = getMinDisToOther(pop['DNA'])

    # fitness = [ fitness[i]  for i in range(len(fitness))]
    # fitness = [(fitness[i] * minDis[i]) for i in range(len(fitness))]
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
        self.pop = dict(DNA=np.random.randint(1, DNA_BOUND[1], N_DIMS * POP_SIZE).reshape(POP_SIZE, N_DIMS),
                        mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values


sd = SmartDim()
plt.ion()
# plotDNA(sd.pop['DNA'][-1])

for i in range(N_GENERATIONS):
    kids = make_kid(sd.pop, N_KID)
    sd.pop = kill_bad(sd.pop, kids)
    print('min dis is ', np.min(getMinDisToOther(sd.pop['DNA'])))
    curGenIndex += 1
    bstDNA = np.concatenate((TargePos, np.array([sd.pop['DNA'][-1]]).T), axis=1)
    ploter.plotDNA(bstDNA)
    # ploter.plotFitness(xAis[:curGenIndex], bstFitness[:curGenIndex] - bstFitness[0])
    ploter.plotFitness(xAis[:curGenIndex], bstFitness[:curGenIndex])

    plt.pause(0.05)

# # 获取所有适应度
# lens = GetLen(sd.pop['DNA'])
# fitness = GetFitness(lens)      # calculate global fitness
# minDis = getMinDisToOther(sd.pop['DNA'])
# fitness = [(fitness[i] * minDis[i]) for i in range(len(fitness))]
# best_idx = np.argmax(fitness)

# plotDNA(sd.pop['DNA'][best_idx])


# DimList = np.random.randint(5, 40, 2 * N_DIMS).reshape(N_DIMS, 2)

# for p in sd.pop['DNA']:
#     bstDNA = np.concatenate((DimList, 10 * np.array([p]).T), axis=1)

#     ploter.plotDNA([bstDNA])
plt.pause(100)
plt.show()
plt.ioff()
