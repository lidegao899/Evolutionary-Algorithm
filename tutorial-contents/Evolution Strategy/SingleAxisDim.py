import numpy as np
import matplotlib.pyplot as plt

N_DIMS = 8  # DIM size
DNA_SIZE = N_DIMS * 2             # DNA (real number)
DNA_BOUND = [0, 40]       # solution upper and lower bounds
N_GENERATIONS = 300
POP_SIZE = 300           # population size
N_KID = 60               # n kids per generation

# 两个目标点
TargePos = np.array([[-2, 20], [-2, 20]])

# plot var
xAis = np.arange(N_GENERATIONS)
bstFitness = np.empty(N_GENERATIONS)
curGenIndex = 0
fig, axs = plt.subplots(1, 2)


def MakePnt():
    return np.random.rand(N_DIMS, 2)

# 生成列表形式的目标点坐标，形式为 x1x1x2x2 y1y1y2y2,与坐标点对应


def MakeTarList():
    xl = TargePos[..., 0].repeat(N_DIMS/2)
    yl = TargePos[..., 1].repeat(N_DIMS/2)
    return xl, yl


txl, tyl = MakeTarList()


def GetFitness(lens):
    arr = []
    for len in lens:
        arr.append(100/abs(len-5))
    return arr

# 获取所有样本的长度


def GetLen(xys):
    # 样本所有点到（0,0）的距离
    sum = []
    for xy in xys:
        xl, yl = xy.reshape((2, N_DIMS))
        # len=np.sum(np.sqrt((xl - TargePos[0])**2 + (yl - TargePos[1])**2))
        len = np.sum(np.sqrt((xl - txl)**2 + (yl - tyl)**2))
        # sum.append(max(len,2))

        sum.append(1/(len))
    return sum

# 计算DNA内最近点的距离


def getMinDisToOther(DNAS):
    sum = []
    for DNA in DNAS:
        minDis = 100000
        xl, yl = DNA.reshape((2, N_DIMS))
        # 计算所有点的距离，取最小值
        for i in range(N_DIMS):
            for j in range(i + 1, N_DIMS):
                len = np.sum(np.sqrt((xl[i]-xl[j])**2+(yl[i]-yl[j])**2))
                minDis = min(minDis, len)
        sum.append(min((minDis/3)**3, 1))
    return sum

def plotDNA(DNA):
    # 清空区域
    axs[0].cla()
    axs[1].cla()

    # drawDNA
    xl, yl = DNA.reshape((2, N_DIMS))
    axs[0].scatter(txl, tyl, s=200, lw=0, c='black', alpha=0.5)
    axs[0].scatter(xl, yl, s=200, lw=0, c='red', alpha=0.5)

    for i in range(len(xl)):
        axs[0].plot([xl[i], txl[i]], [yl[i], tyl[i]], 'r-')

    # draw best fitness
    axs[1].plot(xAis, bstFitness)
    fig.tight_layout()
    plt.pause(0.2)


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

        # kv[N_DIMS:]=pop['DNA'][p1][N_DIMS:]
        # ks[N_DIMS] = pop['mut_strength'][p1]

        # 正态分布标准差
        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(
            ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
        # 正态分布
        kv += ks * np.random.randn(*kv.shape)
        # 限制范围
        kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value

        # 坐标预处理，对其y坐标
        for i in range(N_DIMS,DNA_SIZE):
            kv[i] = tyl[i- N_DIMS]

        # kv[N_DIMS:]=pop['DNA'][p1][N_DIMS:]
        # ks[N_DIMS] = pop['mut_strength'][p1]
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
    fitness = [(fitness[i] * minDis[i]) for i in range(len(fitness))]

    bestFit = np.max(fitness)
    print('max fit', bestFit)
    bstFitness[curGenIndex] = np.max(bestFit)

    idx = np.arange(pop['DNA'].shape[0])
    # 递增排列，取后POP_SIZE位
    # selected by fitness ranking (not value)
    good_idx = idx[np.argsort(fitness)][-POP_SIZE:]
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop


class SmartDim(object):
    def __init__(self):
        # 生成与目标标注相同的Y坐标点标注
        # DNAS = np.array([np.array([np.random.rand(1, N_DIMS),[tyl]]).reshape(1,DNA_SIZE) for i in range(POP_SIZE)])
        # tmp = np.array([np.random.rand(1, N_DIMS),[tyl]]).reshape(1,DNA_SIZE).repeat(POP_SIZE, axis=0)
        self.pop = dict(DNA =np.array([np.random.rand(1, N_DIMS),[tyl]]).reshape(1,DNA_SIZE).repeat(POP_SIZE, axis=0),   # initialize the pop DNA values
                        mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values

sd = SmartDim()
plt.ion()
plotDNA(sd.pop['DNA'][-1])
plt.pause(0.01)
for i in range(N_GENERATIONS):
    kids = make_kid(sd.pop, N_KID)
    sd.pop = kill_bad(sd.pop, kids)
    print('min dis is ', np.min(getMinDisToOther(sd.pop['DNA'])))
    curGenIndex += 1
    plotDNA(sd.pop['DNA'][-1])
    plt.pause(0.01)


# 获取所有适应度
lens = GetLen(sd.pop['DNA'])
fitness = GetFitness(lens)      # calculate global fitness
minDis = getMinDisToOther(sd.pop['DNA'])
fitness = [(fitness[i] * minDis[i]) for i in range(len(fitness))]
best_idx = np.argmax(fitness)

plotDNA(sd.pop['DNA'][best_idx])
plt.ioff()
plt.show()
