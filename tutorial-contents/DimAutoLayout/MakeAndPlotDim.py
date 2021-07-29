# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

N_DIMS = 4  # DIM size
DNA_SIZE = N_DIMS * 2             # DNA (real number)
DNA_BOUND = [0, 40]       # solution upper and lower bounds
N_GENERATIONS = 600
POP_SIZE = 1           # population size
N_KID = 40               # n kids per generation

fig, axs = plt.subplots(1, 3)

# DimList = np.random.randint(5, 40, 3 * N_DIMS).reshape(N_DIMS, 3)


def plotDim(posLeft, posRight, txtHeigh, index):
    arrow_params = {'head_width': 0.2, 'head_length': 0.6,
                    'length_includes_head': True}

    
    dimLen = abs(posLeft-posRight)
    midPost = posLeft + dimLen/2
    axs[index].arrow(midPost, txtHeigh, dimLen/2, 0, **arrow_params)
    axs[index].arrow(midPost, txtHeigh, -dimLen/2, 0, **arrow_params)

    # 绘制标注边界线
    axs[index].plot([posLeft, posLeft], [0, txtHeigh], color='black')
    axs[index].plot([posRight, posRight], [0, txtHeigh], color='black')


def plotDimForList(axs, posLeft, posRight, txtHeigh, index):
    arrow_params = {'head_width': 0.4, 'head_length': 0.6,
                    'length_includes_head': True}
    dimLen = abs(posLeft-posRight)
    midPost = posLeft + dimLen/2
    
    axs[index, 0].arrow(midPost, txtHeigh, dimLen/2, 0, **arrow_params)
    axs[index, 0].arrow(midPost, txtHeigh, -dimLen/2, 0, **arrow_params)

    # 绘制标注边界线
    axs[index,0].plot([posLeft, posLeft], [0, txtHeigh], color='black')
    axs[index,0].plot([posRight, posRight], [0, txtHeigh], color='black')


def plotFstDNA(DimPos):
    axs[0].set_title('初始布局', fontsize = 16)
    for dim in DimPos:
        val = abs(dim[1]-dim[0])
        axs[0].text(dim[0] + val/2, dim[2], str(val),
                    size=15, va="center", ha="center")
        plotDim(dim[0], dim[1], dim[2], 0)

def plotDNA(DimPos):
    # 清空区域
    axs[1].cla()
    axs[2].cla()
    axs[1].set_title('最佳布局', fontsize = 16)
    axs[1].set_yticks(np.arange(0, len(DimPos), 1))

    for dim in DimPos:
        val = abs(dim[1]-dim[0])
        axs[1].text(dim[0] + val/2, dim[2], str(val),
                    size=15, va="center", ha="center")
        plotDim(dim[0], dim[1], dim[2], 1)


def plotFitness(bstFitness):
    xAis = np.arange(len(bstFitness))
    axs[2].set_title('适应度曲线', fontsize = 16)
    axs[2].set_xlabel('迭代次数', fontsize = 16)
    axs[2].set_ylabel('适应度', fontsize = 16)
    axs[2].plot(xAis, bstFitness)
    fig.tight_layout()
    plt.pause(0.02)


def plotFitnessForList(fig, axs, bstFitness, index=0):
    xAis = np.arange(len(bstFitness))
    axs[index, 1].set_title('适应度曲线', fontsize = 16)
    axs[index, 1].plot(xAis, bstFitness)
    fig.tight_layout()
    plt.pause(0.02)
# plotDNA([DimList])
# plt.pause(10)


def plotDNAForList(ax, DimPos, index):
    for dim in DimPos:
        val = abs(dim[1]-dim[0])
        ax[index, 0].text(dim[0] + val/2, dim[2], str(val),
                          size=15, va="center", ha="center")
        plotDimForList(ax, dim[0], dim[1], dim[2], index)


def plotCompare(DNAS, fitCurves):
    fig, axs = plt.subplots(2, 2)

    for i in range(0, len(DNAS)):
        plotDNAForList(axs, DNAS[i], i)
        plotFitnessForList(fig, axs, fitCurves[i], i)

def pause():
    plt.pause(100)
