import numpy as np
import matplotlib.pyplot as plt

N_DIMS = 4  # DIM size
DNA_SIZE = N_DIMS * 2             # DNA (real number)
DNA_BOUND = [0, 40]       # solution upper and lower bounds
N_GENERATIONS = 600
POP_SIZE = 1           # population size
N_KID = 40               # n kids per generation

fig, axs = plt.subplots(1, 2)

# DimList = np.random.randint(5, 40, 3 * N_DIMS).reshape(N_DIMS, 3)


def plotDim(posLeft, posRight, txtHeigh):
    arrow_params = {'head_width': 0.4, 'head_length': 0.6,
                    'length_includes_head': True}
    dimLen = abs(posLeft-posRight)
    midPost = posLeft + dimLen/2
    axs[0].arrow(midPost, txtHeigh, dimLen/2, 0, **arrow_params)
    axs[0].arrow(midPost, txtHeigh, -dimLen/2, 0, **arrow_params)

    # 绘制标注边界线
    axs[0].plot([posLeft, posLeft], [0, txtHeigh], color='black')
    axs[0].plot([posRight, posRight], [0, txtHeigh], color='black')


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


def plotDNA(DimPos):
    # 清空区域
    axs[0].cla()
    axs[1].cla()

    for dim in DimPos:
        val = abs(dim[1]-dim[0])
        axs[0].text(dim[0] + val/2, dim[2], str(val),
                    size=15, va="center", ha="center")
        plotDim(dim[0], dim[1], dim[2])


def plotFitness(bstFitness):
    xAis = np.arange(len(bstFitness))
    axs[1].set_title('best fitness')
    axs[1].plot(xAis, bstFitness)
    fig.tight_layout()
    plt.pause(0.02)


def plotFitnessForList(fig, axs, bstFitness, index=0):
    xAis = np.arange(len(bstFitness))
    axs[index, 1].set_title('best fitness')
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
