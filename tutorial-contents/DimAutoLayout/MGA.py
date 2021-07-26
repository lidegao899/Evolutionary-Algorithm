import numpy as np
import MakeAndPlotDim as ploter
import AutoDimUtil as util

# DNA_SIZE = 10            # DNA length
# POP_SIZE = 20            # population size
# CROSS_RATE = 0.6         # mating probability (DNA crossover)
MUTATION_RATE = 0.1     # mutation probability
# N_GENERATIONS = 200
# X_BOUND = [0, 5]         # x upper and lower bounds


N_DIMS = 8  # DNA size
DNA_SIZE = N_DIMS            # DNA (real number)
CROSS_RATE = 0.2
MUTATE_RATE = 1/N_DIMS
POP_SIZE = 100
N_GENERATIONS = 400
DNA_BOUND = [1,  N_DIMS]       # solution upper and lower bounds


def F(x): return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function

bstFitnessList = []
xAis = np.arange(N_GENERATIONS)

bstDNA = ''

dt = util.DimUtil(DNA_SIZE)
MaxFitness = N_DIMS * DNA_BOUND[1] * np.max(dt.dimWeight)


class MGA(object):
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        # initial DNAs for winner and loser
        # self.pop = np.random.randint(*DNA_bound, size=(1, self.DNA_size)).repeat(pop_size, axis=0)
        self.pop = np.vstack([np.random.permutation(np.arange(1,N_DIMS + 1)) for _ in range(pop_size)])

    # def translateDNA(self, pop):
    #     # convert binary DNA to decimal and normalize it to a range(0, 5)
    #     return pop.dot(2 ** np.arange(self.DNA_size)[::-1]) / float(2 ** self.DNA_size - 1) * X_BOUND[1]

    def get_fitness(self, product):
        return product      # it is OK to use product value as fitness in here

    def crossover(self, loser_winner):      # crossover for loser
        cross_idx = np.empty((self.DNA_size,)).astype(np.bool)
        for i in range(self.DNA_size):
            cross_idx[i] = True if np.random.rand() < self.cross_rate else False  # crossover index
        loser_winner[0, cross_idx] = loser_winner[1, cross_idx]  # assign winners genes to loser
        return loser_winner

    def mutate(self, loser_winner):         # mutation for loser
        mutation_idx = np.empty((self.DNA_size,)).astype(np.bool)
        for i in range(self.DNA_size):
            mutation_idx[i] = True if np.random.rand() < self.mutate_rate else False  # mutation index
        # flip values in mutation points
        tmp = ~loser_winner[0, mutation_idx].astype(np.bool)
        # loser_winner[0, mutation_idx] = ~loser_winner[0, mutation_idx].astype(np.bool)
        for i in range(len(mutation_idx)):
            if  mutation_idx[i] == True:
                loser_winner[0, i] = np.random.randint(1,DNA_BOUND[1])
        return loser_winner

    def evolve(self, n):    # nature selection wrt pop's fitness
        bstFitness = 0
        for _ in range(n):  # random pick and compare n times
            sub_pop_idx = np.random.choice(np.arange(0, self.pop_size), size=2, replace=False)
            sub_pop = self.pop[sub_pop_idx]             # pick 2 from pop

            # 获取适应度
            lens = dt.GetLen(sub_pop)
            fitness = dt.GetFitness(MaxFitness, lens)      # calculate global fitness
            minDis = dt.getMinDisToOther(sub_pop)
            fitness = fitness * minDis

            # # update fitness
            # curBestFit = np.max(fitness)
            # if(curBestFit > bstFitness):
            #     bstFitness = curBestFit
            #     best_idx = np.argmax(fitness)
            #     bstDNA = self.pop[best_idx]
            # bstFitnessList.append(bstFitness)
            

            # product = F(self.translateDNA(sub_pop))
            # fitness = self.get_fitness(product)
            loser_winner_idx = np.argsort(fitness)
            loser_winner = sub_pop[loser_winner_idx]    # the first is loser and second is winner
            loser_winner = self.crossover(loser_winner)
            loser_winner = self.mutate(loser_winner)
            self.pop[sub_pop_idx] = loser_winner

        # DNA_prod = self.translateDNA(self.pop)
        # pred = F(DNA_prod)
        # return DNA_prod, pred


# plt.ion()       # something about plotting
# x = np.linspace(*X_BOUND, 200)
# plt.plot(x, F(x))

# ga = MGA(DNA_size=DNA_SIZE, DNA_bound=[0, 1], cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)

# for _ in range(N_GENERATIONS):                    # 100 generations
#     DNA_prod, pred = ga.evolve(5)          # natural selection, crossover and mutation

#     # something about plotting
#     if 'sca' in globals(): sca.remove()
#     sca = plt.scatter(DNA_prod, pred, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

# plt.ioff();plt.show()

def runLocalMAG():
    ga = MGA(DNA_size=DNA_SIZE, DNA_bound=[0, 1], cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)
    bstFitness = 0

    for generation in range(N_GENERATIONS):                    # 100 generations
        # DNA_prod, pred = ga.evolve(5)          # natural selection, crossover and mutation
        ga.evolve(5)          # natural selection, crossover and mutation
        lens = dt.GetLen(ga.pop)
        fitness = dt.GetFitness(MaxFitness, lens)      # calculate global fitness
        minDis = dt.getMinDisToOther(ga.pop)
        fitness = fitness * minDis
        curBestFit = np.max(fitness)
        if(curBestFit > bstFitness):
            bstFitness = curBestFit
            best_idx = np.argmax(fitness)
            bstDNA = ga.pop[best_idx]
        bstFitnessList.append(bstFitness)

    curBstDNA = np.concatenate(
        (dt.targetPos, np.array([bstDNA]).T), axis=1)
    ploter.plotDNA(curBstDNA)
    ploter.plotFitness(bstFitnessList - bstFitnessList[0])
    # ploter.pause()

runLocalMAG()