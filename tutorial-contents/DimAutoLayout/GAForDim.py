import numpy as np
import MakeAndPlotDim as ploter
import AutoDimUtil as util


N_DIMS = 8  # DNA size
DNA_SIZE = N_DIMS            # DNA (real number)
CROSS_RATE = 0.2
MUTATE_RATE = 1/N_DIMS
POP_SIZE = 100
N_GENERATIONS = 400
DNA_BOUND = [1,  N_DIMS]       # solution upper and lower bounds

bstFitnessList = []
xAis = np.arange(N_GENERATIONS)

bstDNA = ''

dt = util.DimUtil(DNA_SIZE)
MaxFitness = N_DIMS * DNA_BOUND[1] * np.max(dt.dimWeight)

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
        if np.random.rand() < CROSS_RATE:
            i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
            cross_points = np.random.randint(0, 2, size=self.DNA_size).astype(np.bool)   # choose crossover points
            parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
        return parent

    def mutate(self, child, index = 0 ):
        startIdx = int((N_GENERATIONS - index) / N_GENERATIONS * self.DNA_size)
        print('startIdx = ', startIdx)
        for point in range(startIdx):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.randint(DNA_BOUND[0], DNA_BOUND[1])
        return child


    def evolve(self, fitness, index = 0):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child, index)
            parent[:] = child
        self.pop = pop

def getRst():
    bstFitness = 0
    ga = GA(DNA_size=N_DIMS, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
    bstDNA = ga.pop[0]

    for generation in range(N_GENERATIONS):
        startIdx = int((N_GENERATIONS - generation) / N_GENERATIONS * DNA_SIZE)
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

        bstFitnessList.append(bstFitness)
        print('Gen:', generation, '| best fit: %.2f' % bstFitness,)

        # curBstDNA = np.concatenate(
        #     (TargePos, np.array([bstDNA]).T), axis=1)
        # ploter.plotDNA(curBstDNA)
        # ploter.plotFitness(xAis[:curGenIndex],
        #                 bstFitnessList[:curGenIndex] - bstFitnessList[0])
        print('max fit', bstFitness)
        
        ga.evolve(fitness, startIdx)
    curBstDNA = np.concatenate(
        (dt.targetPos, np.array([bstDNA]).T), axis=1)
    return curBstDNA,bstFitnessList

def runLocalGa():
    ga = GA(DNA_size=N_DIMS, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
    bstDNA = ga.pop[0]
    bstFitness = 0

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

        bstFitnessList.append(bstFitness)
        print('Gen:', generation, '| best fit: %.2f' % bstFitness,)

        # curBstDNA = np.concatenate(
        #     (TargePos, np.array([bstDNA]).T), axis=1)
        # ploter.plotDNA(curBstDNA)
        # ploter.plotFitness(xAis[:curGenIndex],
        #                 bstFitnessList[:curGenIndex] - bstFitnessList[0])
        print('max fit', bstFitness)
        
        ga.evolve(fitness, generation)
    curBstDNA = np.concatenate(
        (dt.targetPos, np.array([bstDNA]).T), axis=1)
    ploter.plotDNA(curBstDNA)
    ploter.plotFitness(bstFitnessList - bstFitnessList[0])
    ploter.pause()
runLocalGa()