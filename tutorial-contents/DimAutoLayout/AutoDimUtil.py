import numpy as np
import matplotlib.pyplot as plt

class DimUtil(object):
    def __init__(self, DNA_size):
        self.DNA_size = DNA_size
        self.targetPos,self.lens,self.dimWeight = self.MakeTarList()
        
    def MakeTarList(self):
        keys = np.random.randint(1, 40, self.DNA_size*2).reshape(self.DNA_size, 2)
        keys = np.sort(keys)
        # 防止标准值过小bu
        for key in keys:
            if(key[1] - key[0] < 3):
                key[1] = np.clip(key[0] + np.random.randint(3, 40), 1, 40)
        # 计算长度
        lens = np.array([key[1] - key[0] for key in keys])
        index = np.argsort(-lens)
        keys = keys[index]
        weight = np.exp(2 * np.array(range(1, len(keys)+1)))
        lens = np.array([key[1] - key[0] for key in keys])

        # 生成顺序
        return keys, lens, weight

    def isOverLap(self, dimA, dimB):
        if(dimA[0] > dimB[0] and dimA[0] < dimB[1]):
            return True
        if(dimA[0] < dimB[0] and dimA[1] > dimB[0]):
            return True
        if(dimA[0] == dimB[0]):
            return True
        return False


    def getOverlapNum(self, TargePos):
        XOverlapNum = np.zeros(40)
        for dimRandge in TargePos:
            for index in range(dimRandge[0], dimRandge[1]):
                XOverlapNum[index] += 1
        return np.max(XOverlapNum)


    # NUM_OVERLAP = getOverlapNum()
    # VALID_DIM_RANGE = NUM_OVERLAP
    # DNA_BOUND[1] = NUM_OVERLAP


    def GetFitness(self, MaxFitness,lens):
        return MaxFitness - np.array(lens)


    # 获取所有样本的长度


    def GetLen(self, yList):
        # 样本所有点到（0,0）的距离
        sum = []
        for y in yList:
            lenList = y * self.dimWeight
            len = np.sum(lenList)
            sum.append(len)
        return sum

    # 计算DNA内最近点的距离


    def getMinDisToOther(self, DNAS):
        # 有重叠的标注，高度一样则非法，mindis = 0.01
        sum = []
        for DNA in DNAS:
            minDis = 1
            # 计算所有点的距离，取最小值
            for i in range(self.DNA_size):
                for j in range(i + 1, self.DNA_size):
                    if(self.isOverLap(self.targetPos[i], self.targetPos[j])):
                        if(DNA[i] == DNA[j]):
                            # 范围重叠时，判断高度是否一致
                            minDis *= 0.5
            sum.append(min(minDis, 1))
        return sum