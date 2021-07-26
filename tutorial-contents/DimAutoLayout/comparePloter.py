import numpy as np
import matplotlib.pyplot as plt
import LevelTxtHeigh as evo
import GAForDim as ga
import MakeAndPlotDim as ploter

evoBstDna,evoFitCurve = evo.getEvoRst()
xAis = np.arange(len(evoFitCurve))


# ploter.plotDNA(evoBstDna)
# ploter.plotFitness(xAis,evoFitCurve)
# plt.pause(100)

gaBstDna,gaFitCurve = ga.getRst()
ploter.plotCompare([gaBstDna,evoBstDna],[gaFitCurve,evoFitCurve])
# ploter.plotDNA(gaBstDna)
# ploter.plotFitness(xAis,gaFitCurve)
plt.pause(100)