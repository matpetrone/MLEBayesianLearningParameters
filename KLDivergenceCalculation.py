import math
from BaesyanNetwork import calculateJointDistribution, originalDistribution

def calculateKLDivergency(bayesianNet):
    KLdivergence = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        KLdivergence += originalDistribution(i,j,k,l,m) * (math.log(originalDistribution(i,j,k,l,m)) - math.log(calculateJointDistribution(bayesianNet,i,j,k,l,m)))
    bayesianNet.setDivergenceKL(KLdivergence)
    return KLdivergence

#estimates mean for al KL calculated in the netwok of the array
def meanKL(arr):
    mean = 0.0
    for i in range(len(arr)):
        calculateKLDivergency(arr[i][0])
        mean += arr[i][0].getDivergenceKL()
    return mean/len(arr)