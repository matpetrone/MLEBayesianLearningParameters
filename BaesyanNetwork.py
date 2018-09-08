from Node import Node
import numpy as np

#Create the Graph Network on the babsis of the graph generated on Hugin
class BayesianNet:
    def __init__(self):
        self.nodeA = Node('A')
        self.nodeB = Node('B')
        self.nodeC = Node('C')
        self.nodeD = Node('D')
        self.nodeE = Node('E')

        self.nodeA.children.append(self.nodeB)
        self.nodeA.children.append(self.nodeC)
        self.nodeC.children.append(self.nodeD)
        self.nodeB.children.append(self.nodeD)
        self.nodeC.children.append(self.nodeE)

        self.nodeC.parents.append(self.nodeA)
        self.nodeB.parents.append(self.nodeA)
        self.nodeD.parents.append(self.nodeB)
        self.nodeD.parents.append(self.nodeC)
        self.nodeE.parents.append(self.nodeC)

        self.nodeA.caption = 'Metastatic Cancer'
        self.nodeB.caption = 'Serum Calcium'
        self.nodeC.caption = 'Brain Tumor'
        self.nodeD.caption = 'Coma'
        self.nodeE.caption = 'Severe Headaches'

        self.nodeA.domain = ['Present', 'Absent']
        self.nodeB.domain = ['Increased', 'Not increased']
        self.nodeC.domain = ['Present', 'Absent']
        self.nodeD.domain = ['Present', 'Absent']
        self.nodeE.domain = ['Present', 'Absent']

        self.nodeA.cpt = np.zeros(shape=(1,1))
        self.nodeB.cpt = np.zeros(shape=(2,1))
        self.nodeC.cpt = np.zeros(shape=(2,1))
        self.nodeD.cpt = np.zeros(shape=(4,1))
        self.nodeE.cpt = np.zeros(shape=(2,1))

        self.divergenceKL = None

    def getDivergenceKL(self):
        return self.divergenceKL
    def setDivergenceKL(self, div):
        self.divergenceKL = div

def calculateJointDistribution(bayesianNet, a, b, c, d, e):
    indexCptA = None
    indexCptB = None
    indexCptC = None
    indexCptD = None
    indexCptE = None
    extraIndexC = None

    if (a == 1):
        indexCptA = 0
        valueFromCptA = bayesianNet.nodeA.getCPT()[0]
    elif (a == 0):
        indexCptA = 1
        valueFromCptA = 1 - bayesianNet.nodeA.getCPT()[0]

    if (b == 1):
        indexCptB = 0
        valueFromCptB = bayesianNet.nodeB.getCPT()[indexCptA]
    elif (b == 0):
        indexCptB = 2
        valueFromCptB = 1 - bayesianNet.nodeB.getCPT()[indexCptA]

    if (c == 1):
        indexCptC = 0
        extraIndexC = indexCptB
        valueFromCptC = bayesianNet.nodeC.getCPT()[indexCptA]
    elif (c == 0):
        indexCptC = 1
        extraIndexC = indexCptB + 1
        valueFromCptC = 1 - bayesianNet.nodeC.getCPT()[indexCptA]

    if (d == 1):
        valueFromCptD = bayesianNet.nodeD.getCPT()[extraIndexC]
    elif (d == 0):
        valueFromCptD = 1 - bayesianNet.nodeD.getCPT()[extraIndexC]

    if (e == 1):
        valueFromCptE = bayesianNet.nodeE.getCPT()[indexCptC]
    elif (e == 0):
        valueFromCptE = 1 - bayesianNet.nodeE.getCPT()[indexCptC]


    jointDistribution = valueFromCptA * valueFromCptB * valueFromCptC * valueFromCptD * valueFromCptE

    return jointDistribution

#This function calculate joint distribution for the original bayesian network using const value taken from Hugin file

def originalDistribution(a, b, c, d, e):
    valueFromCptA = None
    valueFromCptB = None
    valueFromCptC = None
    valueFromCptD = None
    valueFromCptE = None

    if (a == 1):
        valueFromCptA = 0.2
        if(b==1):
            valueFromCptB = 0.8
        elif(b==0):
            valueFromCptB = 0.2
        if(c==1):
            valueFromCptC = 0.2
        elif(c==0):
            valueFromCptC=0.8
    elif (a == 0):
        valueFromCptA = 0.8
        if (b == 1):
            valueFromCptB = 0.2
        elif (b == 0):
            valueFromCptB = 0.8
        if (c == 1):
            valueFromCptC = 0.05
        elif (c == 0):
            valueFromCptC = 0.95
    if (e == 1):
        if (c == 1):
            valueFromCptE = 0.8
        elif (c == 0):
            valueFromCptE = 0.6
    elif (e == 0):
        if (c == 1):
            valueFromCptE = 0.2
        elif (c == 0):
            valueFromCptE = 0.4

    if (d == 1):
        if(b==1 and c==1):
            valueFromCptD = 0.8
        elif (b==1 and c==0):
            valueFromCptD = 0.9
        elif (b==0 and c==1):
            valueFromCptD = 0.7
        elif (b==0 and c==0):
            valueFromCptD = 0.05
    elif (d == 0):
        if (b == 1 and c == 1):
            valueFromCptD = 0.2
        elif (b == 1 and c == 0):
            valueFromCptD = 0.1
        elif (b == 0 and c == 1):
            valueFromCptD = 0.3
        elif (b == 0 and c == 0):
            valueFromCptD = 0.95

    distribution = valueFromCptA * valueFromCptB * valueFromCptC * valueFromCptD * valueFromCptE

    return distribution





