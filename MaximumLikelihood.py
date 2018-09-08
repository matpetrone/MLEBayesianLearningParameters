from __future__ import division
from __future__ import print_function
import numpy as np
import csv
from BaesyanNetwork import BayesianNet

#calculate parameter with Maximum Likelihood for the given network
def MLE(csvFile, network):
    # array to collect data from .csv file
    A = []
    B = []
    C = []
    D = []
    E = []
    with open(csvFile) as csvfile:
        reader = csv.DictReader(csvfile, delimiter = ';')
        for row in reader:
            A.append(row['A'])
            B.append(row['B'])
            C.append(row['C'])
            D.append(row['D'])
            E.append(row['E'])

    for i in range(len(network.nodeA.domain)):
        partialData = 0
        totalData = 0
        print ('A =', network.nodeA.domain[i], '| {} = ',)
        for j in range(len(A)):
            if A[j] == network.nodeA.domain[i]:
                partialData += 1
            totalData += 1
        #Calculate likelihood with Laplace Smoothing for k=1, where +2 is due to (lenght of the domain)*k
        likelihoodA = (partialData+1)/(totalData+2)
        print (likelihoodA)
        if (i==0):
            saveCPT(network.nodeA, likelihoodA, i)
            #print(network.nodeA.getCPT())
    print ('')


    for i in range(len(network.nodeC.domain)):
        for j in range(len(network.nodeE.domain)):
            partialData = 0
            totalData = 0
            print ('(E =', network.nodeE.domain[j],') | (C =', network.nodeC.domain[i], ') = ',)
            for k in range(len(E)):
                if C[k] == network.nodeC.domain[i]:
                    if E[k] == network.nodeE.domain[j]:
                        partialData += 1
                    totalData += 1
            likelihoodEgivenC = (partialData+1)/(totalData+2)
            print (likelihoodEgivenC)
            if (j==0):
                saveCPT(network.nodeE, likelihoodEgivenC, i)
                #print(network.nodeE.getCPT())
    print ('')

    for i in range(len(network.nodeA.domain)):
        for j in range(len(network.nodeC.domain)):
            partialData = 0
            totalData = 0
            print ('(C =', network.nodeC.domain[j],') | (A =', network.nodeA.domain[i], ') = ',)
            for k in range(len(C)):
                if A[k] == network.nodeA.domain[i]:
                    if C[k] == network.nodeC.domain[j]:
                        partialData += 1
                    totalData += 1
            likelihoodCgivenA = (partialData+1)/(totalData+2)
            print (likelihoodCgivenA)
            if (j==0):
                saveCPT(network.nodeC, likelihoodCgivenA, i)
                #print(network.nodeC.getCPT())
    print ('')

    for i in range(len(network.nodeA.domain)):
        for j in range(len(network.nodeB.domain)):
            partialData = 0
            totalData = 0
            print ('(B =', network.nodeB.domain[j],') | (A =', network.nodeA.domain[i], ') = ',)
            for k in range(len(B)):
                if A[k] == network.nodeA.domain[i]:
                    if B[k] == network.nodeB.domain[j]:
                        partialData += 1
                    totalData += 1
            likelihoodBgivenA = (partialData+1)/(totalData+2)
            print (likelihoodBgivenA)
            if (j==0):
                saveCPT(network.nodeB, likelihoodBgivenA, i)
                #print(network.nodeB.getCPT())
    print ('')

    for i in range(len(network.nodeB.domain)):
        for j in range(len(network.nodeC.domain)):
            for k in range(len(network.nodeD.domain)):
                partialData = 0
                totalData = 0
                print('(D =', network.nodeD.domain[k], ') | (B =', network.nodeB.domain[i], ', C =', network.nodeC.domain[j], ') =', )
                for l in range(len(D)):
                    if B[l] == network.nodeB.domain[i]:
                        if C[l] == network.nodeC.domain[j]:
                            if D[l] == network.nodeD.domain[k]:
                                partialData += 1
                            totalData += 1
                likelihoodDgivenBC = (partialData+1)/(totalData+2)
                print(likelihoodDgivenBC)
                if (k==0 and i==0):
                    saveCPT(network.nodeD, likelihoodDgivenBC, j)
                    #print(network.nodeD.getCPT())
                elif (k==0 and i==1):
                    saveCPT(network.nodeD, likelihoodDgivenBC, j+2)
                    #print(network.nodeD.getCPT())
    print('')


def saveCPT(node, likelihood, i):
    node.getCPT()[i] = [likelihood]

#Create a BayesianNet() object for every array cell and calculates MLE for each one
def arrayMLE(arr):
    for i in range(len(arr)):
        arr[i][0] = BayesianNet()
        MLE(arr[i][1], arr[i][0])
