import numpy as np

#create Node class that rapresents the random variable in Bayesian Network
class Node:

    def __init__(self, name):
        self.name = name
        self.cpt = np.array
        self.parents = []
        self.children = []
        self.domain = []
        self.caption = None

    def getChildren(self):
        return self.children

    def getParents (self):
        return self.parents

    def getCPT(self):
        return self.cpt