from logging import root
import numpy as np
import heapq
from anytree import NodeMixin, RenderTree
from utils import entropy, informationGainMethod, InformationType, partition
from anytree.exporter import DotExporter

# data is a list of dictionaries, where the keys are "id", and "features" if data is
# a input, and if data is a label type, then data has keys "id" and "features"

wordDictAddrs = 'a2datasets/words.txt'
wordDictRawFile = open(wordDictAddrs, "r")
wordDictRaw = wordDictRawFile.read()
wordDictRawFile.close()
wordList = wordDictRaw.split('\n')
wordDict = {}
for index in range(len(wordList)):
    wordDict[index+1] = wordList[index]

treeId = 0

class InnerTree(NodeMixin):
    def __init__(self, direction, inputList, trainLabels, parentTree, informationGained = None, constraintVariable = None):
        self.parentTree = parentTree
        self.inputList = inputList
        self.trainLabels = trainLabels
        # this constraint variable is for the children, not for the current tree
        self.constraintVariable = constraintVariable
        # this childrenTree0 is when the constraintVariable is equal to 0, thus the features is not present
        self.childrenTree0 = None
        # this childrenTree0 is when the constraintVariable is equal to 1, thus the features is present
        self.childrenTree1 = None
        self.informationGained = informationGained
        self.prediction = None
        self.name = None
        self.parent = parentTree
        self.children = []
        self.direction = direction

        global treeId
        self.id = treeId
        treeId = treeId + 1

    def hasChildren(self):
        if self.childrenTree0 != None and self.childrenTree1 != None:
            return True
        if self.childrenTree0 == None and self.childrenTree1 == None:
            return False
        assert(1==2)

    def addChildren0(self, childrenDecisionTree):
        if (self.childrenTree0 != None):
            print("error")
            assert(1==2)
        self.childrenTree0 = childrenDecisionTree
    
    def addChildren1(self, childrenDecisionTree):
        if (self.childrenTree1 != None):
            print("error")
            assert(1==2)
        self.childrenTree1 = childrenDecisionTree
        self.children = [self.childrenTree0, self.childrenTree1]
        self.name = self.__repr__()

    def getAllConstraints(self):
        constraints = set([])
        root = self
        while(root.parentTree != None):
            constraintVariable = root.parentTree.constraintVariable
            if (constraintVariable != None):
                constraints.add(constraintVariable)
            else:
                assert(1==2)
            root = root.parentTree
        return constraints

    def predict(self):
        if (self.prediction != None):
            return self.prediction

        count0 = 0
        count1 = 0
        for input in self.inputList:
            id = input['id']
            label = self.trainLabels[id]
            if (label == 0):
                count0 = count0 + 1
            else:
                count1 = count1 + 1
        
        if (count0 < count1):
            self.prediction = 1
        else:
            self.prediction = 0
        
        if self.name == None:
            if self.predict() == 0:
                self.name = 'atheism'
            else:
                self.name = 'books'

        return self.prediction

    def __repr__(self) -> str:
        global wordDict
        feature = 'None'
        if self.constraintVariable != None:
            feature = wordDict[self.constraintVariable]
        return str(feature) + ", " \
            + str(self.informationGained)

class QueueNode():
    def __init__(self, parent, informationGain, constraintVariable):
        self.constraintVariable = constraintVariable
        self.informationGain = informationGain
        self.parent = parent

    def __lt__(self, other):
        return self.informationGain > other.informationGain
    
    def __repr__(self):
        return "constraintVariable: " + str(self.constraintVariable)\
         + ", informationGain: " + str(self.informationGain)\
             + ", parentId: " + str(self.parent.id)


class DecisionTree():
    def __init__(self, numberOfFeatures, trainInputList, trainLabels, type):
        self.numberOfFeatures = numberOfFeatures
        self.trainInputList = trainInputList
        self.trainLabels = trainLabels
        self.type = type
        self.PQ = []
        self.rootTree = InnerTree(0, self.trainInputList, self.trainLabels, parentTree=None)
        bestInitialFeature, maximumInformation = self.getBestFeature(set([]), self.trainInputList)
        newQueueNode = QueueNode(self.rootTree, maximumInformation, bestInitialFeature)
        heapq.heappush(self.PQ, newQueueNode)

        def __repr__(self):
            return "numberOfFeatures: " + str(self.numberOfFeatures)\
            + ", pq: " + str(self.PQ)

    def addOneNode(self):
        if(len(self.PQ) > 0):
            self.updateDecisionTree()
        else:
            return False

    def addAllNodes(self):
        while(len(self.PQ) > 0):
            self.updateDecisionTree()

    def emptyQueue(self):
        return len(self.PQ) == 0

    def updateDecisionTree(self):
        bestQueueNode = heapq.heappop(self.PQ)
        parent = bestQueueNode.parent
        parent.constraintVariable = bestQueueNode.constraintVariable
        informationGained = bestQueueNode.informationGain
        parent.informationGained = informationGained
        partition0, partition1 = partition(parent.inputList, parent.constraintVariable)
        children0 = InnerTree(0,partition0, self.trainLabels, parentTree=parent)
        children1 = InnerTree(1,partition1, self.trainLabels, parentTree=parent)
        parent.addChildren0(children0)
        parent.addChildren1(children1)
        
        for children in [children0, children1]:
            constraints = children.getAllConstraints()
            bestFeature, maximumInformation = self.getBestFeature(constraints, children.inputList)
            if (bestFeature == None):
                assert(1==2)
            else:
                newQueueNode = QueueNode(children, maximumInformation, bestFeature)
                heapq.heappush(self.PQ, newQueueNode)
    
    # constraints is a set of variables/features that already have the values predetermined
    # inputList is a list of inputs that we need to find the best feature
    def getBestFeature(self, constraints, inputList):
        maximumInformationaGain = - np.inf
        bestFeature = None
        for expansionFeature in range(self.numberOfFeatures + 1):
            if (expansionFeature == 0 or expansionFeature in constraints):
                continue
            informationGain = informationGainMethod(inputList, expansionFeature, self.trainLabels, self.type)
            if (maximumInformationaGain < informationGain):
                maximumInformationaGain = informationGain
                bestFeature = expansionFeature
        return bestFeature, maximumInformationaGain

    def oneInference(self, features):
        it = self.rootTree
        while (it.constraintVariable != None):
            if (it.constraintVariable in features):
                it = it.childrenTree1
            else:
                it = it.childrenTree0
        return it.predict()

    def accuracyMeasurement(self, inputList, labels):
        correctPredictions = 0
        wrongPredictions = 0
        for input in inputList:
            prediction = self.oneInference(input['features'])
            label = labels[input['id']]
            if label == prediction:
                correctPredictions = correctPredictions + 1
            else:
                wrongPredictions = wrongPredictions + 1
        return correctPredictions/(correctPredictions + wrongPredictions)
