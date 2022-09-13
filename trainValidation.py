import numpy as np
from decisionTree import DecisionTree
from utils import InformationType
import matplotlib.pyplot as plt
from anytree import RenderTree

def convertRawFormatToInputList(rawData):
    inputList = []
    currentId = 1
    currentInput = {}
    currentInput['id'] = currentId
    currentInput['features'] = []
    for dataRow in rawData:
        if not (dataRow[0] == currentId):
            currentId = currentId + 1
            inputList.append(currentInput)
            currentInput = {}
            currentInput['id'] = currentId
            currentInput['features'] = []
        currentInput['features'].append(int(dataRow[1]))
    inputList.append(currentInput)
    return inputList

def convertRawFormatToLabelDict(rawLabel):
    labelDict = {}
    for index in range(len(rawLabel)):
        labelDict[index + 1] = int(rawLabel[index] - 1)
    return labelDict


trainDataAddrs = 'a2datasets/trainData.txt'
testDataAddrs = 'a2datasets/testData.txt'
trainLabelAddrs = 'a2datasets/trainLabel.txt'
testLabelAddrs = 'a2datasets/testLabel.txt'
wordDictAddrs = 'a2datasets/words.txt'

trainDataRaw = np.loadtxt(trainDataAddrs)
testDataRaw = np.loadtxt(testDataAddrs)
trainLabelRaw = np.loadtxt(trainLabelAddrs)
testLabelRaw = np.loadtxt(testLabelAddrs)

wordDictRawFile = open(wordDictAddrs, "r")
wordDictRaw = wordDictRawFile.read()
wordDictRawFile.close()
wordList = wordDictRaw.split('\n')
wordDict = {}
for index in range(len(wordList)):
    wordDict[index+1] = wordList[index]

trainInputList = convertRawFormatToInputList(trainDataRaw)
testInputList = convertRawFormatToInputList(testDataRaw)

trainLabelDict = convertRawFormatToLabelDict(trainLabelRaw)
testLabelDict = convertRawFormatToLabelDict(testLabelRaw)

numberOfFeatures = 6968

averageDT = DecisionTree(numberOfFeatures=numberOfFeatures, trainInputList=trainInputList,
    trainLabels=trainLabelDict, type=InformationType.averageInformation)
weightedDT = DecisionTree(numberOfFeatures=numberOfFeatures, trainInputList=trainInputList,
    trainLabels=trainLabelDict, type=InformationType.weightedInformation)

averageTrainAccs = []
averageTestAccs = []
weightedTrainAccs = []
weightedTestAccs = []

averageTrainAccs.append(averageDT.accuracyMeasurement(inputList=trainInputList, labels=trainLabelDict))
averageTestAccs.append(averageDT.accuracyMeasurement(inputList=testInputList, labels=testLabelDict))

weightedTrainAccs.append(weightedDT.accuracyMeasurement(inputList=trainInputList, labels=trainLabelDict))
weightedTestAccs.append(weightedDT.accuracyMeasurement(inputList=testInputList, labels=testLabelDict))

numberOfExpandedNodes = 0

import time
initialTime = time.time()
maxExpandedNodes = 10

while((not(averageDT.emptyQueue()) or not(weightedDT.emptyQueue())) and numberOfExpandedNodes <= maxExpandedNodes - 1):
    currentTime = time.time()
    print("time passed in minutes", (currentTime-initialTime)/60)
    print("numberOfExpandedNodes", numberOfExpandedNodes)
    if (numberOfExpandedNodes%50 == 0):
        print("average", averageTrainAccs)
        plt.plot(averageTrainAccs, label='train accuraccy average information gain')
        plt.plot(averageTestAccs, label = 'test accuraccy average information gain')
        plt.plot(weightedTrainAccs, label = 'train accuraccy weighted information gain')
        plt.plot(weightedTestAccs, label = 'test accuraccy weighted information gain')
        plt.legend()
        plt.show()
    numberOfExpandedNodes = numberOfExpandedNodes + 1
    averageDT.addOneNode()
    weightedDT.addOneNode()

    averageTrainAccs.append(averageDT.accuracyMeasurement(inputList=trainInputList, labels=trainLabelDict))
    averageTestAccs.append(averageDT.accuracyMeasurement(inputList=testInputList, labels=testLabelDict))

    weightedTrainAccs.append(weightedDT.accuracyMeasurement(inputList=trainInputList, labels=trainLabelDict))
    weightedTestAccs.append(weightedDT.accuracyMeasurement(inputList=testInputList, labels=testLabelDict))


plt.plot(averageTrainAccs, label='train accuraccy average information gain')
plt.plot(averageTestAccs, label = 'test accuraccy average information gain')
plt.plot(weightedTrainAccs, label = 'train accuraccy weighted information gain')
plt.plot(weightedTestAccs, label = 'test accuraccy weighted information gain')
plt.legend()
plt.show()

from anytree.exporter import DotExporter
DotExporter(averageDT.rootTree).to_picture("average IG draw.png")
DotExporter(weightedDT.rootTree).to_picture("weighted IG draw.png")

def mysort(items):
    return sorted(items, key=lambda item: item.direction)
for row in RenderTree(weightedDT.rootTree, childiter=mysort):
    print("%s%s" % (row.pre, row.node.name))

for row in RenderTree(averageDT.rootTree, childiter=mysort):
    print("%s%s" % (row.pre, row.node.name))
