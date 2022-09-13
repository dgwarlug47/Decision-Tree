import enum
import math

class InformationType(enum.Enum):
    averageInformation = 1
    weightedInformation = 2

def partition(inputList, feature):
    setWithFeatureValueEqual0 = []
    setWithFeatureValueEqual1 = []
    for input in inputList:
        if feature in input['features']:
            setWithFeatureValueEqual1.append(input)
        else:
            setWithFeatureValueEqual0.append(input)
    return setWithFeatureValueEqual0, setWithFeatureValueEqual1

def informationGainMethod(inputList, expansionFeature, label, type):
    partition1, partition2 = partition(inputList, expansionFeature)
    entropy1 = entropy(partition1, label)
    entropy2 = entropy(partition2, label)
    entropy3 = entropy(inputList, label)

    if (type ==InformationType.averageInformation):
        return entropy3 - 0.5*entropy1 - 0.5*entropy2
    else:
        N1 = len(partition1)
        N2 = len(partition2)
        N = N1 + N2
        return entropy3 - (N1/N)*entropy1 - (N2/N)*entropy2

def entropy(inputList, label):
    N1 = 0
    N2 = 0
    for input in inputList:
        if label[input['id']] == 0:
            N1 = N1 + 1
        else:
            N2 = N2 + 1

    if N1==0.0 or N2==0.0:
        return 0
    else:
        p1 = N1/(N1+N2)
        p2 = N2/(N1+N2)
    return -p1*math.log(p1,2) -p2*math.log(p2,2)