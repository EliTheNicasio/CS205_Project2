import numpy as np


def leaveOneOutCrossValid(dataset, currentSet, featToAdd, method):

    numClassifiedCorrectly = 0

    featuresToCheck = [] + currentSet
    featuresToCheck.append(0)
    print(currentSet, featToAdd)

    if method == 0:
        featuresToCheck.append(featToAdd)
    elif method == 1:
        featuresToCheck.remove(featToAdd)

    data = np.zeros(dataset.shape)
    data[:, 0] = dataset[:, 0]
    data[:, featuresToCheck] = dataset[:, featuresToCheck]

    for i in range(data.shape[0]):
        objToClassify = data[i, 1:]
        labelObjToClassify = data[i, 0]

        nearestNeighborDist = np.inf
        nearestNeighborLoc = np.inf
        nearestNeighborLabel = -1

        for j in range(data.shape[0]):
            if j != i:
                dist = np.linalg.norm(objToClassify - data[j, 1:])
                if dist < nearestNeighborDist:
                    nearestNeighborDist = dist
                    nearestNeighborLoc = j
                    nearestNeighborLabel = data[j, 0]

        if labelObjToClassify == nearestNeighborLabel:
            numClassifiedCorrectly += 1

    return numClassifiedCorrectly / data.shape[0]


# For method, 0 is forward, 1 is backward
def featureSearch(data, method):

    if method == 0:
        currentSetOfFeatures = []
    elif method == 1:
        currentSetOfFeatures = [*range(1, data.shape[1])]

    bestOverallAcc = 0
    bestFeatures = []

    for i in range(1, data.shape[1]):
        level = i
        if method == 1:
            level = data.shape[1] - i

        print("on the " + str(level) + "th level of the search tree")
        featureToAdd = []
        bestAcc = 0

        for j in range(1, data.shape[1]):
            # print(j, currentSetOfFeatures)
            if (method == 0 and j not in currentSetOfFeatures) or (method == 1 and j in currentSetOfFeatures):
                print("--Considering adding the " + str(j) + "th feature")
                acc = leaveOneOutCrossValid(data, currentSetOfFeatures, j, method)
                print("Accuracy: " + str(acc))

                if acc > bestAcc:
                    bestAcc = acc
                    # print(currentSetOfFeatures, j)
                    featureToAdd = j

        if method == 0:
            currentSetOfFeatures.append(featureToAdd)
        elif method == 1:
            currentSetOfFeatures.remove(featureToAdd)

        print("On level " + str(level) + " I added feature " + str(featureToAdd) + " to current set")
        print("Accuracy: " + str(bestAcc))
        print("Feature Set: " + str(currentSetOfFeatures))

        if bestAcc > bestOverallAcc:
            bestOverallAcc = bestAcc
            bestFeatures = [] + currentSetOfFeatures

    return bestOverallAcc, bestFeatures


def main():
    dataSmall = np.loadtxt('CS205_small_testdata__10.txt', dtype=np.single)
    # dataLarge = np.loadtxt('CS205_large_testdata__45', dtype=np.single)
    acc, feat = featureSearch(dataSmall, 1)

    print("DONE. Best feature subset is " + str(feat) + " with an accuracy of " + str(acc))


if __name__ == '__main__':
    main()
