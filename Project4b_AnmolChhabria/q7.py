from NeuralNet import buildNeuralNet
from NeuralNetUtil import buildExamplesFromXORData
from Testing import average, stDeviation
import random


XORData = buildExamplesFromXORData()
print(XORData)
for numPerceptron in range(0, 100, 1):
    accuracyXORData = []
    for i in range(5):
        nnet, accuracy = buildNeuralNet(XORData, maxItr = 200, hiddenLayerList = [numPerceptron])
        accuracyXORData.append(accuracy)
    print numPerceptron,",",max(accuracyXORData),",",average(accuracyXORData),",",stDeviation(accuracyXORData)
    #print accuracyXORData
    if average(accuracyXORData) == 1.0:
    	break

