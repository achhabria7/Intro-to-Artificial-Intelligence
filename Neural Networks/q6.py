from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from Testing import average, stDeviation, testPenData, testCarData
import cPickle 
from math import pow, sqrt


print("Car Data")
carData = buildExamplesFromCarData()
for numPerceptron in range(0, 41, 5):
	accuracyCarData = []
	for i in range(5):
		print("Iteration:", i, "Num of Perceptron:", numPerceptron)
		nnet, accuracyCar = buildNeuralNet(carData, maxItr = 200, hiddenLayerList = [numPerceptron])
		accuracyCarData.append(accuracyCar)
	print numPerceptron, ',', max(accuracyCarData), ',', average(accuracyCarData), ',', stDeviation(accuracyCarData)

print("Pen Data")
penData = buildExamplesFromPenData() 
for numPerceptron in range(0, 41, 5):
	accuracyPenData =[]
	for i in range(5):
		print("Iteration:", i, "Num of Perceptron:", numPerceptron)
		nnet, accuracyPen = buildNeuralNet(penData, maxItr = 200, hiddenLayerList = [numPerceptron])
		accuracyPenData.append(accuracyPen)
	print numPerceptron, ',',max(accuracyPenData), ',',average(accuracyPenData), ',',stDeviation(accuracyPenData)
