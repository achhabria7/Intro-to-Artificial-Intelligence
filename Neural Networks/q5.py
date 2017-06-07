from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from Testing import testPenData, testCarData, average, stDeviation

accuracyPenData =[]
accuracyCarData = []

for i in range(5):
	nnet, accuracy = testPenData()
	accuracyPenData.append(accuracy)
	nnet, accuracyCar = testCarData()
	accuracyCarData.append(accuracyCar)

print("Pen Data Average Accuracy: ", average(accuracyPenData))
print("Pen Data Max Accuracy: ", max(accuracyPenData))
print("Pen Data St Dev Accuracy: ", stDeviation(accuracyPenData))
print(accuracyPenData)

print("Car Data Average Accuracy: ", average(accuracyCarData))
print("Car Data Max Accuracy: ", max(accuracyCarData))
print("Car Data St Dev Accuracy: ", stDeviation(accuracyCarData))
print(accuracyCarData)