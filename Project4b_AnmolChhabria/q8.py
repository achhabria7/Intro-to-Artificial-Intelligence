from NeuralNetUtil import buildExamplesFromExtraData
from NeuralNet import buildNeuralNet
from Testing import testPenData, testCarData, average, stDeviation


extraData = buildExamplesFromExtraData()
accuracyExtraData = []
for i in range(5):
	print("Iteration ", i)
	nnet, accuracy = buildNeuralNet(extraData, maxItr = 200, hiddenLayerList = [8])
	accuracyExtraData.append(accuracy)
print max(accuracyExtraData), ',', average(accuracyExtraData), ',', stDeviation(accuracyExtraData)