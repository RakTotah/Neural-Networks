import copy, math, numpy as np, random, tqdm
from activation import *

class NN:

    def __init__(self, inputNodes, outputNodes, activation : ActivationFunction, learningRate=0.1, hiddenLayers=None, hiddenLayerNodeNum=None):
        # Graph setup

        self.activation : ActivationFunction = activation

        self.representation = [ [ 0 for _ in range(inputNodes) ], [ 0 for _ in range(outputNodes) ] ]
        # self.representation[i][j] is the node representing the activation value of the jth node of the ith layer, where the 0th layer is the input layer

        self.biases = [[ 0 for _ in range(outputNodes) ]]
        # self.biases[i][j] is the bias of the jth node of the ith layer, where the 0th layer is the first hidden layer, or it is the output layer if there are no hidden layers

        if hiddenLayers and hiddenLayerNodeNum and hiddenLayerNodeNum > 0:
            for _ in range(hiddenLayers):
                self.representation.insert(-1, [ 0 for _ in range(hiddenLayerNodeNum) ])
                self.biases.insert(-1, [ 0 for _ in range(hiddenLayerNodeNum) ])

        # Weight Matrix Setup
        self.weights = []
        # self.weights[i][j][k] is the weight from the kth node of the ith layer to the jth node of the (i+1)th layer

        for i in range(1, len(self.representation)): # Layer
            now = []
            for _ in range(len(self.representation[i])): # Receiving Node
                now.append([ 1 for _ in self.representation[i-1] ])
            self.weights.append(now)

        # Learning Setup
        self.learningRate = learningRate

    def randomize(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = random.randint(-1000, 1000)/1000
        
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j] = random.randint(-1000, 1000)/1000
    
    def getResultFromInput(self, input):
        assert len(input) == len(self.representation[0])
        copied = copy.deepcopy(self.representation)
        copied[0] = input
        for layer in range(1, len(copied)):
            for node in range(len(copied[layer])):
                copied[layer][node] = self.activation.standard(sum([ copied[layer-1][x]*self.weights[layer-1][node][x] for x in range(len(copied[layer-1])) ]) + self.biases[layer-1][node])
        return copied

    def costFunction(self, input, output) -> float:
        assert len(input) == len(self.representation[0]) and len(output) == len(self.representation[-1])

        resulted = self.getResultFromInput(input)[-1]

        return sum([ (output[x] - resulted[x])**2 for x in range(len(resulted)) ])

    def deltaMatrix(self, input, desiredOutput):
        a = self.getResultFromInput(input)
        # a[i][j] is the activation value of the jth node of the ith layer of the resulting neural network after performing a forward pass due to the (input) parameter passed to this function

        assert len(desiredOutput) == len(a[-1])

        z = lambda i, j: sum([ a[i-1][c]*self.weights[i-1][j][c] for c in range(len(a[i-1])) ]) + self.biases[i-1][j]
        
        deltaNodes = [ [ 0 for _ in x] for x in a ]
        # deltaNodes[i][j] is the partial derivative of the cost function with respect to the jth node of the ith layer, where the 0th layer is the input layer

        deltaWeights = [ [ [ 0 for _ in y ] for y in x ] for x in self.weights ]
        # deltaWeights[i][j][k] is the partial derivative of the cost function with respect to the weight from the kth node of the ith layer to the jth node of the (i+1)th layer, where the 0th layer is the input layer

        deltaBiases = [ [ 0 for _ in x ] for x in self.biases ]
        # deltaBiases[i][j] is the partial derivative of the cost function with respect to the bias of the jth node of the ith layer, where the 0th layer is the first hidden layer, or the output layer if there are no hidden layers

        ### Node derivative calculation ###
        for i in range(len(deltaNodes)-1, 0, -1): # Layer
            for j in range(len(deltaNodes[i])): # Node
                if i == len(deltaNodes)-1:
                    deltaNodes[i][j] = 2*(a[i][j] - desiredOutput[j])
                else:
                    deltaNodes[i][j] = sum([ deltaNodes[i+1][c]*self.activation.derivative(z(i+1,c))*self.weights[i][c][j] for c in range(len(deltaNodes[i+1])) ])

        ### Weight derivative calculation ###
        for i in range(len(deltaWeights)):
            for j in range(len(deltaWeights[i])):
                for k in range(len(deltaWeights[i][j])):
                    deltaWeights[i][j][k] = deltaNodes[i+1][j] * self.activation.derivative(z(i+1,j))*a[i][k]
        
        ### Bias derivative calculation ###
        for i in range(len(deltaBiases)):
            for j in range(len(deltaBiases[i])):
                deltaBiases[i][j] = deltaNodes[i+1][j] * self.activation.derivative(z(i+1,j))

        return (deltaWeights, deltaBiases)

    def applyDeltaMatrix(self, deltaMatrix):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= self.learningRate*deltaMatrix[0][i][j][k]

        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j] -= self.learningRate*deltaMatrix[1][i][j]

    def train(self, inputSet, outputSet, costLimit=0.01, maxIterations=5000, printFinalCost=False, monitorCost=False, adaptLearningRate = False):
        assert len(inputSet) == len(outputSet)
        previous = averageCostFunction(self, inputSet, outputSet)
        for x in range(maxIterations):
            totalMatrix = []
            for i in tqdm.tqdm(range(len(inputSet)), leave=False):
                totalMatrix.append(self.deltaMatrix(inputSet[i], outputSet[i]))
            finalWeights = copy.deepcopy(totalMatrix[0][0])
            finalBiases = copy.deepcopy(totalMatrix[0][1])

            for i in range(len(finalWeights)):
                for j in range(len(finalWeights[i])):
                    for k in range(len(finalWeights[i][j])):
                        finalWeights[i][j][k] = sum(x[i][j][k] for x in [ y[0] for y in totalMatrix ])/len(totalMatrix)

            for i in range(len(finalBiases)):
                for j in range(len(finalBiases[i])):
                    finalBiases[i][j] = sum(x[i][j] for x in [ y[1] for y in totalMatrix ])/len(totalMatrix)

            self.applyDeltaMatrix((finalWeights, finalBiases))
            
            now = averageCostFunction(self, inputSet, outputSet)
            if adaptLearningRate and now == previous:
                self.learningRate *= 1.01
            elif adaptLearningRate and percentDifference(now, previous) <= 0.01:
                self.learningRate /= 1.01
            if now <= costLimit:
                break
            if monitorCost:
                print(now, f"\t{self.learningRate}" if adaptLearningRate else "", "\t", f"{x}/{maxIterations}")
        if printFinalCost:
            print(f"Final average cost: {now}")

    def test(self, inputSet, outputSet, testFunction : function=lambda x, y: x == y):
        assert len(inputSet) == len(outputSet)
        return sum([ 1 if testFunction(self.getResultFromInput(inputSet[x]), outputSet[x]) else 0 for x in range(len(inputSet)) ])/len(inputSet)

def percentDifference(calculated, expected):
    return abs(expected-calculated)/expected

def averageCostFunction(neuralNetwork : NN, inputSet : list[float], outputSet : list[float]):
    assert len(inputSet) == len(outputSet)
    return sum(neuralNetwork.costFunction(inputSet[x], outputSet[x]) for x in range(len(inputSet)))/len(inputSet)