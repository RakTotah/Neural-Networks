import copy, math

def sigmoid(x : float) -> float:
    return 1/(1+math.e**-x)

def derivaSigmoid(x : float) -> float:
    return sigmoid(x)*(1-sigmoid(x))

def inverSigmoid(x : float) -> float:
    return math.log(x/(1-x))

class NN:

    def __init__(self, inputNodes, outputNodes, learningRate=0.1, hiddenLayers=None, hiddenLayerNodeNum=None):
        # Graph setup

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
        
    def getResultFromInput(self, input):
        assert len(input) == len(self.representation[0])
        copied = copy.deepcopy(self.representation)
        copied[0] = input
        for layer in range(1, len(copied)):
            for node in range(len(copied[layer])):
                copied[layer][node] = sigmoid(sum([ copied[layer-1][x]*self.weights[layer-1][node][x] for x in range(len(copied[layer-1])) ]) + self.biases[layer-1][node])
        return copied

    def costFunction(self, input, output) -> float:
        assert len(input) == len(self.representation[0]) and len(output) == len(self.representation[-1])

        resulted = self.getResultFromInput(input)[-1]

        return sum([ (output[x] - resulted[x])**2 for x in range(len(resulted)) ])

    def deltaMatrix(self, input, desiredOutput):
        a = self.getResultFromInput(input)
        # a[i][j] is the activation value of the jth node of the ith layer of the resulting neural network after performing a forward pass due to the (input) parameter passed to this function

        assert len(desiredOutput) == len(a[-1])

        
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
                    deltaNodes[i][j] = sum([ deltaNodes[i+1][c]*derivaSigmoid(inverSigmoid(a[i+1][c]))*self.weights[i][c][j] for c in range(len(deltaNodes[i+1])) ])

        ### Weight derivative calculation ###
        for i in range(len(deltaWeights)):
            for j in range(len(deltaWeights[i])):
                for k in range(len(deltaWeights[i][j])):
                    deltaWeights[i][j][k] = deltaNodes[i+1][j] * derivaSigmoid(inverSigmoid(a[i+1][j]))*a[i][k]
        
        ### Bias derivative calculation ###
        for i in range(len(deltaBiases)):
            for j in range(len(deltaBiases[i])):
                deltaBiases[i][j] = deltaNodes[i+1][j] * derivaSigmoid(inverSigmoid(a[i+1][j]))

        return (deltaWeights, deltaBiases)

    def applyDeltaMatrix(self, deltaMatrix):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= self.learningRate*deltaMatrix[0][i][j][k]

        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j] -= self.learningRate*deltaMatrix[1][i][j]
    
def averageCostFunction(neuralNetwork : NN, inputSet : list[float], outputSet : list[float]):
    assert len(inputSet) == len(outputSet)
    return sum(neuralNetwork.costFunction(inputSet[x], outputSet[x]) for x in range(len(inputSet)))/len(inputSet)

