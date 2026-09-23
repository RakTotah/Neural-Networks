import java.util.Arrays;

public class NN {

    // Relatively useful activation functions, though custom ones may be used as long as they are:
    //      1. Defined for the continuous spectrum of real numbers.
    //      2. Differentiable.
    public static activationFunction sigmoid = new activationFunction(x -> 1/(1+Math.exp(-x)) , x -> x);
    public static activationFunction reLU = new activationFunction((x) -> x>0?x:0, x -> x>0?1d:0);
    /**
     * A matrix to represent the nodes of the network, used purely as a structural component, with no calculations using it. <br>
     * {@code representation[i][j]} is the activation value of the jth node of the ith layer.
     */
    private double[][] representation = null;
    /**
     * The weight matrix of the neural network. <br>
     * {@code weights[i][j][k]} is the weight from the jth node of the ith layer to the kth node of the (i+1)th layer, where the 0th layer is the input layer.
     */
    public double[][][] weights = null;
    /**
     * The bias matrix of the neural network. <br>
     * {@code biases[i][j]} is the bias of the jth node of the ith layer, where the 0th layer is the first layer after the input layer.
     */
    public double[][] biases = null;
    public double learningRate = 0.01f;
    private activationFunction activationFunction;

    public NN(int inputNodes, int outputNodes, int hLNum, int hLNodeNum, double learningRate, activationFunction activationFunction){
        this.learningRate = learningRate;
        this.representation = new double[hLNum+2][];
        this.weights = new double[hLNum+1][][];
        this.biases = new double[hLNum+1][];
        this.activationFunction = activationFunction;
        this.initRepresentation(inputNodes, outputNodes, hLNum, hLNodeNum);
        this.initBiases(outputNodes, hLNum, hLNodeNum);
        this.initWeights();
    }

    /**
     * Sets up the weight matrix of the neural network. Or if need be, resets the weight values
     * to their initial state if the matrix was modified.
     * @param inputNodes The number of input nodes.
     * @param outputNodes The number of output nodes.
     * @param hLNum The number of hidden layers.
     * @param hLNodeNum The number of nodes in each hidden layer.
     */
    private void initWeights(){
        int i = 0;
        while (i < this.weights.length){
            this.weights[i] = new double[this.representation[i].length][];
            for (int j = 0; j < this.representation[i].length; j++){
                this.weights[i][j] = new double[this.representation[i+1].length];
            }
            i++;
        }
    }

    /**
     * Sets up the bias matrix of the neural network. Or if need be, resets the bias values
     * to their initial state if the matrix was modified.
     * @param outputNodes The number of output nodes.
     * @param hLNum The number of hidden layers.
     * @param hLNodeNum The number of nodes in each hidden layer.
     */
    private void initBiases(int outputNodes, int hLNum, int hLNodeNum){
        int i = 0;
        // Hidden layer setup
        while (i < hLNum){
            this.biases[i] = new double[hLNodeNum];
            i++;
        }
        // Output layer setup
        this.biases[i] = new double[outputNodes];
    }

    /**
     * Sets up the representation of the neural network. Or if need be, resets the representation values
     * to their initial state if the representation was modified.
     * @param inputNodes The number of input nodes.
     * @param outputNodes The number of output nodes.
     * @param hLNum The number of hidden layers.
     * @param hLNodeNum The number of nodes in each hidden layer.
     */
    private void initRepresentation(int inputNodes, int outputNodes, int hLNum, int hLNodeNum){
        int i = 0;
        // Input layer setup
        this.representation[i] = new double[inputNodes];
        i++;
        // Hidden layer setup
        while (i <= hLNum){
            this.representation[i] = new double[hLNodeNum];
            i++;
        }
        // Output layer setup
        this.representation[i] = new double[outputNodes];
    }

    /**
     * Randomizes all weight and bias values of the neural network to a random number 
     * between -1 and 1 inclusive.
     */
    public void randomize(){

        // Weights
        for (double[][] e : this.weights){
            for (double[] f : e){
                for (int i = 0; i < f.length; i++){
                    f[i] = Math.random() >= 0.5f ? (double)Math.random() : -1 * (double)Math.random();
                }
            }
        }
        // Biases
        for (double[] e : this.biases) {
            for (int i = 0; i < e.length; i++) {
                e[i] = Math.random() >= 0.5f ? (double)Math.random() : -1 * (double)Math.random();
            }
        }
    }

    /**
     * @param inputLst An array of type {@code double[]}.
     * @return The sum of the elements of this array.
     */
    private double sum(double[] inputLst){
        double result = 0;
        for (double e : inputLst){
            result += e;
        }
        return result;
    }

    /**
     * Performs a forward pass on the neural network by considering the input passed to the function as
     * the first layer of the neural network. <h3>Math: </h1>It does this for a given node {@code n} by 
     * considering the activations of all the nodes in the layer prior to {@code n}'s layer multiplied 
     * by the weights of their connections to {@code n}. Once this is all summed up, {@code n}'s bias is added
     * and this final sum is passed as an argument to the chosen activation function of the neural network.'
     * @param input The input to the neural network.
     * @return A 3-dimensional array containing resulting activations of nodes as well as their values
     * before being passed into the neural network's activation function. To better visualize this,
     * look at the resulting 3-dimensional array as a 2-dimensional array, with each element being a
     * tuple-like data structure, where {@code result[i][j]} is a 2-element array, with {@code result[i][j][0]}
     * being the activation of the jth node of the ith layer and {@code result[i][j][1]} is the initial
     * value of this activation before being passed into the activation function. This is very useful when
     * backpropagating.<br>
     * 
     * <h3>Note:</h3>If needed, the complexity of the base neural network forward pass math
     * can be increased easily by modifying the below code such that the 2-element "tuple" storing the
     * activation and its initial value can be expanded to fit 3 or more things to keep track of.
     */
    public double[][][] forwardPass(double[] input){

        // Temp setup
        double[][][] temp = new double[this.representation.length][][];
        for (int i = 0; i < this.representation.length; i++){
            temp[i] = new double[this.representation[i].length][2];
        }

        assert input.length == this.representation[0].length;
        for (int i = 0; i < input.length; i++){
            temp[0][i][0] = input[i];
        }

        for (int layer = 0; layer < temp.length-1; layer++){
            for (int i = 0; i < temp[layer+1].length; i++){
                double result = this.biases[layer][i];
                for (int e = 0; e < temp[layer].length; e++){
                    result += this.weights[layer][e][i] * temp[layer][e][0];
                }
                temp[layer+1][i][1] = result;
                temp[layer+1][i][0] = this.activationFunction.standard(result);
            }
        }
        return temp;
    }

    /**
     * Considers the input array to be the first layer of the neural network, then performs
     * a forward pass and returns the result.
     * @param input The input to the neural network.
     * @return The output layer of the neural network after the forward pass is completed.
     */
    public double[] getAnswer(double[] input){
        assert input.length == this.representation[0].length;
        double[] result = new double[this.representation[this.representation.length-1].length];
        double[][][] temp = this.forwardPass(input);

        for (int i = 0; i < result.length; i++){
            result[i] = temp[result.length][i][0];
        }
        return result;
    }

    /**
     * Prints out the weights, biases, and representation structure of the neural
     * network in that order.
     */
    public void debug(){
        System.out.println("Weights:");
        for (double[][] e : this.weights){
            for (double[] f : e){
                System.out.print(Arrays.toString(f));
                System.out.print(" ");
            }
            System.out.println();
        }
        System.out.println("Biases:");
        for (double[] e : this.biases){
            System.out.println(Arrays.toString(e));
        }
        System.out.println("Representation:");
        for (double[] e : this.representation){
            System.out.println(Arrays.toString(e));
        }
    }
}