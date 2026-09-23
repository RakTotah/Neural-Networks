import java.util.Arrays;

public class NN {

    // Relatively useful activation functions, though custom ones may be used as long as they are:
    //      1. Defined for the continuous spectrum of real numbers.
    //      2. Differentiable.
    public static activationFunction sigmoid = new activationFunction(x -> 1/(1+(double)Math.exp(-x)) , x -> x);
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

    private double sum(double[] inputLst){
        double result = 0;
        for (double e : inputLst){
            result += e;
        }
        return result;
    }

    public double[][] forwardPass(double[] input){

        // Temp setup
        double[][] temp = new double[this.representation.length][];
        for (int i = 0; i < this.representation.length; i++){
            temp[i] = new double[this.representation[i].length];
        }
        temp[0] = input;
        assert input.length == temp[0].length;
        
        for (int layer = 0; layer < temp.length-1; layer++){
            for (int i = 0; i < temp[layer+1].length; i++){
                double result = this.biases[layer][i];
                for (int e = 0; e < temp[layer].length; e++){
                    result += this.weights[layer][e][i] * temp[layer][e];
                }
                temp[layer+1][i] = this.activationFunction.standard(result);
            }
        }

        this.representation = temp;
        return temp;
    }

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