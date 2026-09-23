import java.util.ArrayList;
import java.util.Arrays;

public class NN {

    // Relatively useful activation functions, though custom ones may be used as long as they are:
    //      1. Defined for the continuous spectrum of real numbers.
    //      2. Differentiable.
    public static activationFunction sigmoid = new activationFunction(x -> 1/(1+(float)Math.exp(-x)) , x -> x);
    public static activationFunction reLU = new activationFunction((x) -> x>0?x:0, x -> x>0?1f:0);
    /**
     * A matrix to represent the nodes of the network, used purely as a structural component, with no calculations using it. <br>
     * {@code representation[i][j]} is the activation value of the jth node of the ith layer.
     */
    private float[][] representation = null;
    /**
     * The weight matrix of the neural network. <br>
     * {@code weights[i][j][k]} is the weight from the jth node of the ith layer to the kth node of the (i+1)th layer, where the 0th layer is the input layer.
     */
    public float[][][] weights = null;
    /**
     * The bias matrix of the neural network. <br>
     * {@code biases[i][j]} is the bias of the jth node of the ith layer, where the 0th layer is the first layer after the input layer.
     */
    public float[][] biases = null;
    public float learningRate = 0.01f;
    private activationFunction activationFunction;

    public NN(int inputNodes, int outputNodes, int hLNum, int hLNodeNum, float learningRate, activationFunction activationFunction){
        this.learningRate = learningRate;
        this.representation = new float[hLNum+2][];
        this.weights = new float[hLNum+1][][];
        this.biases = new float[hLNum+1][];
        this.activationFunction = activationFunction;
        this.initRepresentation(inputNodes, outputNodes, hLNum, hLNodeNum);
        this.initBiases(outputNodes, hLNum, hLNodeNum);
        this.initWeights();
    }

    private void initWeights(){
        int i = 0;
        while (i < this.weights.length){
            this.weights[i] = new float[this.representation[i].length][];
            for (int j = 0; j < this.representation[i].length; j++){
                this.weights[i][j] = new float[this.representation[i+1].length];
            }
            i++;
        }
    }

    private void initBiases(int outputNodes, int hLNum, int hLNodeNum){
        int i = 0;
        // Hidden layer setup
        while (i < hLNum){
            this.biases[i] = new float[hLNodeNum];
            i++;
        }
        // Output layer setup
        this.biases[i] = new float[outputNodes];
    }
    private void initRepresentation(int inputNodes, int outputNodes, int hLNum, int hLNodeNum){
        int i = 0;
        // Input layer setup
        this.representation[i] = new float[inputNodes];
        i++;
        // Hidden layer setup
        while (i <= hLNum){
            this.representation[i] = new float[hLNodeNum];
            i++;
        }
        // Output layer setup
        this.representation[i] = new float[outputNodes];
    }

    public void randomize(){

        // Weights
        for (float[][] e : this.weights){
            for (float[] f : e){
                for (int i = 0; i < f.length; i++){
                    f[i] = Math.random() >= 0.5f ? (float)Math.random() : -1 * (float)Math.random();
                }
            }
        }
        // Biases
        for (float[] e : this.biases) {
            for (int i = 0; i < e.length; i++) {
                e[i] = Math.random() >= 0.5f ? (float)Math.random() : -1 * (float)Math.random();
            }
        }
    }

    private float sum(float[] inputLst){
        float result = 0;
        for (float e : inputLst){
            result += e;
        }
        return result;
    }

    public float[][] forwardPass(float[] input){

        // Temp setup
        float[][] temp = new float[this.representation.length][];
        for (int i = 0; i < this.representation.length; i++){
            temp[i] = new float[this.representation[i].length];
        }
        temp[0] = input;
        assert input.length == temp[0].length;
        
        for (int layer = 0; layer < temp.length-1; layer++){
            for (int i = 0; i < temp[layer+1].length; i++){
                temp[layer+1][i] = sum(temp[layer]);
            }
        }
        this.representation = temp;
        return temp;
    }

    public void debug(){
        System.out.println("Weights:");
        for (float[][] e : this.weights){
            for (float[] f : e){
                System.out.print(Arrays.toString(f));
                System.out.print(" ");
            }
            System.out.println();
        }
        System.out.println("Biases:");
        for (float[] e : this.biases){
            System.out.println(Arrays.toString(e));
        }
        System.out.println("Representation:");
        for (float[] e : this.representation){
            System.out.println(Arrays.toString(e));
        }
    }

    public ArrayList<Float> deltaMatrix(ArrayList<ArrayList<Float>> inputLst){
        return new ArrayList<>();
    }
}