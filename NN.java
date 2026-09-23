import java.util.ArrayList;

public class NN {
    private ArrayList<ArrayList<Float>> representation = new ArrayList<>();
    private ArrayList<ArrayList<ArrayList<Float>>> weights = new ArrayList<>();
    private ArrayList<ArrayList<Float>> biases = new ArrayList<>();
    public Float learningRate = 0.01f;

    public NN(int inputNodes, int outputNodes, int hLNum, int hLNodeNum, Float learningRate){
        this.learningRate = learningRate;
        this.setupRepresentation(inputNodes, outputNodes, hLNum, hLNodeNum);
        this.setupBiases(outputNodes, hLNum, hLNodeNum);
        this.setupWeights();
    }

    private void setupWeights(){
        ArrayList<Float> tempInner = new ArrayList<>();
        ArrayList<ArrayList<Float>> tempOuter = new ArrayList<>();

        int i = 0;
        while (i < this.representation.size() - 1){
            tempOuter = new ArrayList<>();
            for (int j = 0; j < this.representation.get(i).size(); j++){
                tempInner = new ArrayList<>();
                for (int k = 0; k < this.representation.get(i+1).size(); k++){
                    tempInner.add(0f);
                }
                tempOuter.add(new ArrayList<>(tempInner));
            }
            this.weights.add(new ArrayList<>(tempOuter));
            // System.out.println(String.format("%d, %d", i, i+1));
            i++;
        }
    }

    private void setupBiases(int outputNodes, int hLNum, int hLNodeNum){
        ArrayList<Float> temp = new ArrayList<>();

        // Hidden layers
        for (int i = 0; i < hLNum; i++){
            temp.clear();
            for (int j = 0; j < hLNodeNum; j++){
                temp.add(0f);
            }
            this.biases.add(new ArrayList<>(temp));
        }

        // Output layer
        temp.clear();
        for (int i = 0; i < outputNodes; i++){
            temp.add(0f);
        }
        this.biases.add(new ArrayList<>(temp));
    }
    private void setupRepresentation(int inputNodes, int outputNodes, int hLNum, int hLNodeNum){
        // Input layer setup
        ArrayList<Float> temp = new ArrayList<>();
        for (int i = 0; i < inputNodes; i++){
            temp.add(0f);
        }
        this.representation.add(new ArrayList<>(temp));

        // Hidden layer setup
        for (int i = 0; i < hLNum; i++){
            temp.clear();
            for (int j = 0; j < hLNodeNum; j++){
                temp.add(0f);
            }
            this.representation.add(new ArrayList<>(temp));
        }

        // Output layer setup
        temp.clear();
        for (int i = 0; i < outputNodes; i++){
            temp.add(0f);
        }
        this.representation.add(new ArrayList<>(temp));
    }

    private void randomize(){

        // Weights
        for (int i = 0; i < this.weights.size(); i++){
            for (int j = 0; i < this.weights.get(i).size(); j++){
                for (int k = 0; i < this.weights.get(i).get(j).size(); k++){
                    if (Math.random() >= 0.5f){
                        this.weights.get(i).get(j).set(k, (float)Math.random());
                    }
                    else{
                        this.weights.get(i).get(j).set(k, -1 * (float)Math.random());
                    }
                }
            }
        }

        // Biases
        for (int i = 0; i < this.biases.size(); i++){
            for (int j = 0; i < this.biases.get(i).size(); j++){
                if (Math.random() >= 0.5f){
                        this.biases.get(i).set(j, (float)Math.random());
                    }
                    else{
                        this.biases.get(i).set(j, -1 * (float)Math.random());
                    }
            }
        }
    }

    public ArrayList<Float> deltaMatrix(ArrayList<ArrayList<Float>> inputLst){
        return new ArrayList<>();
    }
}