import java.util.function.Function;

public class activationFunction {
    private final Function<Float, Float> standardFunc;
    private final Function<Float, Float> derivativeFunc;

    public activationFunction(Function<Float,Float> standard, Function<Float,Float> derivative){
        this.standardFunc = standard;
        this.derivativeFunc = derivative;
    }

    public float standard(float input){
        return this.standardFunc.apply(input);
    }

    public float derivative(float input){
        return this.derivativeFunc.apply(input);
    }
}