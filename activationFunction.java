import java.util.function.Function;

public class activationFunction {
    private final Function<Double, Double> standardFunc;
    private final Function<Double, Double> derivativeFunc;

    public activationFunction(Function<Double,Double> standard, Function<Double,Double> derivative){
        this.standardFunc = standard;
        this.derivativeFunc = derivative;
    }

    public double standard(double input){
        return this.standardFunc.apply(input);
    }

    public double derivative(double input){
        return this.derivativeFunc.apply(input);
    }
}