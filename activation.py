import math

class ActivationFunction:
    def __init__(self, standard, derivative):
        self.standard : function = standard
        self.derivative : function = derivative

reLU = ActivationFunction(lambda x: x if x > 0 else 0, lambda x: 1 if x > 0 else 0)
sigmoid = ActivationFunction(lambda x: 1/(1+math.e**(-x)), lambda x: sigmoid.standard(x)*(1-sigmoid.standard(x)))