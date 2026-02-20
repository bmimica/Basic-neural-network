
class Sequential:
    def __init__(self, layers):
        self.layers = layers

        # counts all parameters
        self.n_parameters = 0
        for i, layer in enumerate(layers):
            self.n_parameters += layer.parameters

    # you pass a list of layers and will apply them in order to the input x.
    def forward(self, x):
        
        # applies all layer forward methods in order
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss, y_train):

        # last output : 
        y = self.layers[-1]

        # derivative of the loss with respect to the last : 
        dL_dy = loss.backward(y, y_train)
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)