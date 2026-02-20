
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    # you pass a list of layers and will apply them in order to the input x.
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss, y_train)
        y = self.layers[-1]
        dL_dy = loss.backward(y, y_train)
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)