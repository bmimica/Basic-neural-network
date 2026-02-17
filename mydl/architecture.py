
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    # you pass a list of layers and will apply them in order to the input x.
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x