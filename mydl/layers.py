import torch

# The layer class is the mother class for all layers. It defines the basic structure and functionality that all layers will inherit.
class Layer:

    def __init__(self):
        self.parameters = {}

    def forward(self, x):
        raise NotImplementedError('Forward method not implemented.')


class Linear(Layer):

    # fan_in: number of input features
    # fan_out: number of output features
    def __init__(self, fan_in, fan_out):

        # super() used because the class Linear calls the __init__ method of the parent class Layer. Thus Linear inherits the properties of Layer. 
        super().__init__()

        # dtype = torch.float32 sets the data type to 32-bit float.
        # requires_grad = False indicates that the parameters will not be updated during training, we will do the training. 
        # device = cuda specifies that the parameters should be stored on the GPU for faster computation.
        self.parameters['W'] = torch.randn((fan_in, fan_out), dtype = torch.float32, requires_grad = False, device = 'cpu')
        self.parameters['b'] = torch.zeros((1,fan_out), dtype = torch.float32, requires_grad = False)
        self.n_parameters = fan_in * fan_out + fan_out

    def forward(self, x):
        return x @ self.parameters['W'] + self.parameters['b']
    

# We define the sigmoid activation function as a layer.
class Sigmoid(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))