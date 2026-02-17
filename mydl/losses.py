import torch

class Loss: 
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        raise NotImplementedError("Forward method not implemented")
    

class MSE(Loss):

    def __init__(self):
        super().__init__()

    # __call__ allows us to call the instance of the class as a function.
    def __call__(self, y_pred, y_true):
        return torch.mean( (y_pred - y_true)**2 )