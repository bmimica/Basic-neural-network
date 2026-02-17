import torch

class Loss: 
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        raise NotImplementedError("Forward method not implemented")
    

class MSE(Loss):

    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true):
        return torch.mean( (y_pred - y_true)**2 )