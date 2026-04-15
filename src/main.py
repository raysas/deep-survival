'''
main user interface for training and evaluating the DeepSurv model
'''

from .dataset import SurvivalDataset
from .model import DeepSurv

class VanillaDeepSurv(DeepSurv):
    def __init__(self, input_dim, hidden_layers=[100], dropout=0.5):
        super().__init__(input_dim, hidden_layers, dropout)

    def forward(self, x):
        return super().forward(x)
    
epochs=100
