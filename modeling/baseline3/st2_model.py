import torch
import torch.nn as nn

class nnClassifier(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: list, p_dropout: float, num_classes: int = 8):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_sz, hidden_sz[0]))
        layers.append(nn.BatchNorm1d(hidden_sz[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout1d(p=p_dropout))
        
        for i in range(len(hidden_sz) - 1):
            layers.append(nn.Linear(hidden_sz[i], hidden_sz[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sz[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_sz[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
