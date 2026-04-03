import torch
import torch.nn as nn

class TemporalClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_state_size: int,
        num_layers: int,
        linear_layer_size: int,
        bidirectional: bool,
        num_classes: int = 8
    ):
        super().__init__()
        self.bidir = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_state_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        linear_input = hidden_state_size * (bidirectional + 1)

        self.linear = nn.Sequential(
            nn.Linear(linear_input, linear_layer_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(linear_layer_size, num_classes)
        )
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)

        if self.bidir:
            h_forward = h[-2]
            h_backward = h[-1]
            h_final = torch.cat((h_forward, h_backward), dim=1)
        else:
            h_final = h[-1]
        
        logits = self.linear(h_final)
        return logits

