import torch
import torch.nn as nn

class MultiHeadClassifier(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        lstm_specs = model_cfg['lstm_specs']
        player_head = model_cfg['player_head']
        group_head = model_cfg['group_head']
        
        self.input_size = lstm_specs['input_size']
        self.bidir = lstm_specs['bidirectional']
        
        self.lstm = nn.LSTM(
            input_size=lstm_specs['input_size'],
            hidden_size=lstm_specs['hidden_state_size'],
            num_layers=lstm_specs['num_layers'],
            bidirectional=self.bidir,
            batch_first=True
        )
        
        self.lstm_output_size = lstm_specs['hidden_state_size'] * (self.bidir + 1)
        self.p_head = nn.Sequential(
            nn.Linear(self.lstm_output_size, player_head['linear_layer_size']),
            nn.BatchNorm1d(player_head['linear_layer_size']),
            nn.ReLU(),
            nn.Dropout(player_head['p_dropout']),
            nn.Linear(player_head['linear_layer_size'], player_head['num_classes'])
        )
        
        self._setup_group_head(group_head)
        
    def forward(self, x, sizes):
        _, (h, _) = self.lstm(x)

        if self.bidir:
            h_forward = h[-2]
            h_backward = h[-1]
            h_final = torch.cat((h_forward, h_backward), dim=1) 
        else:
            h_final = h[-1]
        
        p_logits = self.p_head(h_final)
        
        m = x.size(1) // 2
        x = torch.hstack((h_final, x[: , m, :]))
        
        features = torch.split(x, sizes)
        pooled_f = torch.stack(list(map(lambda x: torch.mean(x, dim=0), features)))
        g_logits = self.g_head(pooled_f)
        
        return p_logits, g_logits
        
    def _setup_group_head(self, config: dict):
        input_size = self.lstm_output_size + self.input_size
        layer_sizes = config['hidden_layer_sizes']
        
        layers = [
            nn.Linear(input_size, layer_sizes[0]),
            nn.BatchNorm1d(layer_sizes[0]),
            nn.ReLU(),
            nn.Dropout1d(p=config['p_dropout'])
        ]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(layer_sizes[-1], config['num_classes']))
        self.g_head = nn.Sequential(*layers)
        