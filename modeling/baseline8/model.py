import torch
import torch.nn as nn

class ResnetLSTMClassifier(nn.Module):
    def __init__(self, config: dict, backbone, imgs_per_clip: int, extraction_mode: bool):
        super().__init__()
        lstm_specs = config['lstm_specs']
        head_cfg = config['linear_head']

        backbone.model.fc = nn.Identity()
        
        self.backbone = backbone
        self.imgs_per_clip = imgs_per_clip
        self.extraction_mode = extraction_mode
        
        self.bidir = lstm_specs['bidirectional']
        self.lstm_input_size = lstm_specs['input_size']
        
        self.lstm = nn.LSTM(
            input_size=lstm_specs['input_size'],
            hidden_size=lstm_specs['hidden_state_size'],
            num_layers=lstm_specs['num_layers'],
            bidirectional=lstm_specs['bidirectional'],
            batch_first=True
        )
        
        self.lstm_out_size = lstm_specs['hidden_state_size'] * (lstm_specs['bidirectional'] + 1)
        self.fc = ClassifierHead(head_cfg, self.lstm_out_size)
        
    def forward(self, x): 
        x = self.backbone(x)
        x = torch.stack(torch.split(x, self.imgs_per_clip))
        
        if self.extraction_mode:
            output, _ = self.lstm(x)
            output = torch.cat((output, x), dim=2)
            return output
            
        _, (h, _) = self.lstm(x)
        
        h_final = torch.hstack((h[-2], h[-1])) if self.bidir else h[-1]
        logits = self.fc(h_final)

        return logits
    
    @property
    def get_output_size(self):
        return self.lstm_out_size + self.lstm_input_size

class GroupLSTMClassifier(nn.Module):
    def __init__(self, config: dict, input_size: int):
        super().__init__()
        lstm_specs = config['lstm_specs']
        head_cfg = config['linear_head']

        self.bidir = lstm_specs['bidirectional']
        
        self.lstm_block = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.LSTM(
                input_size=input_size,
                hidden_size=lstm_specs['hidden_state_size'],
                num_layers=lstm_specs['num_layers'],
                bidirectional=lstm_specs['bidirectional'],
                batch_first=True
            )
        )
        head_input_size = lstm_specs['hidden_state_size'] * (lstm_specs['bidirectional'] + 1)
        self.fc = ClassifierHead(head_cfg, head_input_size)
        
    def forward(self, x):
        _, (h, _) = self.lstm_block(x)
        h_final = torch.hstack((h[-2], h[-1])) if self.bidir else h[-1]
        logits = self.fc(h_final)
        return logits

class ClassifierHead(nn.Module):
    def __init__(self, config: dict, input_size: int):
        super().__init__()
        layer_sizes = [input_size] + config['hidden_layer_sizes']
        p = config['p_dropout']
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU())
            if p[i] != 0:
                layers.append(nn.Dropout1d(p=p[i]))
        
        layers.append(nn.Linear(layer_sizes[-1], config['num_classes']))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)