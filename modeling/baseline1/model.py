import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResnetClassifier(nn.Module):
    """
    A ResNet50-based classifier for group activity recognition.

    Args:
        num_cls (int): Number of output classes.
        freeze_backbone (bool, optional): If True, freeze backbone weights. Defaults to False.
        freeze_up_to (int, optional): If `freeze_backbone` = True, specify number of last block to freeze. Defaults to 2.
    """
    def __init__(self, num_cls: int, freeze_backbone: bool = False, freeze_up_to: int = 2):
        super().__init__()
        self.freeze_up_to = freeze_up_to
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        if freeze_backbone:
            layers_to_freeze = self._get_freezed_layers()
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_cls)
    
    def _get_freezed_layers(self):
        backbone = [
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        ]
        
        freeze_up_to = min(self.freeze_up_to, 4)
        layers = backbone[:freeze_up_to]
    
        layers = [self.model.conv1, self.model.bn1] + layers
        return layers
        
    def forward(self, x):
        return self.model(x)
