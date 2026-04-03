import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss focuses training on hard examples by down-weighting easy examples.

    Args:
        gamma (float, optional): Focusing parameter >= 0, increasing importance of hard examples. Defaults to 3.0.
        alpha (torch.Tensor, optional): Optional 1D tensor of class weights. Defaults to None.
        reduction (str, optional): Reduction method 'none', 'mean', or 'sum'. Defaults to 'mean'.
    """
    def __init__(self, gamma: float = 3.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        CEloss = F.cross_entropy(logits, targets, reduction='none')

        props = torch.exp(-CEloss)
        focal_term = (1 - props) ** self.gamma
        loss = focal_term * CEloss
            
        if self.alpha is not None:
            alpha_t = self.alpha[targets] 
            loss *= alpha_t
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def __repr__(self):
        return f"FocalLoss(gamma: {self.gamma}, alpha: {self.alpha}, reduction: {self.reduction})"
