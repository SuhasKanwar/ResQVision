import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice loss to handle class imbalance, particularly useful for segmentation
    tasks like building footprint damage assessment.
    """
    def __init__(self, smooth=1e-6, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: predictions from model, shape (N, C, H, W)
            targets: ground truth, shape (N, H, W)
        """
        num_classes = logits.shape[1]
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        if self.ignore_index is not None:
            # Create mask for valid classes
            valid_classes = torch.ones(num_classes, dtype=torch.bool, device=logits.device)
            valid_classes[self.ignore_index] = False
            dice_score = dice_score[valid_classes]
            
        return 1. - torch.mean(dice_score)

class FocalLoss(nn.Module):
    """
    Weighted Focal Loss for heavily imbalanced multiclass classification.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: shape (N, C, H, W)
            targets: shape (N, H, W)
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none', ignore_index=self.ignore_index, weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class XBDComboLoss(nn.Module):
    """
    Combine Dice Loss and Focal Loss to leverage stability of Dice over backgrounds
    and Focal's focus on hard misclassified damage severity examples.
    """
    def __init__(self, alpha_focal=None, gamma=2.0, dice_weight=0.5, focal_weight=0.5, ignore_index=4):
        super(XBDComboLoss, self).__init__()
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal = FocalLoss(alpha=alpha_focal, gamma=gamma, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        d_loss = self.dice(logits, targets)
        f_loss = self.focal(logits, targets)
        return self.dice_weight * d_loss + self.focal_weight * f_loss
