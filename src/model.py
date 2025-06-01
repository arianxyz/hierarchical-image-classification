import torch
import torch.nn as nn
from torchvision import models
import torchvision.models as models
import torch.nn as nn
    
class EfficientNetClassifier(nn.Module):
    """
    General EfficientNet-B0 classifier for any number of classes.
    Use this model for clothes, shoes, bags, etc. Just set num_classes as needed.
    """
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super(EfficientNetClassifier, self).__init__()
        self.base = models.efficientnet_b0(pretrained=True)

        for param in self.base.parameters():
            param.requires_grad = True

        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)