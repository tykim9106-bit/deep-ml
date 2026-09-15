import torch
import torch.nn as nn

def build_model() -> nn.Module:
    """
    Tiny MNIST classifier.

    Trainable params:
      conv1: 1 * 12 * 3 * 3       = 108
      conv2: 12 * 13 * 3 * 3      = 1404
      fc:    13 * 2 * 2 * 10 + 10 = 530
      ----------------------------------
      total                        = 2042
    """
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(1, 12, 3, padding=1, bias=False),
                nn.BatchNorm2d(12, affine=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),          # 28 -> 14

                nn.Conv2d(12, 13, 3, padding=1, bias=False),
                nn.BatchNorm2d(13, affine=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),          # 14 -> 7

                nn.AdaptiveAvgPool2d((2, 2)),
            )

            self.classifier = nn.Linear(13 * 2 * 2, 10)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    return TinyNet()