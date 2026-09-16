import torch
import torch.nn as nn


def apply_conv2d():
    conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)

    x = torch.tensor([[[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]]]])

    with torch.no_grad():
        conv.weight.copy_(
            torch.tensor([[[[1.0, 0.0],
                            [0.0, 1.0]]]])
        )
        output = conv(x)

    return output