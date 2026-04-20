import torch
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class Bicubic(torch.nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bicubic')
