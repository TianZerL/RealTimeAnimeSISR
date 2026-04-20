import torch
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class ESPCN(torch.nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, n1 = 64, n2 = 32, f1 = 5, f2 = 3, f3 = 3):
        super().__init__()

        self.feature_maps  = torch.nn.Sequential(
            torch.nn.Conv2d(num_in_ch, n1, f1, 1, f1 // 2, padding_mode='replicate'),
            torch.nn.Tanh(),
            torch.nn.Conv2d(n1, n2, f2, 1, f2 // 2, padding_mode='replicate'),
            torch.nn.Tanh(),
        )

        self.upscale = torch.nn.Sequential(
            torch.nn.Conv2d(n2, num_out_ch * (scale ** 2), f3, 1, f3 // 2, padding_mode='replicate'),
            torch.nn.PixelShuffle(scale)
        )

    def forward(self, x):
        feat = self.feature_maps(x)
        out = self.upscale(feat)
        return out
