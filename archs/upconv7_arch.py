import torch
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Upconv7(torch.nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2):
        super().__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(num_in_ch, 16, 3, 1, 1, padding_mode='replicate'),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(16, 32, 3, 1, 1, padding_mode='replicate'),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(32, 64, 3, 1, 1, padding_mode='replicate'),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(64, 128, 3, 1, 1, padding_mode='replicate'),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(128, 128, 3, 1, 1, padding_mode='replicate'),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(128, 256, 3, 1, 1, padding_mode='replicate'),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, num_out_ch * (scale ** 2), 3, 1, 1, padding_mode='replicate'),
            torch.nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.sequential(x)
