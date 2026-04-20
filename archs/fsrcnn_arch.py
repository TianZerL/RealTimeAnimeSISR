import math
import torch
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class FSRCNN(torch.nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, d=56, s=12, m=4, paper_init_weights=False):
        super().__init__()

        self.feature_extraction = torch.nn.Conv2d(num_in_ch, d, 5, 1, 2, padding_mode='replicate')

        self.shrinking = torch.nn.Sequential(
            torch.nn.Conv2d(d, s, 1, 1, 0),
            torch.nn.PReLU(s)
        )

        mapping_layers = []
        for _ in range(m):
            mapping_layers.extend((
                torch.nn.Conv2d(s, s, 3, 1, 1, padding_mode='replicate'),
                torch.nn.PReLU(s)
            ))

        self.mapping = torch.nn.Sequential(*mapping_layers)

        self.expanding = torch.nn.Sequential(
            torch.nn.Conv2d(s, d, 1, 1, 0),
            torch.nn.PReLU(d)
        )

        self.deconv = torch.nn.ConvTranspose2d(d, num_out_ch, 9, scale, 4, scale - 1)

        if paper_init_weights:
            self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.feature_extraction(x)
        feat = self.shrinking(feat)
        feat = self.mapping(feat)
        feat = self.expanding(feat)
        out = self.deconv(feat)
        return out
