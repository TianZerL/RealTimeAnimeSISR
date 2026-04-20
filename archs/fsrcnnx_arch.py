import math
import torch
from basicsr.utils.registry import ARCH_REGISTRY

class MappingBlock(torch.nn.Module):
    def __init__(self, m, s):
        super().__init__()

        mapping_layers = []
        for _ in range(m):
            mapping_layers.extend((torch.nn.Conv2d(s, s, 3, 1, 1, padding_mode='replicate'), torch.nn.PReLU(s)))
        mapping_layers.append(torch.nn.Conv2d(s, s, 1, 1, 0))

        self.mapping = torch.nn.Sequential(*mapping_layers)

    def forward(self, x, feature):
        return self.mapping(x) + feature

@ARCH_REGISTRY.register()
class FSRCNNX(torch.nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, d = 8, s = 0, m = 4, r = 1):
        super().__init__()

        self.r = r

        shrink_flag = s > 0

        self.feature_extraction = torch.nn.Conv2d(num_in_ch, d, 5, 1, 2, padding_mode='replicate')

        if shrink_flag:
            self.shrink = torch.nn.Sequential(
                torch.nn.PReLU(d),
                torch.nn.Conv2d(d, s, 1, 1, 0)
            )
        else:
            self.shrink = torch.nn.Identity()
            s = d

        self.mapping = MappingBlock(m, s)
        self.mapping_activation = torch.nn.PReLU(s)

        if shrink_flag:
            self.expand = torch.nn.Sequential(
                torch.nn.Conv2d(s, d, 1, 1, 0),
                torch.nn.PReLU(d)
            )
        else:
            self.expand = torch.nn.Identity()

        self.upscale = torch.nn.Sequential(
            torch.nn.Conv2d(d, num_out_ch * (scale ** 2), 3, 1, 1, padding_mode='replicate'),
            torch.nn.PixelShuffle(scale)
        )

    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrink(out)
        feature = out
        for _ in range(self.r):
            out = self.mapping(out, feature)
        out = self.mapping_activation(out)
        out = self.expand(out)
        out = self.upscale(out)
        return out
