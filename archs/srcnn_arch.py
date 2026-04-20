import torch
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class SRCNN(torch.nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, n1 = 64, n2 = 32, f1 = 9, f2 = 5, f3 = 5, paper_init_weights=False):
        super().__init__()

        self.upscale = torch.nn.Upsample(scale_factor=scale, mode='bicubic')

        self.feature_maps  = torch.nn.Sequential(
            torch.nn.Conv2d(num_in_ch, n1, f1, 1, f1 // 2, padding_mode='replicate'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n1, n2, f2, 1, f2 // 2, padding_mode='replicate'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n2, num_out_ch, f3, 1, f3 // 2, padding_mode='replicate'),
        )

        if paper_init_weights:
            self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.upscale(x)
        out = self.feature_maps(feat)
        return out
