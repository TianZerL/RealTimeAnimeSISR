import torch
import torch.nn.functional as F

from .utils import CLBlock

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class SESR(torch.nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, f=16, m=3, feature_size=256):
        super().__init__()

        self.head = CLBlock(num_in_ch, f, middle_channels=feature_size, kernel_size=5, padding_mode='replicate')

        linear_blocks = []
        for _ in range(m):
            linear_blocks.extend((
                CLBlock(f, f, middle_channels=feature_size, kernel_size=3, padding_mode='replicate'),
                torch.nn.PReLU(f)
            ))

        self.body = torch.nn.Sequential(*linear_blocks)

        self.upscale = torch.nn.Sequential(
            CLBlock(f, num_out_ch * (scale ** 2), middle_channels=feature_size, kernel_size=5, padding_mode='replicate'),
            torch.nn.PixelShuffle(scale)
        )

        self.scale = scale

    @torch.no_grad()
    def reparameterize(self):
        rep_sesr = SESR.__new__(SESR)
        super(SESR, rep_sesr).__init__()

        rep_sesr.head = self.head.reparameterize()

        body_rep_blocks = []
        for layer in self.body:
            if hasattr(layer, 'reparameterize') and callable(getattr(layer, 'reparameterize', None)):
                body_rep_blocks.append(layer.reparameterize())
            else:
                body_rep_blocks.append(layer)

        rep_sesr.body = torch.nn.Sequential(*body_rep_blocks)

        upscale_rep_blocks = []
        for layer in self.upscale:
            if hasattr(layer, 'reparameterize') and callable(getattr(layer, 'reparameterize', None)):
                upscale_rep_blocks.append(layer.reparameterize())
            else:
                upscale_rep_blocks.append(layer)
        rep_sesr.upscale = torch.nn.Sequential(*upscale_rep_blocks)

        rep_sesr.scale = self.scale

        return rep_sesr.to(next(self.parameters()).device)

    def forward(self, x):
        feat = self.head(x)
        feat_res = self.body(feat)
        feat = feat + feat_res
        out = self.upscale(feat)
        out = out + F.interpolate(x, scale_factor=self.scale, mode='nearest-exact')
        return out
