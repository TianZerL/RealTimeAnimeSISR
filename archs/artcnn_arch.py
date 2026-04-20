import torch
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class ArtCNN(torch.nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, num_feat=32):
        super().__init__()

        self.head = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, padding_mode='replicate')

        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, padding_mode='replicate'),
        )

        self.upscale = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat, num_out_ch * (scale ** 2), 3, 1, 1, padding_mode='replicate'),
            torch.nn.PixelShuffle(scale)
        )

    def forward(self, x):
        feat = self.head(x)
        feat_res = self.body(feat)
        feat = feat + feat_res
        out = self.upscale(feat)
        return out
