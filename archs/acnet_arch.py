import torch
import torch.nn.functional as F

from itertools import cycle
from typing import Union, List

from .utils import RepBlockBuilderMaker

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class ACNet(torch.nn.Module):
    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 scale=2,
                 num_feat=8,
                 num_block=8,
                 rep_conv3x3_block_head_name='plain',
                 rep_conv3x3_block_body_name: Union[str, List[str]] = 'plain',
                 rep_conv3x3_block_upscale_name='plain',
                 activation_name='lrelu',
                 res_learning=False,
                 use_deconv=False
            ):
        super().__init__()

        rep_conv3x3_block_head_builder = RepBlockBuilderMaker.make_rep_conv3x3_block_builder(rep_conv3x3_block_head_name, in_channels=num_in_ch, out_channels=num_feat, padding_mode='replicate')
        rep_conv3x3_block_upscale_builder = RepBlockBuilderMaker.make_rep_conv3x3_block_builder(rep_conv3x3_block_upscale_name, in_channels=num_feat, out_channels=num_out_ch * (scale ** 2), padding_mode='replicate')

        if isinstance(rep_conv3x3_block_body_name, str):
            rep_conv3x3_block_body_builder_list = [RepBlockBuilderMaker.make_rep_conv3x3_block_builder(rep_conv3x3_block_body_name, in_channels=num_feat, out_channels=num_feat, padding_mode='replicate')]
        else:
            rep_conv3x3_block_body_builder_list = []
            for name in rep_conv3x3_block_body_name:
                rep_conv3x3_block_body_builder_list.append(RepBlockBuilderMaker.make_rep_conv3x3_block_builder(name, in_channels=num_feat, out_channels=num_feat, padding_mode='replicate'))

        activation_builder, self.init_a = RepBlockBuilderMaker.make_activation_builder(activation_name, num_parameters=num_feat, negative_slope=0.2, inplace=True)

        rep_conv3x3_block_body_builders = cycle(rep_conv3x3_block_body_builder_list)
        vgg_blocks = []
        for _ in range(num_block):
            vgg_blocks.extend((
                next(rep_conv3x3_block_body_builders).build(),
                activation_builder.build()
            ))

        self.head = torch.nn.Sequential(
            rep_conv3x3_block_head_builder.build(),
            activation_builder.build()
        )

        self.body = torch.nn.Sequential(*vgg_blocks)

        if use_deconv:
            self.upscale = torch.nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2, padding=0, bias=False)
        else:
            self.upscale = torch.nn.Sequential(
                rep_conv3x3_block_upscale_builder.build(),
                torch.nn.PixelShuffle(scale)
            )

        self.res_learning = res_learning
        self.scale = scale

    @torch.no_grad()
    def reparameterize(self): 
        rep_acnet = ACNet.__new__(ACNet)
        super(ACNet, rep_acnet).__init__()

        head_rep_blocks = []
        for layer in self.head:
            if hasattr(layer, 'reparameterize') and callable(getattr(layer, 'reparameterize', None)):
                head_rep_blocks.append(layer.reparameterize())
            else:
                head_rep_blocks.append(layer)

        rep_acnet.head = torch.nn.Sequential(*head_rep_blocks)

        body_rep_blocks = []
        for layer in self.body:
            if hasattr(layer, 'reparameterize') and callable(getattr(layer, 'reparameterize', None)):
                body_rep_blocks.append(layer.reparameterize())
            else:
                body_rep_blocks.append(layer)

        rep_acnet.body = torch.nn.Sequential(*body_rep_blocks)

        if isinstance(self.upscale, torch.nn.Sequential):
            upscale_rep_blocks = []
            for layer in self.upscale:
                if hasattr(layer, 'reparameterize') and callable(getattr(layer, 'reparameterize', None)):
                    upscale_rep_blocks.append(layer.reparameterize())
                else:
                    upscale_rep_blocks.append(layer)

            rep_acnet.upscale = torch.nn.Sequential(*upscale_rep_blocks)
        else:
            rep_acnet.upscale = self.upscale

        rep_acnet.res_learning = self.res_learning
        rep_acnet.scale = self.scale

        return rep_acnet.to(next(self.parameters()).device)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        out = self.upscale(feat)
        if self.res_learning:
            out = out + F.interpolate(x, scale_factor=self.scale, mode='nearest-exact')
        return out

@ARCH_REGISTRY.register()
class ACNet_Classic(ACNet):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, num_feat=8, num_block=8):
        super().__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            rep_conv3x3_block_head_name='plain',
            rep_conv3x3_block_body_name='plain',
            rep_conv3x3_block_upscale_name='plain',
            activation_name='relu',
            res_learning=False,
            use_deconv=True
        )

@ARCH_REGISTRY.register()
class ACNet_Best(ACNet):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, num_feat=8, num_block=4, rep_conv3x3_block_head_name=None, rep_conv3x3_block_body_name=None, rep_conv3x3_block_upscale_name=None):
        super().__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            rep_conv3x3_block_head_name = rep_conv3x3_block_head_name if rep_conv3x3_block_head_name is not None else 'rrrb',
            rep_conv3x3_block_body_name = rep_conv3x3_block_body_name if rep_conv3x3_block_body_name is not None else 'rrrb',
            rep_conv3x3_block_upscale_name = rep_conv3x3_block_upscale_name if rep_conv3x3_block_upscale_name is not None else 'rrrb',
            activation_name='prelu',
            res_learning=True,
            use_deconv=False
        )

@ARCH_REGISTRY.register()
class ACNet_ECBSR(ACNet):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, num_feat=8, num_block=4):
        super().__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            rep_conv3x3_block_head_name='ecb',
            rep_conv3x3_block_body_name='ecb',
            rep_conv3x3_block_upscale_name='ecb',
            activation_name='prelu',
            res_learning=True,
            use_deconv=False
        )

@ARCH_REGISTRY.register()
class ACNet_ABPN(ACNet):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, num_feat=28, num_block=5):
        super().__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            rep_conv3x3_block_head_name='plain',
            rep_conv3x3_block_body_name='plain',
            rep_conv3x3_block_upscale_name='plain',
            activation_name='relu',
            res_learning=True,
            use_deconv=False
        )
