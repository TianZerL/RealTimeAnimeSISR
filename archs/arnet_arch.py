import torch
import torch.nn.functional as F

from itertools import cycle
from typing import Union, List

from .utils import RepBlockBuilderMaker, ResidualBlock

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ARNet(torch.nn.Module):
    def __init__(self,
                 num_in_ch=1,
                 num_out_ch=1,
                 scale=2,
                 num_feat=8,
                 num_block=4,
                 res_block_res_scale=0.2,
                 res_block_res_scale_learnable=False,
                 rep_conv3x3_block_head_name='plain',
                 rep_conv3x3_block_body_name: Union[str, List[str]] = 'plain',
                 rep_conv3x3_block_upscale_name='plain',
                 activation_name='lrelu',
                 fusion_layer=False,
                 fusion_activation_after_add=False,
                 pre_activation=False,
                 post_activation=False,
                 res_learning=False
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
        residual_blocks = []
        for _ in range(num_block):
            residual_blocks.append(
                ResidualBlock(
                    num_feat=num_feat,
                    res_scale=res_block_res_scale,
                    res_scale_learnable=res_block_res_scale_learnable,
                    conv3x3_1_builder=next(rep_conv3x3_block_body_builders),
                    conv3x3_2_builder=next(rep_conv3x3_block_body_builders),
                    activation_builder=activation_builder,
                    pre_activation=pre_activation,
                    post_activation=post_activation
                )
            )

        self.head = rep_conv3x3_block_head_builder.build()

        self.body = torch.nn.Sequential(*residual_blocks)

        if fusion_layer:
            self.body.append(torch.nn.Conv2d(num_feat, num_feat, 1, 1, 0))
            if fusion_activation_after_add:
                self.fusion_activation = activation_builder.build()
            else:
                self.body.append(activation_builder.build())

        self.upscale = torch.nn.Sequential(
            rep_conv3x3_block_upscale_builder.build(),
            torch.nn.PixelShuffle(scale)
        )

        self.fusion_activation_after_add = fusion_layer and fusion_activation_after_add
        self.res_learning = res_learning
        self.scale = scale

    @torch.no_grad()
    def reparameterize(self):
        rep_arnet = ARNet.__new__(ARNet)
        super(ARNet, rep_arnet).__init__()

        if hasattr(self.head, 'reparameterize') and callable(getattr(self.head, 'reparameterize', None)):
            rep_arnet.head = self.head.reparameterize()
        else:
            rep_arnet.head = self.head

        body_rep_blocks = []
        for layer in self.body:
            if hasattr(layer, 'reparameterize') and callable(getattr(layer, 'reparameterize', None)):
                body_rep_blocks.append(layer.reparameterize())
            else:
                body_rep_blocks.append(layer)

        rep_arnet.body = torch.nn.Sequential(*body_rep_blocks)

        upscale_rep_blocks = []
        for layer in self.upscale:
            if hasattr(layer, 'reparameterize') and callable(getattr(layer, 'reparameterize', None)):
                upscale_rep_blocks.append(layer.reparameterize())
            else:
                upscale_rep_blocks.append(layer)
        rep_arnet.upscale = torch.nn.Sequential(*upscale_rep_blocks)

        if self.fusion_activation_after_add:
            rep_arnet.fusion_activation = self.fusion_activation
        rep_arnet.fusion_activation_after_add = self.fusion_activation_after_add

        rep_arnet.res_learning = self.res_learning
        rep_arnet.scale = self.scale

        return rep_arnet.to(next(self.parameters()).device)

    def forward(self, x):
        feat = self.head(x)
        feat_res = self.body(feat)
        feat = feat + feat_res
        if self.fusion_activation_after_add:
            feat = self.fusion_activation(feat)
        out = self.upscale(feat)
        if self.res_learning:
            out = out + F.interpolate(x, scale_factor=self.scale, mode='nearest-exact')
        return out

@ARCH_REGISTRY.register()
class ARNet_BaseLine(ARNet):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, num_feat=8, num_block=4):
        super().__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            res_block_res_scale=0.2,
            res_block_res_scale_learnable=False,
            rep_conv3x3_block_head_name='plain',
            rep_conv3x3_block_body_name='plain',
            rep_conv3x3_block_upscale_name='plain',
            activation_name='lrelu',
            fusion_layer=False,
            fusion_activation_after_add=False,
            pre_activation=False,
            post_activation=False,
            res_learning=False
        )

@ARCH_REGISTRY.register()
class ARNet_Best(ARNet):
    def __init__(self, num_in_ch=1, num_out_ch=1, scale=2, num_feat=8, num_block=4, rep_conv3x3_block_head_name=None, rep_conv3x3_block_body_name=None, rep_conv3x3_block_upscale_name=None):
        super().__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            res_block_res_scale=0.2,
            res_block_res_scale_learnable=False,
            rep_conv3x3_block_head_name=rep_conv3x3_block_head_name if rep_conv3x3_block_head_name is not None else ('rrrb' if num_feat <= 16 and num_block <= 64 else 'plain'),
            rep_conv3x3_block_body_name=rep_conv3x3_block_body_name if rep_conv3x3_block_body_name is not None else (['rrrb', 'plain'] if num_feat <= 16 and num_block <= 64 else 'plain'),
            rep_conv3x3_block_upscale_name=rep_conv3x3_block_upscale_name if rep_conv3x3_block_upscale_name is not None else ('rrrb' if num_feat <= 16 and num_block <= 64 else 'plain'),
            activation_name='prelu',
            fusion_layer=True,
            fusion_activation_after_add=False,
            pre_activation=False,
            post_activation=False,
            res_learning=True
        )
