import torch
import torch.nn.functional as F

class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def drop_path(self, x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

# Reparameterizable blocks

class RRRBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels = None, padding_mode = 'replicate'):
        super().__init__()

        if middle_channels is None:
            if in_channels <= 8:
                middle_channels = out_channels * 4
            else:
                middle_channels = out_channels * 2

        self.conv1 = torch.nn.Conv2d(in_channels, middle_channels, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(middle_channels, middle_channels, 3, 1, 0)
        self.conv3 = torch.nn.Conv2d(middle_channels, out_channels, 1, 1, 0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.with_idt = in_channels == out_channels

    @torch.no_grad()
    def reparameterize(self):
        device = self.conv1.weight.device

        k0 = self.conv1.weight
        k1 = self.conv2.weight
        k2 = self.conv3.weight

        b0 = self.conv1.bias
        b1 = self.conv2.bias
        b2 = self.conv3.bias

        ki = torch.eye(k1.size(0), k1.size(1)).view(k1.size(0), k1.size(1), 1, 1).to(device)
        ki = F.pad(ki, [1, 1, 1, 1])
        k1 = k1 + ki

        merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
        merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, b0.size(0), 3, 3).to(device)
        merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

        merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
        merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

        if self.with_idt:
            ki = torch.eye(merged_k0k1k2.size(0), merged_k0k1k2.size(1)).view(merged_k0k1k2.size(0), merged_k0k1k2.size(1), 1, 1).to(device)
            ki = F.pad(ki, [1, 1, 1, 1])
            merged_k0k1k2 = merged_k0k1k2 + ki

        rep_conv3x3 = torch.nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, padding_mode=self.padding_mode, device=device)
        rep_conv3x3.weight.data = merged_k0k1k2
        rep_conv3x3.bias.data = merged_b0b1b2

        return rep_conv3x3

    def forward(self, x):
        out = F.pad(x, [1, 1, 1, 1], mode=('constant' if self.padding_mode == 'zeros' else self.padding_mode))
        out = self.conv1(out)
        out = self.conv2(out) + out[:, :, 1:-1, 1:-1]
        out = self.conv3(out)
        if self.with_idt:
            out += x
        return out

class RCBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode = 'replicate'):
        super().__init__()
        self.conv3x3 = torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode=padding_mode)
        self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.with_idt = in_channels == out_channels

    @torch.no_grad()
    def reparameterize(self):
        device = self.conv3x3.weight.device

        k0 = self.conv3x3.weight
        k1 = F.pad(self.conv1x1.weight, [1, 1, 1, 1])

        b0 = self.conv3x3.bias
        b1 = self.conv1x1.bias

        merged_k0k1 = k0 + k1
        merged_b0b1 = b0 + b1

        if self.with_idt:
            ki = torch.eye(merged_k0k1.size(0), merged_k0k1.size(1)).view(merged_k0k1.size(0), merged_k0k1.size(1), 1, 1).to(device)
            ki = F.pad(ki, [1, 1, 1, 1])
            merged_k0k1 = merged_k0k1 + ki

        rep_conv3x3 = torch.nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, padding_mode=self.padding_mode, device = device)
        rep_conv3x3.weight.data = merged_k0k1
        rep_conv3x3.bias.data = merged_b0b1
        return rep_conv3x3

    def forward(self, x):
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.with_idt:
            out += x
        return out

class ACBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode = 'replicate'):
        super().__init__()
        self.conv3x3 = torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode=padding_mode)
        self.conv3x1 = torch.nn.Conv2d(in_channels, out_channels, (3, 1), 1, (1, 0), padding_mode=padding_mode)
        self.conv1x3 = torch.nn.Conv2d(in_channels, out_channels, (1, 3), 1, (0, 1), padding_mode=padding_mode)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode

    @torch.no_grad()
    def reparameterize(self):
        device = self.conv3x3.weight.device

        kernel_3x3 = self.conv3x3.weight
        kernel_3x1 = F.pad(self.conv3x1.weight, [1, 1, 0, 0])
        kernel_1x3 = F.pad(self.conv1x3.weight, [0, 0, 1, 1])

        bias_3x3 = self.conv3x3.bias
        bias_3x1 = self.conv3x1.bias
        bias_1x3 = self.conv1x3.bias

        rep_conv3x3 = torch.nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, padding_mode=self.padding_mode, device=device)
        rep_conv3x3.weight.data = kernel_3x3 + kernel_3x1 + kernel_1x3
        rep_conv3x3.bias.data = bias_3x3 + bias_3x1 + bias_1x3

        return rep_conv3x3

    def forward(self, x):
        return self.conv3x3(x) + self.conv3x1(x) + self.conv1x3(x)

class SeqConv3x3(torch.nn.Module):
    class Conv1x1Conv3x3(torch.nn.Module):
        def __init__(self, in_channels, out_channels, middle_channels = None, padding_mode = 'replicate'):
            super().__init__()
            if middle_channels is None:
                middle_channels = in_channels

            self.conv1x1 = torch.nn.Conv2d(in_channels, middle_channels, 1, 1, 0)
            self.conv3x3 = torch.nn.Conv2d(middle_channels, out_channels, 3, 1, 0)

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.padding_mode = padding_mode

        @torch.no_grad()
        def reparameterize(self):
            device = self.conv1x1.weight.device

            k0 = self.conv1x1.weight
            k1 = self.conv3x3.weight

            b0 = self.conv1x1.bias
            b1 = self.conv3x3.bias

            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, b0.size(0), 3, 3).to(device)
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1).view(-1)

            rep_conv3x3 = torch.nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, padding_mode=self.padding_mode, device=device)
            rep_conv3x3.weight.data = merged_k0k1
            rep_conv3x3.bias.data = merged_b0b1

            return rep_conv3x3

        def forward(self, x):
            out = F.pad(x, [1, 1, 1, 1], mode=('constant' if self.padding_mode == 'zeros' else self.padding_mode))
            out = self.conv1x1(out)
            out = self.conv3x3(out)
            return out

    class Conv1x1Filter3x3(torch.nn.Module):
        def __init__(self, in_channels, out_channels, filter3x3_weight, padding_mode = 'replicate'):
            super().__init__()

            self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0)

            scale = torch.randn(size=(out_channels, 1, 1, 1)) * 1e-3
            self.scale = torch.nn.Parameter(scale)
            bias = torch.randn(out_channels) * 1e-3
            bias = torch.reshape(bias, (out_channels, ))
            self.bias = torch.nn.Parameter(bias)

            self.register_buffer('filter_kernel', filter3x3_weight.view(1, 1, 3, 3).repeat(out_channels, 1, 1, 1))

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.padding_mode = padding_mode

        @torch.no_grad()
        def reparameterize(self):
            device = self.conv1x1.weight.device

            k0 = self.conv1x1.weight
            b0 = self.conv1x1.bias

            tmp = self.scale * self.filter_kernel
            k1 = torch.zeros((self.out_channels, self.out_channels, 3, 3), device=device)
            for i in range(self.out_channels):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias

            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, b0.size(0), 3, 3).to(device)
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1).view(-1)

            rep_conv3x3 = torch.nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, padding_mode=self.padding_mode, device=device)
            rep_conv3x3.weight.data = merged_k0k1
            rep_conv3x3.bias.data = merged_b0b1

            return rep_conv3x3

        def forward(self, x):
            out = F.pad(x, [1, 1, 1, 1], mode=('constant' if self.padding_mode == 'zeros' else self.padding_mode))
            out = self.conv1x1(out)
            out = F.conv2d(input=out, weight=self.scale * self.filter_kernel, bias=self.bias, stride=1, groups=self.out_channels)
            return out

    def __init__(self, seq_type, in_channels, out_channels, middle_channels = None, padding_mode = 'replicate'):
        super().__init__()

        if seq_type == 'conv1x1-conv3x3':
            self.seqconv3x3 = self.Conv1x1Conv3x3(in_channels=in_channels, out_channels=out_channels, middle_channels=middle_channels, padding_mode=padding_mode)
        elif seq_type == 'conv1x1-sobelx':
            self.seqconv3x3 = self.Conv1x1Filter3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                filter3x3_weight=torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]),
                padding_mode=padding_mode
            )
        elif seq_type == 'conv1x1-sobely':
            self.seqconv3x3 = self.Conv1x1Filter3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                filter3x3_weight=torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]),
                padding_mode=padding_mode
            )
        elif seq_type == 'conv1x1-laplacian':
            self.seqconv3x3 = self.Conv1x1Filter3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                filter3x3_weight=torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]),
                padding_mode=padding_mode
            )
        else:
            raise ValueError('The type of seqconv is not supported!')

    def forward(self, x):
        return self.seqconv3x3(x)

    @torch.no_grad()
    def reparameterize(self):
        return self.seqconv3x3.reparameterize()

class ECBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels = None, with_idt = False, padding_mode = 'replicate'):
        super().__init__()

        if middle_channels is None:
            middle_channels = out_channels * 2

        self.conv3x3 = torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode=padding_mode)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', in_channels, out_channels, middle_channels, padding_mode=padding_mode)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', in_channels, out_channels, middle_channels, padding_mode=padding_mode)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', in_channels, out_channels, middle_channels, padding_mode=padding_mode)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', in_channels, out_channels, middle_channels, padding_mode=padding_mode)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_idt and (self.in_channels == self.out_channels):
            self.with_idt = True
        else:
            self.with_idt = False
        self.padding_mode = padding_mode

    def forward(self, x):
        out = self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x)
        if self.with_idt:
            out = out + x
        return out

    @torch.no_grad()
    def reparameterize(self):
        device = self.conv3x3.weight.device

        conv0 = self.conv3x3
        conv1 = self.conv1x1_3x3.reparameterize()
        conv2 = self.conv1x1_sbx.reparameterize()
        conv3 = self.conv1x1_sby.reparameterize()
        conv4 = self.conv1x1_lpl.reparameterize()

        k0 = conv0.weight + conv1.weight + conv2.weight + conv3.weight + conv4.weight
        b0 = conv0.bias + conv1.bias + conv2.bias + conv3.bias + conv4.bias

        if self.with_idt:
            ki = torch.eye(k0.size(0), k0.size(1)).view(k0.size(0), k0.size(1), 1, 1).to(device)
            ki = F.pad(ki, [1, 1, 1, 1])
            k0 = k0 + ki

        rep_conv3x3 = torch.nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, padding_mode=self.padding_mode, device=device)
        rep_conv3x3.weight.data = k0
        rep_conv3x3.bias.data = b0

        return rep_conv3x3

class CLBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels = None, kernel_size = 3, padding_mode = 'replicate'):
        super().__init__()

        if middle_channels is None:
            middle_channels = out_channels * 2

        padding_size = kernel_size // 2

        self.convkxk = torch.nn.Conv2d(in_channels, middle_channels, kernel_size, 1, padding_size, padding_mode=padding_mode)
        self.conv1x1 = torch.nn.Conv2d(middle_channels, out_channels, 1, 1, 0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.with_idt = in_channels == out_channels

    def forward(self, x):
        out = self.convkxk(x)
        out = self.conv1x1(out)
        if self.with_idt:
            out += x
        return out

    @torch.no_grad()
    def reparameterize(self):
        device = self.convkxk.weight.device

        k0 = self.convkxk.weight
        k1 = self.conv1x1.weight

        b0 = self.convkxk.bias
        b1 = self.conv1x1.bias


        merged_k0k1 = F.conv2d(input=k0.permute(1, 0, 2, 3), weight=k1).permute(1, 0, 2, 3)
        merged_b0b1 = F.conv2d(input=b0.view(1, -1, 1, 1), weight=k1, bias=b1).view(-1)

        if self.with_idt:
            ki = torch.eye(merged_k0k1.size(0), merged_k0k1.size(1)).view(merged_k0k1.size(0), merged_k0k1.size(1), 1, 1).to(device)
            ki = F.pad(ki, [self.padding_size, self.padding_size, self.padding_size, self.padding_size])
            merged_k0k1 = merged_k0k1 + ki

        rep_convkxk = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 1, self.padding_size, padding_mode=self.padding_mode, device=device)
        rep_convkxk.weight.data = merged_k0k1
        rep_convkxk.bias.data = merged_b0b1

        return rep_convkxk

class PlainConv3x3Block(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, padding_mode='replicate'):
        super().__init__(in_channels, out_channels, 3, 1, 1, padding_mode=padding_mode)

    @torch.no_grad()
    def reparameterize(self):
        return self

class RepBlockBuilder():
    def __init__(self, rep_type, *args, **kwds):
        self.rep_type = rep_type
        self.args = args
        self.kwds = kwds

    def build(self):
        obj = self.rep_type(*self.args, **self.kwds)

        if not hasattr(obj, 'reparameterize'):
            obj.reparameterize = torch.no_grad()(lambda: obj)

        return obj

class RepBlockBuilderMaker():
    def make_activation_builder(activation_name, num_parameters, negative_slope, inplace=True):
        if activation_name == 'prelu':
            activation_builder = RepBlockBuilder(rep_type=torch.nn.PReLU, num_parameters=num_parameters)
            init_a = 0.25
        elif activation_name == 'lrelu':
            activation_builder = RepBlockBuilder(rep_type=torch.nn.LeakyReLU, negative_slope=negative_slope, inplace=inplace)
            init_a = 0.2
        elif activation_name == 'relu':
            activation_builder = RepBlockBuilder(rep_type=torch.nn.ReLU, inplace=inplace)
            init_a = 0.0
        else:
            raise RuntimeError('Unsupported activation')

        return activation_builder, init_a

    def make_rep_conv3x3_block_builder(rep_conv3x3_block_name, in_channels, out_channels, padding_mode='replicate'):
        if rep_conv3x3_block_name == 'rrrb':
            rep_conv3x3_block_builder = RepBlockBuilder(rep_type=RRRBlock, in_channels=in_channels, out_channels=out_channels, padding_mode=padding_mode)
        elif rep_conv3x3_block_name == 'rcb':
            rep_conv3x3_block_builder = RepBlockBuilder(rep_type=RCBlock, in_channels=in_channels, out_channels=out_channels, padding_mode=padding_mode)
        elif rep_conv3x3_block_name == 'acb':
            rep_conv3x3_block_builder = RepBlockBuilder(rep_type=ACBlock, in_channels=in_channels, out_channels=out_channels, padding_mode=padding_mode)
        elif rep_conv3x3_block_name == 'ecb':
            rep_conv3x3_block_builder = RepBlockBuilder(rep_type=ECBlock, in_channels=in_channels, out_channels=out_channels, padding_mode=padding_mode)
        elif rep_conv3x3_block_name == 'plain':
            rep_conv3x3_block_builder = RepBlockBuilder(rep_type=PlainConv3x3Block, in_channels=in_channels, out_channels=out_channels, padding_mode=padding_mode)
        else:
            raise RuntimeError('Unsupported rep conv3x3 block')

        return rep_conv3x3_block_builder

# Attention  blocks

class CABlock(torch.nn.Module):
    def __init__(self, num_feat, reduction=1, bias=False):
        super().__init__()
        self.ca = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(num_feat, num_feat // reduction, 1, 1, 0, bias=bias),
            torch.nn.PReLU(num_feat // reduction),
            torch.nn.Conv2d(num_feat // reduction, num_feat, 1, 1, 0, bias=bias),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.ca(x)

class SABlock(torch.nn.Module):
    def __init__(self, kernel_size = 1, bias=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size, 1, kernel_size // 2, bias=bias)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_feat = torch.mean(x, 1, True)
        max_feat, _ = torch.max(x, 1, True)
        feat = torch.cat((avg_feat, max_feat), 1)
        return x * self.sigmoid(self.conv(feat))

# ResBlock

class ResidualBlock(torch.nn.Module):
    def __init__(self,
                num_feat=8,
                res_scale=1.0,
                res_scale_learnable=False,
                conv3x3_1_builder=None,
                conv3x3_2_builder=None,
                activation_builder=None,
                pre_activation=False,
                post_activation=False
            ):
        super().__init__()

        if conv3x3_1_builder is None:
            conv3x3_1_builder = RepBlockBuilder(rep_type=PlainConv3x3Block, in_channels=num_feat, out_channels=num_feat, padding_mode='replicate')

        if conv3x3_2_builder is None:
            conv3x3_2_builder = conv3x3_1_builder

        if activation_builder is None:
            activation_builder = RepBlockBuilder(rep_type=torch.nn.LeakyReLU, negative_slope=0.2, inplace=True)

        if res_scale_learnable:
            self.res_scale = torch.nn.Parameter(torch.tensor(float(res_scale)))
        else:
            self.res_scale = float(res_scale)

        self.residual = torch.nn.Sequential(
            conv3x3_1_builder.build(),
            activation_builder.build(),
            conv3x3_2_builder.build(),
        )

        if pre_activation:
            self.residual.insert(0, activation_builder.build())

        if post_activation:
            self.residual.append(activation_builder.build())

    @torch.no_grad()
    def reparameterize(self):
        rep_blocks = []
        for layer in self.residual:
            rep_blocks.append(layer.reparameterize())

        rep_residual_block = ResidualBlock.__new__(ResidualBlock)
        super(ResidualBlock, rep_residual_block).__init__()

        rep_residual_block.res_scale = self.res_scale
        rep_residual_block.residual = torch.nn.Sequential(*rep_blocks)

        return rep_residual_block.to(next(self.parameters()).device)

    def forward(self, x):
        return x + self.residual(x) * self.res_scale
