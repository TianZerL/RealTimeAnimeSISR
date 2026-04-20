import argparse, sys
import torch

from pathlib import Path

from utils import get_torch_weights_numpy, conv_to_glsl, pixelsuffle_4to1_to_glsl, PReLU, Identity, ResidualArg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from archs import acnet_arch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type = str, default = './', help = 'output dir')
    parser.add_argument('-w', '--weights', type = str, help = 'weights file path')
    parser.add_argument('-p', '--prefix', type = str, help = 'output file name and array perfix')
    parser.add_argument('-f', '--feat', type = int, default = 8, help = 'feat number')
    parser.add_argument('-b', '--block', type = int, default = 8, help = 'block number')
    parser.add_argument('-F', '--factor', type = float, default = 1.2, help = 'scale threshold for enabling shaders')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = acnet_arch.ACNet_Best(1, 1, 2, num_feat=args.feat, num_block=args.block)
    model.load_state_dict(torch.load(args.weights, weights_only=True, map_location='cpu')['params_ema'])
    model = model.reparameterize().eval()

    kernels, biases, alphas = get_torch_weights_numpy(model, layout='hwcn')

    model_name = f'ACNet F{args.feat}B{args.block}'

    glsl = ''
    factor = args.factor
    l = 0
    glsl += conv_to_glsl(
        input_name='LUMA',
        output_name='TMP1_TEX',
        num_in_ch=1,
        num_out_ch=args.feat,
        kernels=kernels[l],
        biases=biases[l],
        activation=PReLU(alphas=alphas[l]),
        kernel_size=3,
        desc=f'{model_name} head conv 1x8x3x3',
        factor=factor
    )
    l += 1

    for b in range(0, args.block, 2):
        glsl += conv_to_glsl(
            input_name='TMP1_TEX',
            output_name='TMP2_TEX',
            num_in_ch=args.feat,
            num_out_ch=args.feat,
            kernels=kernels[l],
            biases=biases[l],
            activation=PReLU(alphas=alphas[l]),
            kernel_size=3,
            desc=f'{model_name} body block {b + 1} conv 8x8x3x3',
            factor=factor
        )
        l += 1
        glsl += conv_to_glsl(
            input_name='TMP2_TEX',
            output_name='TMP1_TEX',
            num_in_ch=args.feat,
            num_out_ch=args.feat,
            kernels=kernels[l],
            biases=biases[l],
            activation=PReLU(alphas=alphas[l]),
            kernel_size=3,
            desc=f'{model_name} body block {b + 2} conv 8x8x3x3',
            factor=factor
        )
        l += 1

    glsl += conv_to_glsl(
        input_name='TMP1_TEX',
        output_name='TMP2_TEX',
        num_in_ch=args.feat,
        num_out_ch=4,
        kernels=kernels[l],
        biases=biases[l],
        activation=Identity(),
        kernel_size=3,
        desc=f'{model_name} upscale conv 8x4x3x3',
        residual_args=[ResidualArg('LUMA', 1.0, 1)],
        factor=factor
    )
    l += 1

    glsl += pixelsuffle_4to1_to_glsl(
        input_name='TMP2_TEX',
        desc=f'{model_name}upscale pixelshuff',
    )

    with open(output_dir / f'{args.prefix}.glsl', 'w') as f:
        f.write(glsl)
