import argparse, sys
import onnx
import numpy as np

from pathlib import Path

from utils import conv_to_glsl, deconv4x4_8to1_to_glsl, ReLU

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def get_onnx_weights_numpy(model : onnx.ModelProto, layout='nhwc'):
    kernels = []
    biases = []

    if layout == 'nchw':
        axes = [0, 1, 2, 3]
    elif layout == 'nhwc':
        axes = [0, 2 ,3, 1]
    elif layout == 'hwnc':
        axes = [2 ,3, 0, 1]
    elif layout == 'hwcn':
        axes = [2 ,3, 1, 0]
    else:
        raise ValueError('Unsupported layout')

    for node in model.graph.node:
        if node.op_type == 'Conv':
            for initializer in model.graph.initializer:
                if initializer.name == node.input[1]:
                    weight_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                    weight_data = weight_data.reshape(tuple(initializer.dims))

                    if len(weight_data.shape) == 4:  # Conv2d
                        transposed = weight_data.transpose(*axes)
                        kernels.append(transposed)

            if len(node.input) > 2:
                for initializer in model.graph.initializer:
                    if initializer.name == node.input[2]:
                        bias_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                        biases.append(bias_data)

        if node.op_type == 'ConvTranspose':
            for initializer in model.graph.initializer:
                if initializer.name == node.input[1]:
                    weight_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                    weight_data = weight_data.reshape(tuple(initializer.dims))

                    if len(weight_data.shape) == 4:
                        transposed = weight_data.transpose(1, 0, 2, 3)
                        transposed = transposed.transpose(*axes)
                        kernels.append(transposed)

            if len(node.input) > 2:
                for initializer in model.graph.initializer:
                    if initializer.name == node.input[2]:
                        bias_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                        biases.append(bias_data)

    return kernels, biases

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type = str, default = './', help = 'output dir')
    parser.add_argument('-w', '--weights', type = str, help = 'weights file path')
    parser.add_argument('-p', '--prefix', type = str, help = 'output file name and array perfix')
    parser.add_argument('-F', '--factor', type = float, default = 1.2, help = 'scale threshold for enabling shaders')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(args.weights)

    kernels, biases = get_onnx_weights_numpy(model, layout='hwcn')

    model_name = f'ACNet Classic'

    glsl = ''
    factor = args.factor
    l = 0
    glsl += conv_to_glsl(
        input_name='LUMA',
        output_name='TMP1_TEX',
        num_in_ch=1,
        num_out_ch=8,
        kernels=kernels[l],
        biases=biases[l],
        activation=ReLU(),
        kernel_size=3,
        desc=f'{model_name} conv 0 1x8x3x3',
        factor=factor
    )
    l += 1

    for b in range(0, 8, 2):
        glsl += conv_to_glsl(
            input_name='TMP1_TEX',
            output_name='TMP2_TEX',
            num_in_ch=8,
            num_out_ch=8,
            kernels=kernels[l],
            biases=biases[l],
            activation=ReLU(),
            kernel_size=3,
            desc=f'{model_name} conv {b + 1} 8x8x3x3',
            factor=factor
        )
        l += 1
        glsl += conv_to_glsl(
            input_name='TMP2_TEX',
            output_name='TMP1_TEX',
            num_in_ch=8,
            num_out_ch=8,
            kernels=kernels[l],
            biases=biases[l],
            activation=ReLU(),
            kernel_size=3,
            desc=f'{model_name} conv {b + 2} 8x8x3x3',
            factor=factor
        )
        l += 1

    glsl += deconv4x4_8to1_to_glsl(
        input_name='TMP1_TEX',
        kernels=kernels[l],
        desc=f'{model_name} deconv 8x1x2x2',
        factor=factor
    )

    with open(output_dir / f'{args.prefix}.glsl', 'w') as f:
        f.write(glsl)
