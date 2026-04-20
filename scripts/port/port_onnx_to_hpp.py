import argparse, sys
import onnx
import numpy as np

from pathlib import Path

from utils import write_weights

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def get_onnx_weights(model : onnx.ModelProto, layout='nhwc'):
    kernels = []
    biases = []
    alphas = []

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
                        kernels.extend(transposed.flatten().tolist())

            if len(node.input) > 2:
                for initializer in model.graph.initializer:
                    if initializer.name == node.input[2]:
                        bias_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                        biases.extend(bias_data.flatten().tolist())

        if node.op_type == 'Mul':
            for initializer in model.graph.initializer:
                if initializer.name == node.input[1]:
                    alphas_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                    if len(alphas_data) > 1:
                        alphas.extend(alphas_data.flatten().tolist())

        if node.op_type == 'PRelu':
            for initializer in model.graph.initializer:
                if initializer.name == node.input[1]:
                    alphas_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                    alphas.extend(alphas_data.flatten().tolist())

        if node.op_type == 'ConvTranspose':
            for initializer in model.graph.initializer:
                if initializer.name == node.input[1]:
                    weight_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                    weight_data = weight_data.reshape(tuple(initializer.dims))

                    if len(weight_data.shape) == 4:
                        transposed = weight_data.transpose(1, 0, 2, 3)
                        transposed = transposed.transpose(*axes)
                        kernels.extend(transposed.flatten().tolist())

            if len(node.input) > 2:
                for initializer in model.graph.initializer:
                    if initializer.name == node.input[2]:
                        bias_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                        biases.extend(bias_data.flatten().tolist())

    return kernels, biases, alphas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type = str, default = './', help = 'output dir')
    parser.add_argument('-w', '--weights', type=str, help="ONNX weights file path")
    parser.add_argument('-p', '--prefix', type=str, help="output file name and array prefix")
    parser.add_argument('-l', '--layout', type=str, default='nhwc', help="output layout of weights per layer")
    args = parser.parse_args()

    model = onnx.load(args.weights)

    write_weights(args.output_dir, args.prefix, args.layout, *get_onnx_weights(model, layout=args.layout))
