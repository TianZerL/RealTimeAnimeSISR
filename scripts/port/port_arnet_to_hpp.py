import argparse, sys
import torch

from pathlib import Path

from utils import write_weights, get_torch_weights

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from archs import arnet_arch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type = str, default = './', help = 'output dir')
    parser.add_argument('-w', '--weights', type = str, help = "weights file path")
    parser.add_argument('-p', '--prefix', type = str, help = "output file name and array perfix")
    parser.add_argument('-f', '--feat', type = int, default = 8, help = "feat number")
    parser.add_argument('-b', '--block', type = int, default = 8, help = "block number")
    parser.add_argument('-l', '--layout', type=str, default='nhwc', help="output layout of weights per layer")
    args = parser.parse_args()

    model = arnet_arch.ARNet_Best(1, 1, 2, num_feat=args.feat, num_block=args.block)
    model.load_state_dict(torch.load(args.weights, weights_only=True, map_location='cpu')['params_ema'])
    model = model.reparameterize().eval()

    write_weights(args.output_dir, args.prefix, args.layout, *get_torch_weights(model, layout=args.layout))
