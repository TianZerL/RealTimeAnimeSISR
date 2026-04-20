import argparse
import torch

def interpolate_weights(weights_a, weights_b, alpha):
    weights_interp = {}
    for key in weights_a.keys():
        weights_interp[key] = alpha * weights_a[key] + (1 - alpha) * weights_b[key]
    return weights_interp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_name', type = str, default='weights_interp', help = 'output filename')
    parser.add_argument('-a', '--weights_a', type = str, help = 'input 1')
    parser.add_argument('-b', '--weights_b', type = str, help = 'input 2')
    parser.add_argument('-f', '--factor', type = float, help = 'interpolate factor')
    parser.add_argument('--key_a', type = str, default='params_ema', help = 'input 1 key')
    parser.add_argument('--key_b', type = str, default='params_ema', help = 'input 2 key')
    parser.add_argument('--key_o', type = str, default='params_ema', help = 'output key')
    args = parser.parse_args()


    weights_a = torch.load(args.weights_a, weights_only=True, map_location='cpu')[args.key_a]
    weights_b = torch.load(args.weights_b, weights_only=True, map_location='cpu')[args.key_b]

    weights_interp = interpolate_weights(weights_a=weights_a, weights_b=weights_b, alpha=args.factor)

    if args.key_o:
        weights_interp = {
            args.key_o: weights_interp
        }

    torch.save(weights_interp, f'{args.output_name}.pth')
