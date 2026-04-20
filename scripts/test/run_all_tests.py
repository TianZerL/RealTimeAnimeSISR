import argparse
import os

from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--options_dir', type=str, help="options dir")
    args = parser.parse_args()

    options_dir = Path(args.options_dir)
    for opt in tqdm(options_dir.rglob(f'*.yml')):
        print(f'test for {opt}:\n\n')
        os.system(f'python test.py -opt {opt}')
        print(f'finished {opt}.\n\n')
