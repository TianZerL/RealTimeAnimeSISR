import re, argparse, csv
from typing import Dict, Union, Optional, List
from pathlib import Path

def parse_test_log(log_file_path: str) -> Dict[str, Union[str, Dict[str, Dict[str, float]]]]:
    model_name: Optional[str] = None
    results: Dict[str, Dict[str, float]] = {}
    current_dataset: Optional[str] = None

    pth_pattern = re.compile(r'pretrain_network_g: .+/([^/]+)\.pth')
    net_pattern = re.compile(r'Network \[(\w+)\] is created\.')

    dataset_pattern = re.compile(r'Validation (\S+)')
    psnr_pattern = re.compile(r'# psnr:\s*([\d\.]+)')
    ssim_pattern = re.compile(r'# ssim:\s*([\d\.]+)')

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if model_name is None:
                m = pth_pattern.search(line)
                if m:
                    model_name = m.group(1)
                    continue

            if model_name is None:
                m = net_pattern.search(line)
                if m:
                    model_name = m.group(1)
                    continue

            m = dataset_pattern.search(line)
            if m:
                current_dataset = m.group(1)
                results[current_dataset] = {}
                continue

            m = psnr_pattern.search(line)
            if m and current_dataset is not None:
                results[current_dataset]['psnr'] = float(m.group(1))
                continue

            m = ssim_pattern.search(line)
            if m and current_dataset is not None:
                results[current_dataset]['ssim'] = float(m.group(1))

    return {
        "model_name": model_name if model_name is not None else "Unknown",
        "results": results
    }

def save_to_csv(records: List[Dict], output_path: str):
    if not records:
        return
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help="result dir")
    parser.add_argument('-o', '--output_name', type=str, help="output csv file name")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    records = []
    for log_file in input_dir.rglob('*.log'):
        if ('archive' in log_file.parent.name.lower()):
            continue
        result = parse_test_log(log_file)
        record = { "model_name": result['model_name'] }
        for dataset, metrics in result["results"].items():
            record[f'{dataset}_psnr'] = metrics["psnr"]
            record[f'{dataset}_ssim'] = metrics["ssim"]
        records.append(record)

    save_to_csv(records, f'{args.output_name}.csv')
