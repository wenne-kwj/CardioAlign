"""Evaluate matching external prediction and reference tensors."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from engine.io import load_config, load_tensor, write_json, _json_value
from engine.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--target", required=True, help="External NPY or NPZ target array")
    parser.add_argument("--config", default=str(ROOT / "configs/evaluation/reconstruction.json"))
    parser.add_argument("--output", required=True, help="New metrics JSON path")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("--output must be a new file")
    torch.set_num_threads(args.threads)
    prediction = load_tensor(args.prediction, key="prediction", channels=3)
    target = load_tensor(args.target, key="target", channels=3)
    with torch.inference_mode():
        result = evaluate(prediction, target, load_config(args.config))
    write_json(args.output, result)
    print(json.dumps(_json_value(result["overall"]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
