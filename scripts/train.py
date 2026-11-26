"""Train a selected model on externally prepared paired tensors."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from engine.io import load_config, load_pairs
from engine.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Model JSON configuration")
    parser.add_argument("--pairs", required=True, help="External NPZ with input and target arrays")
    parser.add_argument("--output", required=True, help="New checkpoint path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    image, target = load_pairs(args.pairs)
    result = train(load_config(args.config), image, target, args.output, device=args.device, steps=args.steps, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
