"""Predict fluorescence using a model checkpoint and external input tensors."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from engine.io import load_tensor
from engine.inference import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="External NPY or NPZ input array")
    parser.add_argument("--output", required=True, help="New prediction NPY path")
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix != ".npy" or output.exists():
        parser.error("--output must be a new .npy file")
    torch.set_num_threads(args.threads)
    image = load_tensor(args.input, key="input", channels=1)
    prediction = predict(args.checkpoint, image, device=args.device, weights=args.weights, seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, prediction.numpy(), allow_pickle=False)
    print(f"Saved prediction array with shape {tuple(prediction.shape)}")


if __name__ == "__main__":
    main()
