#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export HI-SLAM2's Omnidata depth prior to TorchScript."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "third_party/HI-SLAM2/pretrained_models/omnidata_dpt_depth_v2.ckpt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "mono_priors/models/omnidata_dpt_depth_v2_512.pt",
    )
    parser.add_argument("--input-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    hislam2_python = root / "third_party/HI-SLAM2/hislam2"
    sys.path.insert(0, str(hislam2_python))

    from midas.omnidata import OmnidataModel

    if args.input_size <= 0 or args.input_size % 32 != 0:
        raise ValueError("--input-size must be a positive multiple of 32")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    wrapper = OmnidataModel("depth", str(args.checkpoint), device="cpu")
    model = wrapper.model.eval()
    example = torch.zeros(1, 3, args.input_size, args.input_size)
    with torch.inference_mode():
        reference = model(example)
        traced = torch.jit.trace(
            model, example, check_trace=False, strict=False
        ).eval()
        candidate = traced(example)
    max_error = (reference - candidate).abs().max().item()
    if max_error > 1.0e-5:
        raise RuntimeError(
            f"TorchScript parity check failed: max abs error {max_error}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(args.output))
    print(f"Saved: {args.output}")
    print(f"Input: 1x3x{args.input_size}x{args.input_size}")
    print(f"Max parity error: {max_error:.8g}")


if __name__ == "__main__":
    main()
