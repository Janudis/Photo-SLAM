#!/usr/bin/env python3
"""Run the HI-SLAM2 reconstruction batch at a 1 cm distance threshold."""

from evaluate_reconstructions_hislam2 import main


if __name__ == "__main__":
    raise SystemExit(main(distance_threshold_m=0.01))
