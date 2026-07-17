"""Process one AWR1843 RAMPCNN/CRUW MATLAB ADC frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process one RAMPCNN/CRUW AWR1843 MATLAB ADC frame."
    )
    parser.add_argument(
        "sequence",
        nargs="?",
        type=Path,
        default=Path(r"D:\Datasets\RAMPCNN\2019_05_29_pcms005"),
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/examples/output/rampcnn")
    )
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    from pyradar.utils.io import RAMPCNNReader
    from pyradar.utils.vis import save_frame_figures

    reader = RAMPCNNReader(args.sequence)
    frame = reader.read(args.frame)
    result = reader.radar.build_pipeline().process(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "point_cloud.npy", result.pointCloud.to_numpy())
    outputs = save_frame_figures(result, args.output)
    print(f"ADC: {frame.data.shape}; points: {len(result.pointCloud)}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
