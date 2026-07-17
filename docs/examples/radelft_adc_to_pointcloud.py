"""Read one segmented RaDelft raw frame and produce radar products."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process one segmented RaDelft cascade ADC frame with pyradar."
    )
    parser.add_argument(
        "raw_directory",
        nargs="?",
        type=Path,
        default=Path(r"D:\Datasets\RaDelft\Scene6\RawData"),
    )
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/examples/output/radelft")
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    import matplotlib

    matplotlib.use("Agg")
    from pyradar.utils.io import RaDelftReader
    from pyradar.utils.vis import save_frame_figures

    reader = RaDelftReader(
        args.raw_directory,
        calibrationPath=args.calibration,
    )
    frame = reader.read(args.frame)
    result = reader.radar.build_pipeline().process(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "point_cloud.npy", result.pointCloud.to_numpy())
    outputs = save_frame_figures(result, args.output)
    print(f"ADC: {frame.data.shape}; points: {len(result.pointCloud)}")
    print(f"rangeBinSize: {reader.radar.rangeBinSize:.6f} m")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
