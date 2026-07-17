"""ColoRadar TI2243 ADC to RD, RA, and a velocity point cloud.

Run from the repository root::

    python docs/examples/coloradar_adc_to_pointcloud.py --data-root D:/Datasets/ColoRadar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process one ColoRadar cascade ADC frame with pyradar."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("docs/examples/data/ColoRadar"),
    )
    parser.add_argument("--sequence", default="2_28_2021_outdoors_run0")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/examples/output/coloradar")
    )
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    from matplotlib import pyplot as plt

    from pyradar.utils.io import ColoRadarReader
    from pyradar.utils.vis import save_frame_figures

    reader = ColoRadarReader(
        args.data_root,
        args.sequence,
        applyCalibration=not args.no_calibration,
    )
    frame = reader.read(args.frame)
    result = reader.radar.build_pipeline().process(frame)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "point_cloud.npy", result.pointCloud.to_numpy())
    outputs = save_frame_figures(result, args.output)
    print(f"Radar: {reader.radar.hardware} / {reader.radar.profileName}")
    print(f"ADC: {frame.data.shape} {frame.dims}")
    rdShape = (
        result.rangeDopplerCube.shape if result.rangeDopplerCube is not None else None
    )
    raShape = result.rangeAngleCube.shape if result.rangeAngleCube is not None else None
    print(f"RD: {rdShape}")
    print(f"RA: {raShape}")
    print(f"Points: {len(result.pointCloud)}")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
