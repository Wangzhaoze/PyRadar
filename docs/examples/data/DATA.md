# Example data

This directory contains one original ColoRadar four-chip cascade ADC frame and
the metadata required to decode it. It exists only as a separately downloadable
example asset and is pruned from both the pyradar wheel and source distribution.

## Included capture

- Dataset: ColoRadar
- Sequence: `2_28_2021_outdoors_run0`
- Sensor: TI AWR2243 four-chip cascade
- Frame: `frame_0.bin`
- Upstream project: <https://arpg.colorado.edu/coloradar/>
- Dataset/code license stated upstream: Apache-2.0

Binary files are stored through Git LFS. File sizes and SHA256 digests are listed
in `manifest.json`; verify them after downloading an example release asset.

From the repository root:

```console
python docs/examples/coloradar_adc_to_pointcloud.py \
  --data-root docs/examples/data/ColoRadar \
  --frame 0
```

No RaDelft, ColoRadar+, or RAMPCNN/CRUW capture is redistributed here. Their
example scripts require a local dataset path because their redistribution terms
are absent or more restrictive.
