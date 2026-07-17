# Data adapters and licensing

Dataset adapters live under `pyradar.utils.io`. They are responsible only for
binary/MAT decoding, metadata parsing, calibration loading, and construction of
an `ADCFrame` with a `Radar`. No dataset name or hardware branch appears in
`pyradar.rsp`.

| Adapter | Capture | Redistribution in this repository |
| --- | --- | --- |
| `ColoRadarReader` | TI four-chip cascade binary ADC | one Apache-2.0 frame |
| `ColoRadarPlusReader` | ColoRadar+ capture | local path only |
| `RaDelftReader` | segmented four-device cascade ADC | local path only |
| `RAMPCNNReader` | AWR1843 MATLAB ADC | local path only |

## Local examples

The real-data scripts accept a dataset root and write plots to a caller-selected
output directory:

```console
python docs/examples/radelft_adc_to_pointcloud.py D:/Datasets/RaDelft --frame 0
python docs/examples/coloradar_adc_to_pointcloud.py --data-root D:/Datasets/ColoRadarPlus --frame 0
python docs/examples/rampcnn_adc_to_pointcloud.py D:/Datasets/RAMPCNN --frame 0
```

Use paths appropriate to the local extraction layout shown by each script's
`--help` output.

## Redistribution policy

- ColoRadar identifies its dataset and code under Apache-2.0. The committed
  frame retains that license and is listed by SHA256 in
  `docs/examples/data/manifest.json`.
- RAMPCNN/CRUW download terms restrict use to academic or non-profit research.
  Those files are never committed.
- The local RaDelft material does not contain a single repository-wide grant for
  redistribution, so it is referenced by path only.
- The ColoRadar+ project page does not currently state a dataset redistribution
  license, so it is referenced by path only.

Users remain responsible for accepting the terms of each original dataset.
The relevant project pages are [ColoRadar](https://arpg.colorado.edu/coloradar/),
[ColoRadar+](https://arpg.github.io/coloradarplus/), and
[CRUW](https://www.cruwdataset.org/download).
