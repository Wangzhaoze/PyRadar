# Getting Started

## Install

`pyradar` v1 release candidates are distributed on GitHub Releases rather than
PyPI because that distribution name is already occupied. Install a downloaded
wheel with:

```console
python -m pip install pyradar-1.0.0rc1-py3-none-any.whl
```

For a source checkout:

```console
python -m pip install -e ".[examples]"
```

Python 3.10 through 3.13 are supported.

## Define the radar first

The minimum custom model combines five physical pieces. Positions are metres in
FLU and times are seconds.

```python
import numpy as np

from pyradar.base import FMCW, Radar, Sampler, SIMO, Transceivers

waveform = FMCW(
    startFrequency=77e9,
    slope=40e12,
    adcStartTime=4e-6,
    rampEndTime=36e-6,
    idleTime=8e-6,
)
sampler = Sampler(
    numSamples=256,
    numLoops=64,
    sampleRate=10e6,
)
rx = np.column_stack(
    (np.zeros(8), np.arange(8) * 1.95e-3, np.zeros(8))
)
radar = Radar(
    waveform=waveform,
    sampler=sampler,
    transceivers=Transceivers(
        txPositions=np.array([[0.0, 0.0, 0.0]]),
        rxPositions=rx,
    ),
    mimo=SIMO(numRx=8),
    name="lab_ula",
)
```

The model now derives its axes and limits:

```python
print(radar.rangeResolution, radar.rangeBinSize)
print(radar.velocityResolution, radar.maxUnambiguousVelocity)
print(radar.arrayGeometry, radar.unambiguousFov)
```

## Process one frame

Describe dimensions at the point where the raw array enters the library:

```python
result = radar.build_pipeline().process(
    adc,
    dims=("loop", "emission", "rx", "sample"),
    timestamp=12.4,
    frameId=31,
)

xyzPowerSnrVelocity = result.pointCloud.to_numpy()
```

The canonical ADC shape is always `loop/emission/rx/sample`. A missing dimension
is inserted only when the model proves it is a singleton. This avoids silently
swapping receivers and chirps when two axes happen to have equal lengths.

## Simulate a modeled target

The same radar can generate canonical ADC for a point target. MIMO codes,
emission timing, bistatic TX/RX geometry, and sampled-band phase all come from
the model:

```python
from pyradar.sim import PointTarget

target = PointTarget(
    position=np.array([20.0, 3.0, 0.5]),
    velocity=np.array([-2.0, 0.0, 0.0]),
    rcs=5.0,
)
synthetic = radar.simulate(target, noisePower=1e-4, seed=8)
syntheticResult = radar.process_adc(synthetic)
```

See [Radar-aware simulation](tutorials/simulation.md) for propagation paths,
quantization, scene sampling, and optional mesh ray tracing.

## Load YAML or JSON

{meth}`pyradar.base.Radar.from_config` accepts the same camelCase field names as
the Python constructors:

```yaml
name: lab_ula
waveform:
  startFrequency: 77000000000.0
  slope: 40000000000000.0
  adcStartTime: 0.000004
  rampEndTime: 0.000036
  idleTime: 0.000008
sampler:
  numSamples: 256
  numLoops: 64
  sampleRate: 10000000.0
transceivers:
  txPositions: [[0.0, 0.0, 0.0]]
  rxPositions:
    - [0.0, 0.000000, 0.0]
    - [0.0, 0.001950, 0.0]
mimo:
  type: simo
processing:
  fft:
    rangeFftSize: 512
    dopplerFftSize: 64
    rangeWindow: hann
  cfar:
    method: os
    pfa: 0.0001
```

## Dataset adapters

Readers decode storage layout and create an `ADCFrame` plus an appropriate radar
profile. Signal processing remains in `pyradar.rsp`:

```python
from pyradar.utils.io import ColoRadarReader

reader = ColoRadarReader(datasetRoot)
frame = reader.read_frame(0)
result = frame.radar.process_adc(frame)
```

See [Data and licenses](data-and-licenses.md) for expected local layouts.

## Continue from here

```{toctree}
:maxdepth: 1

radar-model
data-and-licenses
```
