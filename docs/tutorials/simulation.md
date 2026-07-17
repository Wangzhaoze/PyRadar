# Radar-aware simulation

Simulation in `pyradar` starts from the same {class}`~pyradar.base.Radar` model
used to process measured ADC.  This is important: array phase, emission order,
slow-time spacing, waveform slope, and sample timing cannot silently disagree
between synthetic and recorded data.

## Point-target signal model

For transmitter $m$, receiver $n$, and a target at $\mathbf p(t)$, the bistatic
path length is

$$
L_{mn}(t)=\lVert\mathbf p(t)-\mathbf p_{T,m}\rVert
          +\lVert\mathbf p(t)-\mathbf p_{R,n}\rVert .
$$

Its derivative $\dot L_{mn}$ is positive for a receding path.  After ideal FMCW
dechirping, the simulator uses the narrowband approximation

$$
f_b \approx \frac{S L_{mn}}{c}-\frac{\dot L_{mn}}{\lambda},
\qquad
s_{mn}[k,e]=a\,q_m[e]\,
\exp\!\left(j2\pi\left[f_b\tau_k-\frac{L_{mn}(t_e)}{\lambda}\right]\right),
$$

where $S$ is chirp slope, $q_m[e]$ is the MIMO code at emission $e$, $t_e$ is
the real emission time, and $\tau_k$ is fast time relative to the sampled-band
center.  This convention makes approaching targets positive Doppler and matches
the steering vectors and TDM phase correction in `pyradar.rsp`.

```python
import numpy as np

from pyradar.sim import PointTarget

target = PointTarget(
    position=np.array([18.0, 2.0, 0.5]),
    velocity=np.array([-1.2, 0.0, 0.0]),
    rcs=4.0,
)
adc = radar.simulate(target, noisePower=1e-4, seed=3)
result = radar.process_adc(adc)
```

`simulate_adc` supports SIMO, TDM, BPM, and DDM through their strategy objects.
It does not contain a separate MIMO switch or assume that TX indices are emitted
in numerical order.

## Three abstraction levels

{class}`~pyradar.sim.PointTarget` is the default for processing tests and scene
prototypes.  It creates one path for every TX/RX pair from physical geometry.
Set `propagationLoss=True` to include ideal free-space field loss; leave it off
when testing only bin and phase recovery.

{class}`~pyradar.sim.PropagationPath` represents an already known path length,
path rate, complex gain, TX, and RX.  Use it for measured channel models or the
output of an external ray tracer:

```python
from pyradar.sim import PropagationPath, simulate_paths

paths = [
    PropagationPath(pathLength=24.0, pathRate=-2.0, txId=0, rxId=0),
    PropagationPath(
        pathLength=27.5,
        pathRate=-1.7,
        txId=0,
        rxId=0,
        amplitude=0.2j,
    ),
]
adc = simulate_paths(radar, paths)
```

{class}`~pyradar.sim.TrimeshRayTracer` is an optional CPU geometry backend.  It
loads `trimesh` lazily, traces intersections, and can produce piecewise SBR
paths.  Geometry and electromagnetic gain remain separate: a
{class}`~pyradar.sim.RayPath` must be converted to a propagation path before ADC
synthesis.  Install this backend with `python -m pip install -e ".[simulation]"`.

## Quantization and limits

{func}`~pyradar.sim.quantize_adc` quantizes real and imaginary components using
the sampler bit depth or an explicit bit depth.  It returns floating complex
values on quantization levels so downstream algorithms do not depend on a
particular packed integer format.

The v1 simulator assumes linear sawtooth FMCW, constant target velocity during a
frame, ideal dechirping, isotropic point targets, and no antenna pattern unless
it is included in a path gain.  Mesh tracing is deliberately not presented as a
full-wave solver.  Hardware nonlinearities and measured channel responses belong
in {class}`~pyradar.base.Calibration` or a future simulation backend.
