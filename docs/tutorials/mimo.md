# SIMO, TDM, BPM, and DDM

Each MIMO strategy receives canonical ADC and returns decoded virtual channels.
It owns both `channelMap` and slow-time timing, so downstream code does not need
hardware branches.

## Virtual channels

For far-field monostatic MIMO, the phase centre for TX $i$ and RX $j$ is

$$
\mathbf{p}_{ij}=\mathbf{p}^{\mathrm{tx}}_i
                 +\mathbf{p}^{\mathrm{rx}}_j.
$$

`channelMap` defines which `(txId, rxId)` pair occupies each decoded channel.
Geometry is built from this map rather than assuming TX-major ordering elsewhere.

## SIMO

SIMO has one transmitter and one emission per loop. Decoding removes only the
singleton emission axis:

```python
mimo = SIMO(numRx=4)
```

## TDM

TDM records one emission per active TX in every loop. `txOrder` is the physical
TX id fired at each emission index and may be arbitrary:

```python
mimo = TDM(numTx=3, numRx=4, txOrder=(2, 0, 1))
```

A moving target accumulates extra phase because TX channels are sampled at
different times. If channel $m$ was emitted at offset $\tau_m$ and the shifted
Doppler bin corresponds to $f_D$, the library applies

$$
C_m(f_D)=\exp(-j2\pi f_D\tau_m).
$$

Offsets come from `emissionTimeOffsets` when supplied; otherwise they are derived
from emission index and chirp interval. Compensation therefore follows real TX
timing and does not assume an ascending TX id.

The ordinary TDM velocity limit is based on one full emission cycle. When
`Calibration.overlapPairs` contains repeated phase centres, the model-aware
pipeline evaluates Doppler candidates separated by the slow-time PRF. Each
candidate is corrected with the real TX timestamps, and the candidate minimizing
calibrated overlap-channel phase error supplies point-cloud radial velocity and
the final DoA correction. Candidates are limited to the raw chirp-rate Nyquist
interval. If overlap energy is zero, order zero is retained rather than guessed.

This behavior is controlled by `PointCloudConfig.unwrapTdmVelocity`. The selected
integer alias order is recorded in `FrameResult.metadata["dopplerAmbiguityOrder"]`.

## BPM

Two-TX BPM uses two coded emissions. With the default Hadamard matrix,

$$
\begin{bmatrix}y_0\\y_1\end{bmatrix}
=
\begin{bmatrix}1&1\\1&-1\end{bmatrix}
\begin{bmatrix}x_0\\x_1\end{bmatrix},
\qquad
\mathbf{x}=\mathbf{H}^{-1}\mathbf{y}.
$$

```python
mimo = BPM(numRx=4)
```

The code matrix is validated as invertible. The target should not change
materially over the two coded chirps.

## DDM

DDM transmits all configured TX channels in one emission using slow-time linear
phase codes

$$
c_i[\ell]=\exp(j2\pi q_i\ell),
$$

where `dopplerOffsets[i]` is $q_i$ in cycles per raw chirp. Codes must be
orthogonal over `codeLength`. The decoder correlates each complete code period:

$$
\hat{x}_i[b]=\frac{1}{L}\sum_{\ell=0}^{L-1}
c_i^*[\ell]y[bL+\ell].
$$

```python
mimo = DDM(
    numTx=4,
    numRx=4,
    dopplerOffsets=(0.0, 0.25, 0.5, 0.75),
    codeLength=4,
)
```

The number of raw loops must be divisible by `codeLength`, and the target is
assumed approximately constant during one code block.

## Numerical contract

All four decoders accept `(loop, emission, rx, ...)`; wrong emission or RX sizes
raise `ValueError`. Synthetic noiseless encode/decode tests require relative error
below $10^{-6}$.
