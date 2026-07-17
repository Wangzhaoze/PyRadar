# Calibration stages

Calibration is split by the domain where a correction is physically meaningful.
Every configured array is shape-checked; a mismatch raises an error rather than
silently disabling calibration.

## ADC domain

`adcFrequencyResponse` and `adcChannelGain` act before range FFT. Frequency
response corrects sample-domain complex gain, while channel gain corrects
emission/RX response. Arrays must broadcast exactly according to the documented
canonical ADC axes.

This stage is appropriate for measured front-end gain/phase response and static
per-channel imbalance represented in the capture's native channel layout.

## Range domain

After range FFT and MIMO decoding, `rangeCoupling` is subtracted from
`(loop, virtual, range)` data. It models direct TX/RX leakage and static coupling
as a complex range profile:

$$
X_\mathrm{clean}[\ell,m,k]
=X[\ell,m,k]-C[m,k].
$$

Subtracting a slow-time mean is not equivalent: mean removal also removes all
stationary scene reflectors.

## Array domain

`arrayPhase` multiplies virtual channels after Doppler compensation and before
DoA:

$$
\tilde{X}[r,d,m]=X[r,d,m]G_m.
$$

The gain convention is correction gain, not measured error. Calibrate before
coherently merging duplicate phase centres.

## Capture profiles

Dataset adapters parse native JSON, MAT, or text calibration files and construct
`Calibration`. Parsed physical units must be converted to SI once at the reader
boundary. A profile should record source filenames and calibration assumptions in
frame metadata for reproducibility.

Overlap antenna pairs identify repeated phase centres used for TDM velocity
disambiguation. The implementation evaluates candidates within the raw chirp-rate
Nyquist interval and selects the calibrated phase-consistency minimum. Invalid
pair indices or array-calibration shapes are errors; zero-energy evidence falls
back to ambiguity order zero.
