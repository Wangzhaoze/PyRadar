# FMCW physics and axes

This tutorial states the conventions used by `FMCW`, `Sampler`, and the derived
properties on `Radar`. The v1 implementation assumes a linear sawtooth chirp.

## Transmitted and received phase

During the ramp, instantaneous transmit frequency is

$$
f_\mathrm{tx}(t) = f_0 + S t,
$$

where $f_0$ is `startFrequency` in hertz and $S$ is `slope` in hertz per
second. Ignoring constant phase, the complex transmit signal is

$$
s_\mathrm{tx}(t) = \exp\left[j2\pi\left(f_0t + \frac{S}{2}t^2\right)\right].
$$

A target at range $R$ introduces round-trip delay $\tau=2R/c$. After dechirping,
the dominant stationary-target beat frequency is

$$
f_b \approx S\tau = \frac{2SR}{c},
\qquad
R \approx \frac{c f_b}{2S}.
$$

Motion also contributes Doppler. The narrowband approximation
$f_D=2v_r/\lambda$ is used for the slow-time velocity axis. Range-Doppler
coupling is not corrected automatically in v1 and should be considered for long,
fast chirps or high velocities.

## Sampled bandwidth and range

If $N_s$ samples are acquired at rate $f_s$, capture duration and sampled chirp
bandwidth are

$$
T_s = \frac{N_s}{f_s},
\qquad
B_\mathrm{sampled}=S T_s.
$$

The physical range resolution is

$$
\Delta R_\mathrm{resolution} = \frac{c}{2B_\mathrm{sampled}}.
$$

For an $N_R$-point range FFT, bin spacing is

$$
\Delta R_\mathrm{bin} = \frac{c f_s}{2 S N_R}.
$$

These are equal only when $N_R=N_s$. Zero padding reduces bin spacing but does
not resolve two targets inside the waveform's physical resolution.

With complex sampling, v1 reports

$$
R_\max = \frac{c f_s}{2S}.
$$

For real sampling, the usable one-sided beat bandwidth is $f_s/2$, so the model
halves this limit. Front-end analogue bandwidth may impose a smaller practical
limit and belongs in capture-profile validation.

## Slow time and velocity

Let $T_\mathrm{slow}$ be the interval between two decoded samples for the same
virtual channel. It depends on MIMO:

- SIMO: one chirp interval;
- TDM: one complete TX emission cycle;
- BPM: one two-code block;
- DDM: one configured code period.

For $N_D$ decoded slow-time samples,

$$
\Delta v_\mathrm{resolution}
= \frac{\lambda}{2N_D T_\mathrm{slow}},
\qquad
v_\max = \frac{\lambda}{4T_\mathrm{slow}}.
$$

If the Doppler FFT has $N_V$ bins, its displayed spacing is
$\lambda/(2N_VT_\mathrm{slow})$. The shifted velocity axis is centred at zero.

## Model check

```python
print(f"sampled bandwidth: {radar.sampledBandwidth / 1e9:.3f} GHz")
print(f"range resolution:  {radar.rangeResolution:.3f} m")
print(f"range bin:         {radar.rangeBinSize:.3f} m")
print(f"velocity bin:      {radar.velocityBinSize:.3f} m/s")
```

The model rejects an ADC capture whose `adcStartTime + captureDuration` extends
past `rampEndTime`. This catches a common unit or profile mismatch before an axis
is generated.
