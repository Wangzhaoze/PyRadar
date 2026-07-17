# Native and GPU acceleration roadmap

The v1 public API is backend-neutral but only the NumPy/SciPy implementation is
released. Native code should be introduced only after profiling demonstrates a
repeatable bottleneck on representative ULA and 192-channel 4D frames.

## Stable boundary

Acceleration must preserve these Python contracts:

- input arrays, named dimensions, dtype behavior, and validation errors;
- `RadarCube`, `CFARResult`, `DoAResult`, and point-cloud result semantics;
- CPU/GPU numerical tolerances and FFT normalization;
- deterministic ordering for detections, clusters, and tracks.

Backend selection should be explicit, for example `backend="numpy"`,
`backend="native"`, or `backend="cupy"`. It must never depend on dataset names.

## Candidate CPU stack

Use `pybind11` with `scikit-build-core` and CMake for kernels that cannot be
expressed efficiently through NumPy/SciPy. The first candidates are batched
sparse-array beamforming, large OS-CFAR rank windows, and dense RDA formation.
FFTs should continue using a proven FFT provider rather than a custom transform.

Native extensions must:

1. accept contiguous and strided arrays or make the copy visible in benchmarks;
2. release the GIL during long kernels;
3. validate shape, dtype, and overflow at the Python boundary;
4. expose no hardware- or dataset-specific entry points;
5. retain a pure-Python reference implementation for correctness tests.

## CUDA/CuPy backend

The GPU path should use an array namespace layer and CuPy primitives for FFT,
matrix products, covariance, and beamforming. Host/device transfers must occur at
pipeline boundaries, not once per detection. Tracking remains on CPU until its
cost is material because its state is small and association is sequential.

CUDA support should be an optional package extra or separate release asset. A
CPU-only installation must not import CuPy or probe a CUDA driver.

## Benchmark gate

Add `asv` or `pytest-benchmark` cases for:

- 64 x 1 x 8 x 256 SIMO ULA;
- 128 x 12 x 16 x 256 TDM cascade;
- detection-first sparse DoA at 32, 128, and 512 detections;
- optional dense 512 x 128 x 181 RDA output.

A native backend is enabled by default only when it is at least 2x faster at the
pipeline level, stays within the documented numerical tolerance, and does not
increase installation failures. Kernel-only speedups do not satisfy this gate.

## Wheel matrix

Future native releases need CPython 3.10-3.13 wheels for Windows x86-64, Linux
manylinux x86-64/aarch64, and macOS x86-64/arm64. Build with `cibuildwheel`, test
each wheel in a clean environment, inspect shared-library dependencies, and keep
the source distribution capable of falling back to pure Python. CUDA wheels
require a separate matrix keyed by supported CUDA major version.
