# Release notes

## 1.0.0rc1

This release candidate replaces the experimental 0.1 API with an immutable,
radar-model-centric architecture.

- Adds SIMO, arbitrary-order TDM, two-channel BPM, and phase-coded DDM models.
- Adds named ADC/cube dimensions and typed detection, point-cloud, cluster, track,
  and frame results.
- Adds parameter and model-aware FFT APIs, four CFAR families, geometry-aware DoA,
  deterministic DBSCAN, and 2D/3D tracking.
- Adds TI AWR2243 cascade and AWR1843 hardware profiles plus four dataset adapters.
- Standardizes on SI units, radians, FLU coordinates, camelCase model fields, and
  snake_case algorithm functions.

The 0.1 API is intentionally not retained. Release candidates are published as
GitHub wheel/sdist assets and are not uploaded to PyPI.
