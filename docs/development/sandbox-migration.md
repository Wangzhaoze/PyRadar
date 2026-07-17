# Sandbox migration

The historical `pyradar_sandbox` directory was removed before v1.  Reusable
radar behavior was rewritten against current contracts; copied reference code,
GUI helpers, and hardware side effects were not moved into the installable
package.

| Historical area | v1 destination | Decision |
| --- | --- | --- |
| `base/*` | `pyradar.base` | Replaced by immutable validated models and named data containers. |
| `rsp/fft.py` | `pyradar.rsp.fft` | Rewritten with parameter and Radar-aware APIs. |
| `rsp/doa.py` | `pyradar.rsp.doa` | Rewritten; adds arbitrary geometry, duplicate phase centers, MUSIC, Capon, Bartlett, and ESPRIT. |
| `rsp/cfar.py`, STFT detection helpers | `pyradar.rsp.cfar` | Replaced by statistically parameterized CA/GOCA/SOCA/OS CFAR and NMS. Experimental undocumented variants were not promoted. |
| `rsp/stft.py` | `pyradar.rsp.time_frequency` | Replaced by general STFT, micro-Doppler, and zoom FFT results with physical coordinates. |
| clutter and integration helpers | `pyradar.rsp.preprocessing` | Reimplemented as composable axis-aware functions. |
| `sim/core.py`, `demo_sim.py` | `pyradar.sim` | Rebuilt as point-target/path ADC synthesis, scene sampling, range-image rendering, and a `Radar.simulate` entry point. |
| Mitsuba and experimental LiDAR ray scripts | `pyradar.sim.raytracing` | Replaced by backend-neutral ray math and an optional Trimesh CPU backend. Hard-coded assets, GUI state, and backend selection were removed. |
| converter helpers | `pyradar.utils.converter` and `Radar` properties | Rewritten with SI units, radians, exact FFT bin centers, and validation. |
| point-cloud helpers | `pyradar.utils.pointcloud` | Retains deterministic voxel/random sampling, bounds, and point-to-point ICP without requiring Open3D. |
| pose helpers | `pyradar.utils.geometry` | Retains FLU rigid transform creation, composition, inversion, and point transforms. Camera/Torch conversion utilities were outside radar scope. |
| heat-map helpers | `pyradar.utils.vis` | Replaced by typed RD, RA, and point-cloud plotting functions. |
| TI DCA1000/TSW1400 parsing | `pyradar.utils.io.ti` | Reimplemented as pure decoders and Radar-validated frame readers. Live UDP/serial control is intentionally not imported by the package. |
| copied `reference_mmwave` tree | no vendored package | Algorithm concepts are covered by native v1 implementations and tests. The third-party source remains available through its upstream project and is not exposed under the `pyradar` namespace. |

The migration is intentionally behavioral rather than path-compatible.  The
experimental 0.1 imports are not aliases in v1; callers should build a `Radar`,
use `pyradar.rsp`, `pyradar.sim`, or `pyradar.utils`, and rely on named dimensions
at raw-array boundaries.
