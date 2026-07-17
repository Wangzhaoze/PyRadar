"""Radar-model-centric ADC-to-track processing pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyradar.base.cube import (
    ADCFrame,
    DetectionSet,
    FrameResult,
    RadarCube,
)

from .cfar import cfar_2d
from .clustering import dbscan
from .doa import estimate_doa, steering_vector
from .fft import doppler_fft, range_fft
from .mimo import unwrap_tdm_velocity
from .pointcloud import detections_to_pointcloud
from .tracking import MultiTargetTracker

if TYPE_CHECKING:
    from pyradar.base.radar import Radar


class ADCToPointCloudPipeline:
    """Stateful detection-first processing pipeline built from one Radar.

    Processing order is ADC calibration, range FFT, MIMO decode, range-domain
    calibration, Doppler FFT, RD CFAR, MIMO phase correction, DoA, point cloud,
    clustering, and tracking. Dense RA/RDA products are optional.
    """

    def __init__(self, radar: Radar) -> None:
        self.radar = radar
        self._tracker = (
            MultiTargetTracker(radar.processing.tracking)
            if radar.processing.tracking.enabled
            else None
        )

    def reset(self) -> None:
        """Reset stateful downstream algorithms such as tracking."""

        if self._tracker is not None:
            self._tracker.reset()

    def process(
        self,
        adc: ArrayLike | ADCFrame,
        *,
        dims: Sequence[str] | None = None,
        timestamp: float | None = None,
        frameId: int | str | None = None,
    ) -> FrameResult:
        """Process one ADC frame using the attached Radar model."""

        if isinstance(adc, ADCFrame):
            frame = adc.canonical(self.radar)
            if timestamp is not None or frameId is not None:
                frame = ADCFrame(
                    frame.data,
                    frame.dims,
                    radar=self.radar,
                    timestamp=timestamp if timestamp is not None else frame.timestamp,
                    frameId=frameId if frameId is not None else frame.frameId,
                    metadata=frame.metadata,
                )
        else:
            frame = ADCFrame.from_array(
                adc,
                dims=dims,
                radar=self.radar,
                timestamp=timestamp,
                frameId=frameId,
            ).canonical(self.radar)

        calibratedAdc = self.radar.calibration.apply_adc(frame.data)
        calibratedFrame = ADCFrame(
            calibratedAdc,
            frame.dims,
            radar=self.radar,
            timestamp=frame.timestamp,
            frameId=frame.frameId,
            metadata=frame.metadata,
        )
        rangeResult = range_fft(calibratedFrame, radar=self.radar)
        if not isinstance(rangeResult, RadarCube):
            raise RuntimeError("Model-aware range_fft did not return a RadarCube.")

        rangeVirtual = self.radar.mimo.decode(rangeResult.data)
        rangeVirtual = self.radar.calibration.apply_range(rangeVirtual)
        virtualCube = RadarCube(
            data=rangeVirtual,
            dims=("loop", "virtual", "range"),
            stage="range_mimo",
            radar=self.radar,
            coords={"range": rangeResult.coords["range"]},
            metadata={"channelMap": self.radar.mimo.channelMap},
        )
        dopplerResult = doppler_fft(virtualCube, radar=self.radar, loopAxis=0)
        if not isinstance(dopplerResult, RadarCube):
            raise RuntimeError("Model-aware doppler_fft did not return a RadarCube.")
        rdFull = dopplerResult.transpose("range", "doppler", "virtual")

        correction = self.radar.mimo.phase_correction(
            rdFull.shape[1], self.radar.waveform.chirpInterval
        )
        rangeBins = self.radar.rangeBinIndices
        uncorrected = rdFull.data[rangeBins]
        corrected = rdFull.data * correction[None, :, :]
        corrected = self.radar.calibration.apply_array(corrected, channelAxis=2)
        corrected = corrected[rangeBins]
        rdCube = RadarCube(
            data=corrected,
            dims=("range", "doppler", "virtual"),
            stage="range_doppler",
            radar=self.radar,
            coords={
                "range": self.radar.rangeAxis,
                "doppler": self.radar.velocityAxis,
            },
            metadata={
                "rangeBinIndices": rangeBins,
                "channelMap": self.radar.mimo.channelMap,
                "tdmPhaseCompensated": self.radar.mimo.scheme == "tdm",
            },
        )

        power = np.mean(np.abs(rdCube.data) ** 2, axis=2)
        detections, localBins = self._detect(power, rangeBins)
        radialVelocity, phaseAdjustment, ambiguityOrders = self._resolve_velocities(
            uncorrected, localBins
        )
        azimuth, elevation, azimuthBins, elevationBins = self._estimate_angles(
            rdCube.data, localBins, phaseAdjustment
        )
        points = detections_to_pointcloud(
            detections,
            radar=self.radar,
            azimuth=azimuth,
            elevation=elevation,
            azimuthBin=azimuthBins,
            elevationBin=elevationBins,
            radialVelocity=radialVelocity,
            timestamp=frame.timestamp,
            frameId=frame.frameId,
        )

        clusters = None
        clusterConfig = self.radar.processing.clustering
        if clusterConfig.enabled:
            clusters = dbscan(
                points,
                eps=clusterConfig.eps,
                minSamples=clusterConfig.minSamples,
                velocityScale=clusterConfig.velocityScale,
                dimensions=clusterConfig.dimensions,
            )
        tracks = None
        if self._tracker is not None:
            deltaTime = (
                self.radar.sampler.framePeriod if frame.timestamp is None else None
            )
            tracks = self._tracker.update(
                points,
                clusters=clusters,
                timestamp=frame.timestamp,
                deltaTime=deltaTime,
            )

        rangeAngleCube, rdaCube = self._dense_angle_products(rdCube)
        return FrameResult(
            adcFrame=frame,
            rangeCube=virtualCube,
            rangeDopplerCube=rdCube
            if self.radar.processing.retainRangeDoppler
            else None,
            rangeAngleCube=rangeAngleCube,
            rangeDopplerAngleCube=rdaCube,
            detections=detections,
            pointCloud=points,
            clusters=clusters,
            tracks=tracks,
            metadata={
                "coordinateFrame": "FLU",
                "units": {"distance": "m", "velocity": "m/s", "angle": "rad"},
                "dopplerAmbiguityOrder": ambiguityOrders,
            },
        )

    def _resolve_velocities(
        self,
        uncorrected: NDArray[np.complex128],
        localBins: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128], NDArray[np.int64]]:
        count = localBins.shape[0]
        channels = uncorrected.shape[-1]
        velocity = np.empty(count, dtype=float)
        adjustment = np.ones((count, channels), dtype=np.complex128)
        orders = np.zeros(count, dtype=np.int64)
        if count == 0:
            return velocity, adjustment, orders
        velocity[:] = self.radar.velocityAxis[localBins[:, 1]]
        overlaps = self.radar.calibration.overlapPairs
        enabled = self.radar.processing.pointCloud.unwrapTdmVelocity
        if (
            self.radar.mimo.scheme != "tdm"
            or not enabled
            or overlaps is None
            or overlaps.shape[0] == 0
        ):
            return velocity, adjustment, orders
        aliasCorrection = self.radar.mimo.phase_correction(
            uncorrected.shape[1], self.radar.waveform.chirpInterval
        )
        dopplerSize = uncorrected.shape[1]
        for outputId, (rangeId, dopplerId) in enumerate(localBins):
            rangeSelection = slice(
                max(0, rangeId - 1), min(uncorrected.shape[0], rangeId + 2)
            )
            dopplerSelection = np.asarray(
                [
                    (dopplerId - 1) % dopplerSize,
                    dopplerId,
                    (dopplerId + 1) % dopplerSize,
                ]
            )
            snapshots = uncorrected[rangeSelection][:, dopplerSelection, :].reshape(
                -1, channels
            )
            decision = unwrap_tdm_velocity(
                snapshots,
                radar=self.radar,
                dopplerBin=int(dopplerId),
            )
            velocity[outputId] = decision.velocity
            orders[outputId] = decision.ambiguityOrder
            adjustment[outputId] = decision.phaseCorrection / aliasCorrection[dopplerId]
        return velocity, adjustment, orders

    def _detect(
        self,
        power: NDArray[np.float64],
        rangeBins: NDArray[np.int64],
    ) -> tuple[DetectionSet, NDArray[np.int64]]:
        config = self.radar.processing.cfar
        result = cfar_2d(
            power,
            method=config.method,
            trainingCells=config.trainingCells,
            guardCells=config.guardCells,
            pfa=config.pfa,
            rankFraction=config.rankFraction,
            minSnrDb=config.minSnrDb,
            peakGrouping=config.peakGrouping,
        )
        mask = result.detections.copy()
        pointConfig = self.radar.processing.pointCloud
        ranges = rangeBins * self.radar.rangeBinSize
        mask[ranges < pointConfig.minRange] = False
        if pointConfig.maxRange is not None:
            mask[ranges > pointConfig.maxRange] = False
        local = np.argwhere(mask)
        if local.size == 0:
            return DetectionSet.empty(), np.empty((0, 2), dtype=np.int64)
        snr = result.snr[local[:, 0], local[:, 1]]
        order = np.argsort(-snr, kind="stable")
        if config.maxDetections is not None:
            order = order[: config.maxDetections]
        local = local[order]
        rangeIndex, dopplerIndex = local[:, 0], local[:, 1]
        return (
            DetectionSet(
                rangeBin=rangeBins[rangeIndex],
                dopplerBin=dopplerIndex,
                power=power[rangeIndex, dopplerIndex],
                noise=result.noise[rangeIndex, dopplerIndex],
                threshold=result.threshold[rangeIndex, dopplerIndex],
                snr=result.snr[rangeIndex, dopplerIndex],
            ),
            local,
        )

    def _estimate_angles(
        self,
        rd: NDArray[np.complex128],
        localBins: NDArray[np.int64],
        phaseAdjustment: NDArray[np.complex128],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
        count = localBins.shape[0]
        azimuth = np.empty(count, dtype=float)
        elevation = np.empty(count, dtype=float)
        azimuthBins = np.empty(count, dtype=np.int64)
        elevationBins = np.empty(count, dtype=np.int64)
        dopplerSize = rd.shape[1]
        for outputId, (rangeId, dopplerId) in enumerate(localBins):
            rangeSelection = slice(max(0, rangeId - 1), min(rd.shape[0], rangeId + 2))
            dopplerSelection = np.asarray(
                [
                    (dopplerId - 1) % dopplerSize,
                    dopplerId,
                    (dopplerId + 1) % dopplerSize,
                ]
            )
            snapshots = rd[rangeSelection][:, dopplerSelection, :].reshape(
                -1, rd.shape[2]
            )
            snapshots = snapshots * phaseAdjustment[outputId]
            estimate = estimate_doa(snapshots, radar=self.radar, numSources=1)
            azimuth[outputId] = estimate.azimuth[0]
            elevation[outputId] = estimate.elevation[0]
            azimuthBins[outputId] = int(
                np.argmin(np.abs(estimate.azimuthAxis - estimate.azimuth[0]))
            )
            elevationBins[outputId] = int(
                np.argmin(np.abs(estimate.elevationAxis - estimate.elevation[0]))
            )
        return azimuth, elevation, azimuthBins, elevationBins

    def _dense_angle_products(
        self, rdCube: RadarCube
    ) -> tuple[RadarCube | None, RadarCube | None]:
        processing = self.radar.processing
        if not (processing.retainRangeAngle or processing.retainRangeDopplerAngle):
            return None, None
        azimuthAxis = np.linspace(
            *processing.doa.azimuthFov, processing.doa.azimuthBins
        )
        vectors = steering_vector(
            self.radar.virtualArray,
            self.radar.wavelength,
            azimuthAxis,
            0.0,
        )
        rd = rdCube.data
        rangeAngle = np.empty((rd.shape[0], azimuthAxis.size), dtype=float)
        rdaData = (
            np.empty((rd.shape[0], rd.shape[1], azimuthAxis.size), dtype=np.float32)
            if processing.retainRangeDopplerAngle
            else None
        )
        # A visual RA map uses the strongest Doppler cell at each range. This
        # keeps the default product practical for 128-loop 4D captures while
        # preserving a full RDA beam cube when it is explicitly requested.
        rdPower = np.mean(np.abs(rd) ** 2, axis=2)
        strongest = np.argmax(rdPower, axis=1)
        snapshots = rd[np.arange(rd.shape[0]), strongest]
        rangeAngle[:] = np.abs(snapshots @ vectors.conj().T) ** 2
        if rdaData is not None:
            for start in range(0, rd.shape[0], 32):
                stop = min(start + 32, rd.shape[0])
                beam = np.einsum(
                    "rdv,av->rda", rd[start:stop], vectors.conj(), optimize=True
                )
                rdaData[start:stop] = np.abs(beam) ** 2
        raCube = (
            RadarCube(
                data=rangeAngle,
                dims=("range", "azimuth"),
                stage="range_angle",
                radar=self.radar,
                coords={"range": self.radar.rangeAxis, "azimuth": azimuthAxis},
            )
            if processing.retainRangeAngle
            else None
        )
        rdaCube = (
            RadarCube(
                data=rdaData,
                dims=("range", "doppler", "azimuth"),
                stage="range_doppler_angle",
                radar=self.radar,
                coords={
                    "range": self.radar.rangeAxis,
                    "doppler": self.radar.velocityAxis,
                    "azimuth": azimuthAxis,
                },
            )
            if rdaData is not None
            else None
        )
        return raCube, rdaCube


__all__ = ["ADCToPointCloudPipeline"]
