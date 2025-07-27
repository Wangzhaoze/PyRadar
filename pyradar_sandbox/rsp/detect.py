import numpy as np



def os_cfar(samples: np.array, ws: int, ngc: int = 2, tos: int = 8) -> np.array:
    """Ordered Statistic Constant False Alarm Rate detector.

    Arguments:
        samples: Non-Coherently integrated samples
        ws: Window Size
        ngc: Number of guard cells
        tos: Scaling factor

    Return:
        mask
    """
    ns: int = len(samples)
    k: int = int(3.0 * ws/4.0)

    # Add leading and trailing zeros into order to run the algorithm over
    # the entire samples
    samples = np.append(np.zeros(ws), samples)
    samples = np.append(samples, np.zeros(ws))

    mask = np.zeros(ns)
    for idx in range(ns):
        # tcells: training cells
        pre_tcells = samples[ws + idx - ngc - (ws // 2) : ws + idx - ngc]
        post_tcells = samples[ws + idx + ngc + 1: ws + idx + ngc + (ws // 2) + 1]
        tcells = np.array([])
        tcells = np.append(tcells, pre_tcells)
        tcells = np.append(tcells, post_tcells)
        tcells = np.sort(tcells)
        if samples[ws + idx] > tcells[k] * tos:
            mask[idx] = 1
    return mask


class ObjectDetected:
    """Object detected.

    Definition of object detected by applying CFAR

    NOTE: It's possible to have multiple detections on the same object
    depending on the resolution of the radar sensor
    """

    vidx: int = -1      # Velocity bin index
    ridx: int = -1      # Range bin index
    aidx: int = -1      # Azimuth bin index
    eidx: int = -1      # Elevation bin
    snr: float = 0      # Signal over Noise ratio

    def __str__(self) -> str:
        return f"Obj(SNR:{self.snr:.2f})"

    def __repr__(self) -> str:
        return self.__str__()


def nq_cfar_2d(samples, ws: int, ngc: int,
             quantile: float = 0.75, tos: int = 8) -> np.array:
    """N'th quantile statistic Constant False Alarm Rate detector.

    The principle is exactly the same as the Ordered Statistic
    Constant False Alarm Rate detector. This routine just applies
    it on a 2D signal.

    Arguments:
        samples: 2D signal to filter
        ws (int): Window size
        ngc (int): Number of guard cells
        quantile (float): Order of the quantile to compute for the noise
            power estimation
        tos (int): Scaling factor for detection an object
    """
    nx, ny = samples.shape
    mask = np.zeros((nx, ny))
    detections: list[ObjectDetected] = []

    for xidx in range(nx):
        # Before CUT (Cell Under Test) start index on the x-axis
        xbs: int = xidx - ws
        xbs = xbs if (xbs > 0) else 0

        # Before CUT (Cell Under Test) end index on the x-axis
        xbe: int = xidx - ngc
        xbe = xbe if (xbe > 0) else 0

        # After CUT (Cell Under Test) start index on the x-axis
        xas: int = xidx + ngc + 1
        # After CUT (Cell Under Test) end index on the x-axis
        xae: int =  xidx + ws + 1
        xae = xae if (xae < nx) else nx

        for yidx in range(ny):
            # Before CUT (Cell Under Test) start index on the y-axis
            ybs: int = yidx - ws
            ybs = ybs if (ybs > 0) else 0

            # Before CUT (Cell Under Test) end index on the y-axis
            ybe: int = yidx - ngc

            # After CUT (Cell Under Test) start index on the y-axis
            yas: int = yidx + ngc + 1

            # After CUT (Cell Under Test) end index on the y-axis
            yae: int =  yidx + ws + 1
            yae = yae if (yae < ny) else ny

            tcells = np.array([])
            if xbe > 0:
                tcells = samples[xbs:xbe, ybs:yae].reshape(-1)

            if xas < nx - 1:
                tcells = np.append(
                    tcells,
                    samples[xas:xae, ybs:yae].reshape(-1)
                )

            if ybe > 0:
                tcells = np.append(
                    tcells,
                    samples[xbe:xas, ybs:ybe,].reshape(-1)
                )

            if yas < nx - 1:
                tcells = np.append(
                    tcells,
                    samples[xbe:xas, yas:yae,].reshape(-1)
                )
            m = np.quantile(tcells, quantile, method="weibull")
            if samples[xidx, yidx] > (m * tos):
                mask[xidx, yidx] = 1
                obj = ObjectDetected()
                obj.vidx = xidx
                obj.ridx = yidx
                obj.snr = samples[xidx, yidx] / m
                detections.append(obj)
    return mask, detections
