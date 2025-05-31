"""Pipeline of Generating ADC Cube."""
import numpy as np
from typing import List
from scripts.waveform import WaveForm
from scripts.path import Path
from scripts.const import speedOfLight
from scripts.utils import range_resolution, velocity_resolution

def ray2sig(path: Path, chirp: WaveForm) -> np.ndarray:
    """
    Simulates an Intermediate Frequency (IF) signal based on path characteristics and signal configuration.

    The simulation models how the signal propagates over a specific path and how the Doppler shift,
    path length, and signal polarization influence the received signal. The signal is represented 
    as a complex-valued IF signal in the frequency domain.

    Args:
        path (Path): A Path object containing path-specific information such as:
                      - `length`: the length of the path,
                      - `dopplerSpeed`: the Doppler shift due to relative motion between transmitter and receiver.
        chirp (WaveForm): A WaveForm object that represents the chirp signal. It contains:
                          - `waveLength`: the wavelength of the signal,
                          - `chirpSlope`: the slope of the chirp,
                          - `Polarization`: the polarization type used in the signal,
                          - `t`: the time vector in one frame,
                          - `k`: the index vector of sample within one chirp.

    Returns:
        np.ndarray: The simulated Intermediate Frequency (IF) signal as a complex-valued NumPy array.
                    The signal accounts for the Doppler shift, path length, and polarization effects.

    """

    # Calculate the range (r_t) based on the path length and Doppler speed
    r_t = path.length / 2 + path.dopplerSpeed * chirp.t

    # Initial phase shift due to the round-trip travel time of the signal over the path length
    # Phi_0 accounts for the initial phase shift as the signal travels over the path length.
    Phi_0 = 2 * r_t / chirp.waveLength

    # Calculate the range frequency (fR), which depends on the chirp slope and the path distance
    # fR models the frequency shift due to the round-trip travel time of the signal.
    fR = 2 * chirp.chirpSlope * r_t / speedOfLight

    # Calculate the Doppler frequency shift (fD), which depends on the relative motion (Doppler speed) of the path
    # fD models the frequency shift due to relative motion between the transmitter and receiver.
    fD = 2 * path.dopplerSpeed / chirp.waveLength

    # Signal polarization handling: If the polarization is 'HX', we use the absolute value of the path's eHY component
    # Polarization affects the amplitude of the signal, and this models how the signal interacts with the medium.
    if chirp.Polarization == 'HX':
        polarAmplitude = np.abs(path.eHY)  # Polarization-dependent amplitude adjustment
    else:
        # For other polarizations, a default amplitude (1) is used.
        # TODO: Extend this to handle different polarization types if needed in the future.
        polarAmplitude = 1  

    # Phase shift caused by antenna spacing(currently not applied in the calculation but could be added later).
    # phaseShift = np.exp(1j * signal.phaseShift)

    # Calculate the Intermediate Frequency (IF) signal using the path and chirp characteristics
    # The formula incorporates Doppler shifts, range-based frequency shifts, and polarization effects.
    S_IF = polarAmplitude * np.exp(1j * 2 * np.pi * ((fR + fD) * (chirp.k) + Phi_0))

    return S_IF

def cycle2binary(cycle_data: dict, chirp: WaveForm) -> np.ndarray:
    """
    Converts cycle data into a binary format suitable for further processing.
    
    Args:
        cycle_data (dict): Dictionary containing cycle data, including the number of receivers and antenna data.
        chirp (WaveForm): An instance of the WaveForm class containing chirp parameters such as number of chirps per frame, 
                          number of samples per chirp, bandwidth, start frequency, chirp duration, and polarization.
    
    Returns:
        np.ndarray: A 3D NumPy array representing the ADC samples for each antenna, chirp, and sample point.
    """
    
    # Extract the number of receivers from cycle data
    numOfRx = cycle_data['numOfRx']

    # Initialize a 3D NumPy array to store the ADC samples for each antenna, chirp, and sample point
    # The shape of the array is (numOfRx, numChirpsperFrame, numSamplesPerChirps)
    bin_map: np.ndarray = np.zeros((numOfRx, chirp.numChirpsperFrame, chirp.numSamplesPerChirps))

    # Calculate range and velocity resolution based on chirp parameters
    res_range = range_resolution(chirp.Bandwidth)
    res_velocity = velocity_resolution(chirp.startFrequency, chirp.numChirpsperFrame, chirp.chirpDuration)

    # Iterate through each antenna's data in the cycle
    for IdxAntenna, AntennaData in enumerate(cycle_data['AntennaData']):
        # Extract the list of paths associated with the current antenna
        paths: List[dict] = AntennaData['Paths']

        for path_dict in paths:
            # Convert path dictionary to Path object
            path = Path.from_dict(path_dict)

            # Calculate range index based on path length
            r_t = path.length / 2
            IdxRange = int(r_t / res_range)
            
            # Calculate velocity index based on doppler speed
            IdxVelocity = int(path.dopplerSpeed / res_velocity) + int(chirp.numChirpsperFrame / 2)
            
            # Determine polarization-dependent amplitude adjustment
            if chirp.Polarization == 'HX':
                polarAmplitude = np.abs(path.eHY)  # Polarization-dependent amplitude adjustment
            else:
                # For other polarizations, a default amplitude (1) is used.
                # TODO: Extend this to handle different polarization types if needed in the future.
                polarAmplitude = 1  

            # Accumulate the amplitude in the bin map
            bin_map[IdxAntenna, IdxVelocity, IdxRange] += polarAmplitude
            
    # Return the accumulated ADC signal cube
    return bin_map
            
def cycle2adc(cycle_data: dict, chirp: WaveForm) -> np.ndarray:
    """
    Converts the cycle data into an ADC cube by processing the signal paths and accumulating 
    the sampled Intermediate Frequency (IF) signals for each antenna.

    Args:
        cycle_data (dict): A dictionary containing cycle-specific data, including 
                            the number of receivers (`numOfRx`), antenna data (`AntennaData`), 
                            and path details for each antenna.
        chirp (WaveForm): A WaveForm object containing information about the chirp parameters, 
                          including the number of chirps per frame (`numChirpsperFrame`) 
                          and the number of samples per chirp (`numSamplesPerChirps`).

    Returns:
        np.ndarray: A 3D NumPy array (ADC_CUBE) with shape `(numOfRx, numChirpsperFrame, numSamplesPerChirps)` 
                    that represents the accumulated sampled IF signal for each antenna.
    """
    
    # Extract the number of receivers from cycle data
    numOfRx = cycle_data['numOfRx']

    # Initialize a 3D NumPy array to store the ADC samples for each antenna, chirp, and sample point
    # The shape of the array is (numOfRx, numChirpsperFrame, numSamplesPerChirps)
    ADC_CUBE: np.ndarray = np.zeros((numOfRx, chirp.numChirpsperFrame, chirp.numSamplesPerChirps), dtype=np.complex128)

    # Iterate through each antenna's data in the cycle
    for IdxAntenna, AntennaData in enumerate(cycle_data['AntennaData']):
        # Extract the list of paths associated with the current antenna
        paths: List[dict] = AntennaData['Paths']

        # Iterate through each path in the current antenna's paths
        for path_dict in paths:
            # Convert the path data (in dictionary format) into a Path object
            # The Path object will encapsulate information about the path, such as hops, Doppler speed, etc.
            path = Path.from_dict(path_dict)

            # Accumulate the sampled Intermediate Frequency (IF) signal for the current path
            # The `ray2sig` function processes the path and chirp data to return a signal
            # The result is added to the corresponding slice of the ADC_CUBE array
            ADC_CUBE[IdxAntenna] += ray2sig(path, chirp)

    # Return the accumulated ADC signal cube
    return ADC_CUBE

