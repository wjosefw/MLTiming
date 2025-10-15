import numpy as np


def extract_signal_window_by_fraction(vector, time_step, fraction = 0.1,
                                      window_low = 140, window_high = 10):
    """
    Extract a fixed-size window around the threshold crossing obtained via fractional amplitude.

    The implementation is vectorised across pulses to reduce Python-layer overhead while
    preserving the legacy behaviour, including zero padding whenever the requested window
    extends beyond the available samples.
    """

    vector = np.asarray(vector)
    if vector.ndim != 2:
        raise ValueError("`vector` must be a 2-D array with shape (num_signals, num_samples).")

    num_signals, num_samples = vector.shape
    window_low = int(window_low)
    window_high = int(window_high)
    window_size = window_low + window_high

    new_vector = np.zeros((num_signals, window_size), dtype = np.float32)
    if num_signals == 0:
        return new_vector, np.empty((0,), dtype = np.int64)

    threshold = fraction

    # Locate the first index where each signal crosses the threshold.
    above_threshold = vector > threshold
    first_cross = above_threshold.argmax(axis = 1)

    # Validate that every signal actually crosses the threshold.
    valid_cross = above_threshold[np.arange(num_signals), first_cross]
    if not np.all(valid_cross):
        missing = np.where(~valid_cross)[0][0]
        raise IndexError(f"Signal index {missing} never crosses the threshold {threshold}.")

    row_idx = np.arange(num_signals)
    prev_idx = first_cross - 1

    prev_vals = vector[row_idx, prev_idx]
    next_vals = vector[row_idx, first_cross]

    # Fractional offset for the precise crossing between the two surrounding samples.
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        frac_offset = (threshold - prev_vals) / (next_vals - prev_vals)

    cross_samples = (first_cross - 1) + frac_offset
    central_index = np.floor(cross_samples).astype(np.int64)
    start = np.maximum(central_index - window_low, 0)

    # Determine how many samples are available for each signal when the window spills over the edges.
    lengths = np.minimum(window_size, np.maximum(num_samples - start, 0))

    # Gather the windowed samples via advanced indexing; clamp indices to stay in-bounds.
    offsets = np.arange(window_size)
    indices = start[:, None] + offsets
    if num_samples == 0:
        gathered = np.zeros((num_signals, window_size), dtype = vector.dtype)
    else:
        clipped = np.clip(indices, 0, num_samples - 1)
        gathered = vector[row_idx[:, None], clipped]

    mask = offsets < lengths[:, None]
    if gathered.dtype != np.float32:
        gathered = gathered.astype(np.float32, copy = False)
    new_vector[mask] = gathered[mask]

    delays = (-start).astype(np.int64, copy = False)
    return new_vector, delays

def calculate_slope_y_intercept(vector, time_step, threshold=0.1):
    """
    Calculate the time at which a vector exceeds a given threshold by linear interpolation.

    Parameters:
    ----------
    - Vector (numpy.ndarray): A 1D array representing the signal or data to analyze.
    - Time_step (float): The time interval between consecutive samples in the vector.  
    - Threshold (float): The threshold value to identify the crossing point.

    Returns:
    -------
    time (float): The estimated time at which the signal first exceeds the threshold value. 
    """
    
    t = np.arange(vector.shape[0]) * time_step
    index = np.where(vector > threshold)[0][0]
    t1 = t[index]
    t0 = t[index - 1]
    m = (vector[index] - vector[index - 1]) / (t1 - t0)
    b = vector[index - 1] - m*t0
    time = (threshold - b) / m
    return time
