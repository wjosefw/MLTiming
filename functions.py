import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def gauss(x, H, A, x0, sigma):
    """Gaussian on a constant background."""
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


@dataclass(frozen = True)
class GaussianFit:
    """A gauss() fit to a histogram: its parameters (sigma in the units of the data), their standard
    errors (perr, in the order H, A, x0, sigma), the histogram fitted (bin centres and counts) and
    whether the fit worked (success; message says why not)."""
    H: float
    A: float
    x0: float
    sigma: float
    perr: np.ndarray
    centers: np.ndarray
    counts: np.ndarray
    success: bool
    message: str = 'OK'

    @property
    def fwhm(self):
        return 2.35482 * abs(self.sigma)


def fit_histogram(centers, counts):
    """gauss() least-squares fitted to histogram counts at bin centres, started from their mean and
    standard deviation. Returns a GaussianFit; success is False (and the parameters NaN) if the
    histogram is empty or the fit does not converge."""
    centers, counts = np.asarray(centers, dtype = float), np.asarray(counts, dtype = float)
    nan4 = np.full(4, np.nan)
    if counts.sum() <= 0:
        return GaussianFit(*nan4, nan4, centers, counts, False, 'Empty histogram.')
    mean = np.sum(centers * counts) / np.sum(counts)
    sigma = np.sqrt(np.sum(counts * (centers - mean) ** 2) / np.sum(counts))
    try:
        popt, pcov = curve_fit(gauss, centers, counts, p0 = [counts.min(), counts.max(), mean, sigma])
    except (RuntimeError, ValueError) as e:
        return GaussianFit(*nan4, nan4, centers, counts, False, str(e))
    with np.errstate(invalid = 'ignore'):
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else nan4
    ok = bool(np.all(np.isfinite(popt)))
    return GaussianFit(*popt, perr, centers, counts, ok, 'OK' if ok else 'Non-finite parameters.')


def fit_gaussian(array, shift = 0.0, nbins = 51, range = 0.5, limit = 1):
    """Gaussian fit to the peak of the distribution of array - shift: the peak is located on a
    histogram over [-limit, limit], then a histogram of nbins over the peak +- range is fitted with
    fit_histogram. Returns a GaussianFit."""
    values = np.ravel(np.asarray(array, dtype = float) - shift)
    counts, edges = np.histogram(values, bins = nbins, range = [-limit, limit])
    centers = 0.5 * (edges[1:] + edges[:-1])
    peak = centers[np.argmax(counts)]
    counts, edges = np.histogram(values, bins = nbins, range = (peak - range, peak + range))
    return fit_histogram(0.5 * (edges[1:] + edges[:-1]), counts)


def get_gaussian_params(array, shift, nbins = 51, range = 0.5, limit = 1):
    """fit_gaussian as ((H, x0, FWHM), perr). Raises RuntimeError if the fit fails."""
    fit = fit_gaussian(array, shift, nbins = nbins, range = range, limit = limit)
    if not fit.success:
        raise RuntimeError(f'Gaussian fit failed: {fit.message}')
    return (fit.H, fit.x0, fit.fwhm), fit.perr


def calculate_gaussian_center(vector, nbins = 51, limit = 3):
    """Centroid (x0) of a gauss() fit to the histogram over [-limit, limit] of each row of vector
    (2-D). NaN, with a warning, for a row whose fit fails."""
    centroid = []
    for row in np.atleast_2d(vector):
        counts, edges = np.histogram(row, bins = nbins, range = [-limit, limit])
        fit = fit_histogram(0.5 * (edges[1:] + edges[:-1]), counts)
        if not fit.success:
            warnings.warn(f'Gaussian centroid fit failed: {fit.message}')
        centroid.append(fit.x0 if fit.success else np.nan)
    return np.array(centroid, dtype = 'float64')


def plot_gaussian(array, shift, nbins = 51, label = ' ', range = 0.8, limit = 1):
    """Histogram of array - shift around its peak (as fit_gaussian) with the Gaussian fit drawn over it."""
    fit = fit_gaussian(array, shift, nbins = nbins, range = range, limit = limit)
    width = fit.centers[1] - fit.centers[0]
    lo, hi = fit.centers[0] - width / 2, fit.centers[-1] + width / 2
    values = np.ravel(np.asarray(array, dtype = float) - shift)
    _, _, patches = plt.hist(values, bins = nbins, range = (lo, hi), alpha = 0.5, label = label)
    if fit.success:
        x = np.linspace(lo, hi, 500)
        plt.plot(x, gauss(x, fit.H, fit.A, fit.x0, fit.sigma), color = patches[0].get_facecolor())
    else:
        warnings.warn(f'Gaussian fit failed: {fit.message}')
    plt.legend()
    return fit

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def momentos(vector, order = 4):
    """
    Moments of each pulse: sum over time of vector * t**k, k = 0 .. order - 1, with t the sample
    times rescaled to [0, 1].

    Parameters:
    vector (array-like): The input data array with shape (Nev, Nt, Nc).

    Returns:
    numpy.ndarray: The moments, shape (Nev, order, Nc).
    """
    vector = np.asarray(vector, dtype = np.float64)
    t = np.linspace(0, 1, vector.shape[1])
    return np.einsum('ntc,tk->nkc', vector, t[:, None] ** np.arange(order))

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def normalize_by_max(pulse_array, fit_samples=5):
    """
    Normalize pulse data by the maximum of each pulse: the vertex of a parabola least-squares
    fitted to the fit_samples samples centred on the highest one (the highest sample itself where
    the fit has no maximum within those samples).

    Parameters:
    pulse_array (numpy.ndarray): 3D array of pulse data with shape (n_pulses, n_samples, n_channels).
    fit_samples (int): Odd number of samples of the parabola fit (1: the highest sample). Defaults
                       to 5: from the highest sample alone up to 15 samples timing is the same, while
                       the 60 samples once used here cost ~1 ps CTR (CNN and a fixed-threshold LED).

    Returns:
    numpy.ndarray: Normalized pulse data with the same shape as input.
    """
    if fit_samples < 1 or fit_samples % 2 == 0:
        raise ValueError(f"`fit_samples` must be odd and at least 1 (got {fit_samples}).")
    x = np.asarray(pulse_array, dtype = np.float64)
    peak = x.max(axis = 1)                                     # (n_pulses, n_channels)
    if fit_samples >= 3:
        half = fit_samples // 2
        k = np.arange(-half, half + 1)
        idx = np.clip(x.argmax(axis = 1)[:, None, :] + k[None, :, None], 0, x.shape[1] - 1)
        v = np.take_along_axis(x, idx, axis = 1)                # (n_pulses, fit_samples, n_channels)
        design = np.stack([k ** 2, k, np.ones_like(k)], axis = 1).astype(np.float64)
        a, b, c = np.einsum('pk,nkc->pnc', np.linalg.pinv(design), v)
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            vertex = c - b ** 2 / (4 * a)
            inside = np.abs(b / (2 * a)) <= half            # the vertex lies within the fitted samples
        peak = np.where((a < 0) & inside & np.isfinite(vertex), vertex, peak)
    return (x / peak[:, None, :]).astype(np.asarray(pulse_array).dtype)

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def simpsons_rule_array(y, h):
    """"Calculate integral using Simpsons' rule"""
    n = y.shape[1]

    # Handle the even and odd index multiplications
    even_indices = np.arange(2, n - 1, 2)  # Indices for 2y_i terms (even indices)
    odd_indices = np.arange(1, n, 2)       # Indices for 4y_i terms (odd indices)

    # Apply Simpson's rule across all rows at once
    # Add y[0] + y[-1]
    integral = y[:, 0] + y[:, -1]

    # Add 4 * y[odd_indices]
    integral += 4 * np.sum(y[:, odd_indices], axis=1)

    # Add 2 * y[even_indices]
    integral += 2 * np.sum(y[:, even_indices], axis=1)

    # Multiply by h/3 and return the result
    return integral * h / 3

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def normalize(data, method = 'standardization'):

    if method not in ['min-max', 'max', 'standardization']:
        raise ValueError("Invalid method. Choose from 'min-max', 'max', 'standardization'.")

    if method == 'min-max':
        if len(data.shape) == 3:
            min_vals = np.min(data[:, :, 0], axis=0)
            max_vals = np.max(data[:, :, 0], axis=0)
            normalized_data_dec0 = (data[:, :, 0] - min_vals) / (max_vals - min_vals)
            normalized_data_dec1 = (data[:, :, 1] - min_vals) / (max_vals - min_vals)
            normalized_data = np.stack((normalized_data_dec0, normalized_data_dec1), axis=-1) # Concatenate the normalized channels back togethe
            params = (min_vals, max_vals)   
        
        if len(data.shape) == 2:
            max_vals = np.max(data, axis = 0)
            min_vals = np.min(data, axis = 0)
            normalized_data = (data - min_vals) / (max_vals - min_vals)
            params = (min_vals, max_vals)   

    elif method == 'max':
        if len(data.shape) == 3:
            max_vals = np.max(data[:, :, 0], axis=0)
            params = max_vals
            normalized_data_dec0 = data[:, :, 0] / max_vals
            normalized_data_dec1 = data[:, :, 1] / max_vals
            normalized_data = np.stack((normalized_data_dec0, normalized_data_dec1), axis=-1)
        
        if len(data.shape) == 2:
            max_vals = np.max(data, axis=0)
            params = max_vals
            normalized_data = data / max_vals
    
    elif method == 'standardization':
        if len(data.shape) == 3:
            means = np.mean(data[:, :, 0], axis=0)
            stds = np.std(data[:, :, 0], axis=0)
            normalized_data_dec0 = (data[:, :, 0] - means) / stds
            normalized_data_dec1 = (data[:, :, 1] - means) / stds
            normalized_data = np.stack((normalized_data_dec0, normalized_data_dec1), axis=-1)
            params = (means, stds)
        
        if len(data.shape) == 2:
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
            normalized_data = (data - means) / stds
            params = (means, stds)

    
    return normalized_data, params

def normalize_given_params(data, params, channel=0, method='standardization'):
    """
    Normalize the given data using the specified method and parameters.

    Parameters:
    data (array-like): The input data array with shape (N, M, Nc), where N is the number of events,
                       M is the number of time points, and Nc is the number of channels.
    params (tuple): A tuple containing the parameters needed for normalization. For 'min-max' method,
                    it should be (min_values, max_values). For 'standardization' method, it should be 
                    (mean_values, std_devs). Both min_values and max_values, or mean_values and std_devs should have 
                    lengths equal to M (second dimension of data).
    channel (int): The channel to normalize. Defaults to 0.
    method (str): The normalization method to use. Choose from 'min-max' or 'standardization'. Defaults to 'standardization'.

    Returns:
    array-like: The normalized data array with the same shape as the input data.
    
    Raises:
    ValueError: If the method is not one of 'min-max' or 'standardization'.
    ValueError: If params is not a tuple with two elements.
    ValueError: If the lengths of params[0] and params[1] do not match the second dimension (M) of data.
    """
    
    
    if method not in ['min-max', 'standardization']:
        raise ValueError("Invalid method. Choose from 'min-max' or 'standardization'.")
    
    # Check if params is a tuple and has two elements
    if not isinstance(params, tuple) or len(params) != 2:
        raise ValueError("Params must be a tuple with two elements.")
    
    if len(params[0]) != data.shape[1] or len(params[1]) != data.shape[1]:
        raise ValueError("Length of params[0] and params[1] must match the second dimension (axis = 1) of data.")
    
    # Create a copy of the original data to avoid modifying it
    data_copy = np.copy(data)
    
    if method == 'min-max':
        normalized_data = (data_copy[:, :, channel] - params[0]) / (params[1] - params[0])
    elif method == 'standardization':
        normalized_data = (data_copy[:, :, channel] - params[0]) / params[1]

    return normalized_data

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def create_virtual_pairs(pulses, starts, window, time_step, delay_time = 1):
    """Virtual coincidence pairs for pairwise training, cut from the full waveforms.

    Each pair holds two copies of the same pulse, each delayed by its own random whole number of samples (uniform in
    0..delay_time ns, rounded to the sampling grid); the label is the difference of the two delays, d0 - d1 (ns).
    The copies are cut from the full waveform at the crop start, so the samples that move into the crop are the
    real ones that precede it. Pulses arrive at random sub-sample phases already, so whole-sample delays need no
    interpolation. Samples outside the record are 0.

    pulses: full waveforms (N, M); starts: crop start of each pulse (N,), i.e. minus the delays returned by
    extract_signal_window_by_fraction; window: crop length (before + after).
    Returns pairs (N, window, 2) float32 and labels (N,) float32."""

    pulses = np.asarray(pulses, dtype = np.float32)
    n, m = pulses.shape
    shifts = np.rint(np.random.uniform(0, delay_time, size = (n, 2)) / time_step).astype(np.int64)
    pos = np.asarray(starts, dtype = np.int64)[:, None, None] + np.arange(window)[None, :, None] - shifts[:, None, :]
    inside = (pos >= 0) & (pos < m)
    pairs = np.where(inside, pulses[np.arange(n)[:, None, None], np.clip(pos, 0, m - 1)], 0).astype(np.float32)
    labels = ((shifts[:, 0] - shifts[:, 1]) * time_step).astype(np.float32)
    return pairs, labels

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def edge_level_seen(pulses, time_step, after, delay_time, fraction = 0.1, n_max = 20000, seed = 0):
    """How far up the rising edge the network sees: the median pulse amplitude (fraction of the maximum) reached at
    the mean time left visible after the threshold crossing, over virtual-pair members (whole-sample delays).

    The window and delay worked best where this is about 0.15-0.18 (a window ending ~1.0-1.2 ns after the 10 %
    crossing, delays up to ~1 ns, for LaBr3 + PMT pulses at 0.133, 0.2 and 0.4 ns sampling). Use it to choose
    after / delay_time for pulses of another shape. pulses: max-normalised full waveforms (N, M)."""

    x = np.asarray(pulses[:n_max], dtype = np.float64)
    rows = np.arange(len(x))
    i = np.maximum((x > fraction).argmax(1), 1)
    frac = (fraction - x[rows, i - 1]) / (x[rows, i] - x[rows, i - 1])
    t = np.arange(0, 3, 0.01)                                     # ns after the crossing
    pos = np.minimum((i - 1 + frac)[:, None] + t[None] / time_step, x.shape[1] - 1.001)
    k = np.floor(pos).astype(int)
    edge = np.median(x[rows[:, None], k] * (1 - (pos - k)) + x[rows[:, None], k + 1] * (pos - k), 0)
    rng = np.random.default_rng(seed)
    shifts = np.rint(rng.uniform(0, delay_time, (20, len(frac))) / time_step)
    visible = time_step * np.mean(np.clip(after - 1 - frac - shifts, 0, None))
    return float(np.interp(visible, t, edge))

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def threshold_crossing(pulses, threshold, anchor = 0.5):
    """
    Where each pulse crosses `threshold` on its rising edge, in samples (fractional, linearly
    interpolated), by the anchor rule (as in MLPILEUP, Positronium_singles and PyGeneticGFN): from
    the first sample above max(threshold, anchor * maximum), back to the last sample at or below
    the threshold. Baseline noise that crosses the threshold but never reaches the anchor level
    cannot fire it. NaN for a pulse with no such crossing.

    pulses: (N, M) or one pulse (M,). Returns (N,) or a float.
    """
    if not 0 < anchor < 1:
        raise ValueError(f"`anchor` must be a fraction of the maximum, 0 < anchor < 1 (got {anchor}).")
    x = np.atleast_2d(np.asarray(pulses))
    n, m = x.shape
    rows, samples = np.arange(n), np.arange(m)
    level = np.maximum(threshold, anchor * x.max(axis = 1))
    above = x > level[:, None]
    anchor_idx = above.argmax(axis = 1)
    below = (x <= threshold) & (samples[None, :] < anchor_idx[:, None])
    last_below = np.where(below, samples[None, :], -1).max(axis = 1)
    valid = above[rows, anchor_idx] & (last_below >= 0)
    k = np.clip(last_below, 0, m - 2)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        cross = k + (threshold - x[rows, k]) / (x[rows, k + 1] - x[rows, k])
    cross = np.where(valid, cross, np.nan)
    return cross if np.ndim(pulses) == 2 else float(cross[0])

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def calculate_slope_y_intercept(vector, time_step, threshold = 0.1, anchor = 0.5):
    """
    Leading-edge time (ns from the first sample): where the pulse crosses `threshold` on its rising
    edge (threshold_crossing, anchor rule), linearly interpolated. NaN where there is no crossing.

    vector: one pulse (M,) or pulses (N, M). Returns a float or (N,).
    """
    return time_step * threshold_crossing(vector, threshold, anchor = anchor)

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def Calculate_CFD(array, fraction, shift, time_step):
    """
    Constant fraction discrimination of each pulse: the bipolar signal c(t) = y(t - shift) -
    fraction * y(t), the delayed copy linearly interpolated (shift in ns, any value), timed at the
    first upward zero crossing of c after its minimum (linearly interpolated). 0 for a pulse with no
    such crossing; the last sample time if the minimum is the last sample.

    Parameters:
    array: pulses (N, M) or one pulse (M,).
    fraction (float): the fraction of the undelayed pulse subtracted.
    shift (float): the delay in ns.
    time_step (float): the sampling step in ns.

    Returns:
    Timestamps in ns from the first sample, (N,) or a float.
    """
    x = np.atleast_2d(np.asarray(array, dtype = np.float64))
    n, m = x.shape
    rows = np.arange(n)

    # Delayed copy: the fractional part of the delay by linear interpolation, then whole samples
    res = shift % time_step
    idel = int(shift / time_step)
    interp = x.copy()
    interp[:, 1:] = x[:, 1:] - (x[:, 1:] - x[:, :-1]) / time_step * res
    delayed = np.roll(interp, idel, axis = 1)
    delayed[:, :idel] = interp[:, :idel]
    c = delayed - fraction * x

    # First upward zero crossing after the minimum
    t = np.arange(m) * time_step
    imin = c.argmin(axis = 1)
    i = np.arange(m - 1)
    candidate = (i[None, :] >= imin[:, None]) & (((c[:, :-1] < 0) & (c[:, 1:] > 0)) | (c[:, :-1] == 0))
    found = candidate.any(axis = 1)
    k = candidate.argmax(axis = 1)
    c0, c1, t0, t1 = c[rows, k], c[rows, np.minimum(k + 1, m - 1)], t[k], t[np.minimum(k + 1, m - 1)]
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        slope = (c1 - c0) / (t1 - t0)
        crossing = -(c0 - slope * t0) / slope
    times = np.where(c0 == 0, t0, crossing)
    times = np.where(found, times, 0.0)
    times = np.where(imin == m - 1, t[-1], times)
    return times if np.ndim(array) == 2 else float(times[0])

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def extract_signal_window_by_fraction(vector, time_step, fraction = 0.1, window_low = 140, window_high = 10, anchor = 0.5):
    """
    Extracts a windowed segment of a signal around a threshold crossing point defined by a fraction
    of the maximum signal amplitude. Also computes delay indices relative to the crossing point.

    Vectorised across pulses, with zero padding whenever the requested window extends
    beyond the available samples.

    The crossing is found by the anchor rule (threshold_crossing).
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

    # Crossing of each pulse by the anchor rule (threshold_crossing)
    cross_samples = threshold_crossing(vector, fraction, anchor = anchor)
    if np.isnan(cross_samples).any():
        missing = np.flatnonzero(np.isnan(cross_samples))[0]
        raise IndexError(f"Signal index {missing} has no rising-edge crossing of the threshold {fraction}.")

    row_idx = np.arange(num_signals)
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

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def create_dataloaders(input, target, batch_size = 32, shuffle = True):
  dataset = torch.utils.data.TensorDataset(torch.from_numpy(input).float(), torch.from_numpy(np.expand_dims(target, axis = -1)).float())
  dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
  return dataloader

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

def select_channel(data, channel = None):
    """
    Accepts either a single-detector array of shape (N, M) or a coincidence array of
    shape (N, M, 2) and returns a single-detector (N, M) array.

    For (N, M, 2) input, `channel` (0 or 1) selects which detector to train/test on,
    so users with paired coincidence measurements don't need to pre-split their data
    into separate files.
    """

    if data.ndim == 2:
        return data

    if data.ndim == 3:
        if channel not in (0, 1):
            raise ValueError("`channel` must be 0 or 1 when the input data has shape (N, M, 2).")
        return data[:, :, channel]

    raise ValueError(f"Expected data of shape (N, M) or (N, M, 2), got shape {data.shape}.")
