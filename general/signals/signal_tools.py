import numpy as np
import scipy
from scipy import signal as s
from scipy.optimize import curve_fit


def filtfilt_nonan(b, a, data, axis=-1, fillval=None, **kwargs):
    nidx = np.isnan(data)
    tmp = data.copy()
    if fillval is None:
        tmp[nidx] = np.mean(tmp[~nidx], axis=axis)
    else:
        tmp[nidx] = fillval

    tmp = scipy.signal.filtfilt(b, a, tmp, axis=axis, **kwargs)
    tmp[nidx] = np.nan
    return tmp


def mean_filter_nd(data, filterdims, ignore_nan=False, output=None, **kwargs):
    if ignore_nan:
        raise NotImplemented()

    assert not np.any(np.isnan(data))
    assert not np.any(np.isinf(data))

    if output is None:
        res = scipy.ndimage.uniform_filter(data, size=filterdims, **kwargs)
    else:
        scipy.ndimage.uniform_filter(data, size=filterdims, output=output, **kwargs)
        res = output

    assert not np.any(np.isnan(res))
    assert not np.any(np.isinf(res))

    return res


def time_signal_sine(cmplx_ampl, t_vect, f, phase=0) -> np.ndarray:
    """
    Create a np.ndarray with amplitude values of a sine signal given its amplitude and frequency.

    :param cmplx_ampl: complex amplitude
    :param t_vect: sampling times for time signal
    :param f: frequency of sine signal
    :param phase: phase in radiant
    """
    return np.real(cmplx_ampl * np.exp(1j * (2 * np.pi * f * t_vect + phase)))


def db2pow(db):
    """
    converts db to a power ratio
    :param db:
    :return:

    >>> db2pow(10)
    10.0
    """
    return 10 ** (db / 10.0)


def additive_white_gaussian_noise(signal, snr_desired_db: float, snr_signal_db=np.inf, seed=None,
                                  axis=None) -> np.ndarray:
    """
    this function takes time domain data (signal) and generates noise that, when added to the signal,
    makes it reach snr_desired_db.

    :param signal:
    :param snr_desired_db:  target SNR  (dB)
    :param snr_signal_db: estimated SNR of the given data in signal (dB)
    :param seed random seed
    :param axis
    :return:

    >>> signal= np.sin(np.linspace(0, 100*np.pi, 1000))
    >>> noise=additive_white_gaussian_noise(signal, snr_desired_db=0, seed=42)
    >>> np.var(signal)
    0.49949999999999994
    >>> np.var(noise)
    0.4784735423708929
    """

    signal = np.array(signal, dtype=float)
    if snr_desired_db == snr_signal_db:
        return np.zeros_like(signal)
    if snr_desired_db > snr_signal_db:
        raise ValueError("Could not increase the SNR by magic!")

    signal_noise_power = np.var(signal, axis=axis)
    snr_desired = db2pow(snr_desired_db)
    snr_signal = db2pow(snr_signal_db)

    # %%%snr=signal/noise
    # signalnoise=signal+noise
    # signalnoise=noise*SNR+noise
    # signalnoise=noise (SNR+1)
    # noise=signalnoise / (SNR+1)
    #
    # snr_new=signal/(noise+noise_new)
    # noise_new=signal/snr_new-noise
    # noise_new=noise*SNR/snr_new-noise
    # noise_new=noise*(SNR/snr_new-1)

    noise = signal_noise_power / (snr_signal + 1)
    if np.isinf(snr_signal):  # var_signal_noise=signal
        newnoise = signal_noise_power / snr_desired
    else:
        newnoise = noise * (snr_signal / snr_desired - 1)

    r = np.random.RandomState(seed=seed)  # type:RandomState
    return newnoise ** 0.5 * r.standard_normal(signal.shape)


def gaussian(x, a, x0, sigma):
    """
    Args:
        x: arguments
        a: amplitude
        x0: mean of gaussian function
        sigma: std.deviation

    Returns: gaussian function values with type as x
    """
    return a * np.exp(-0.5*((x - x0) / sigma )**2)


def running_mean(x, N):
    """
    simple moving average
    Args:
        x: data
        N: size of moving average filter

    Returns:

    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def get_gaussian_fit_parameters(data):
    """
    gaussian fitting function with inital guess for the parameters
    Args:
        data: data to fit the gaussian to

    Returns: parameters of gaussian function

    """
    x = range(len(data))
    amplitude_init = np.max(data)
    mean_init = np.argmax(data)
    data_gradient = np.gradient(running_mean(data, 5))
    turning_point1 = s.argrelmax(data_gradient)
    turning_point2 = s.argrelmin(data_gradient)
    if len(turning_point1) == 1 and len(turning_point2) == 1 and len(turning_point1[0]) == 1 and len(
            turning_point2[0]) == 1:
        sigma_init = turning_point2[0] - turning_point1[0]
    else:
        sigma_init = 200
        print(
            "Couldn't detect initial sigma for gaussian fitting from data. Using standard value of " + str(sigma_init))

    try:
        popt, pcov = curve_fit(gaussian, x, data, p0=[amplitude_init, mean_init, sigma_init])
        amplitude = popt[0]
        mean = popt[1]
        sigma = popt[2]
    except:
        print("Couldn't fit data to gauss. Parameters are set to zero.")
        amplitude = 0
        mean = 0
        sigma = 0
    return amplitude, mean, sigma


def Notch_Filter(time, band, freq, ripple, order, filter_type, data, bandpass = True):
    # TODO: make a test
    # Required input defintions are as follows;
    # time:   Time between samples
    # band:   The bandwidth around the centerline freqency that you wish to filter
    # freq:   The centerline frequency to be filtered
    # ripple: The maximum passband ripple that is allowed in db
    # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
    #         IIR filters are best suited for high values of order.  This algorithm
    #         is hard coded to FIR filters
    # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
    # data:         the data to be filtered
    from scipy.signal import iirfilter
    from scipy.signal import lfilter
    fs   = 1/time
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    if bandpass:
        btype = 'bandpass'
    else:
        btype = 'bandstop'
    b, a = iirfilter(order, [low, high], rp=ripple, btype=btype,
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def rolling_window(a, window):
    # TODO: make a test
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def get_quant_signal(signal, no_qu_steps=3):
    """
    This function quantizes a signal in no_qu_steps that lies in the range -1..1 . Default is a quantization to values [-1,0,1].

    :param signal: unflipped original signal
    :param no_qu_steps: number of quantization steps e.g. -1, 0, 1 NOTE: always odd
    :return:

    >>> interactive = False
    >>> signal = np.linspace(-1, 1, 101)
    >>> no_qu_steps = 5
    >>> squant = get_quant_signal(signal, no_qu_steps=no_qu_steps)
    >>> if interactive:
    ...     import matplotlib.pyplot as plt
    ...     fig = plt.figure()
    ...     ax = fig.add_subplot(1, 1, 1)
    ...     p = ax.plot(signal)
    ...     p = ax.plot(squant)
    ...     plt.grid(True)
    ...     plt.show()
    >>> squant
    array([-1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. ,
           -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -0.5, -0.5,
           -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
           -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0. ,  0. ,  0. ,  0. ,
            0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
            0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.5,  0.5,  0.5,  0.5,  0.5,
            0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
            0.5,  0.5,  0.5,  0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,
            1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,
            1. ,  1. ])
    >>> tr = 0.2
    >>> np.all(squant[np.logical_and(signal > 0.2, signal <= 0.6)] == 0.5)
    True
    >>> np.all(squant == -squant[::-1])
    True
    """

    # minimum number of quant. steps and odd
    assert no_qu_steps >= 3  # minimum number of quantization steps
    assert no_qu_steps % 2 != 0  # no. of quantization steps has to be odd

    # threshold and quantisation levels
    threshold_levels = np.linspace(-1, 1, no_qu_steps + 1)  # generate vector of thresholds
    threshold_levels = threshold_levels[1:-1]
    threshold_levels[-1] += 1e-10  # raise the last threshold for symmetry reasons
    quantization_levels = np.linspace(-1, 1, no_qu_steps)  # generate vector of quantization steps

    qidx = np.digitize(signal, threshold_levels)
    return quantization_levels[qidx]
