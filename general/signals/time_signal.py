import functools
import logging
import timeit
import warnings
from typing import Tuple, Dict, List

#import dask.base
#import distributed.protocol
#import numba
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.signal

from general.tools.signal_tools import get_quant_signal

logger = logging.getLogger(__name__)

linspace_lru = functools.lru_cache(maxsize=16)(np.linspace)


class TimeSignal(np.ndarray):
    """
    This class extends an np.ndarray to include a sampling frequency "f_samp" and an offset "t_start".
    The last dimension of the ndarray should be the samples.

    >>> t = np.linspace(0, 10, 100)
    >>> data = [np.sin(t*1), np.cos(t)]
    >>> sig = TimeSignal(data, 1/(t[1]-t[0]))
    >>> bool(np.all(t == sig.t))
    True
    >>> sig_view = sig[..., 5:]
    >>> sig_view.t[0] == sig.t[5]
    True
    >>> sig_view = sig[1:3]
    >>> np.allclose(sig_view.t, sig.t)
    True
    >>> sig_view = sig[0]
    >>> np.allclose(sig_view.t, sig.t)
    True
    >>> sig2 = TimeSignal(np.arange(100))
    >>> bool(np.all(sig2 == sig2.t))
    True

    #create a Timesignal from another Timesignal object
    >>> sig3 = TimeSignal(np.asarray(sig2), sig2)
    >>> sig3.f_samp == sig2.f_samp
    True
    >>> sig3.t_start == sig2.t_start
    True

    >>> tssum = np.sum(sig, axis=0)
    >>> type(tssum) is TimeSignal
    True

    >>> tscat = TimeSignal.concatenate([sig2, sig3])
    >>> type(tscat) is TimeSignal
    True

    >>> tsmean = sig.mean(axis=0)
    >>> type(tsmean) is TimeSignal
    True



    """

    # see docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def __new__(cls, data, f_samp=1.0, t_start=0.0):
        obj = np.asarray(data).view(cls)  # make it our type
        if isinstance(f_samp, TimeSignal):  # copy attributes from a other TimeSignal object given as second argument
            obj.f_samp = float(f_samp.f_samp)
            obj.t_start = float(f_samp.t_start)
        else:  # f_samp and t_start given
            obj.f_samp = float(f_samp)
            obj.t_start = float(t_start)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.f_samp = getattr(obj, 'f_samp', None)
        self.t_start = getattr(obj, '_new_tstart', None)
        if self.t_start is None:
            self.t_start = getattr(obj, 't_start', None)  # time of the first sample of the signal

        if self.t_start is not None:
            self.t_start = float(self.t_start)
        if self.f_samp is not None:
            self.f_samp = float(self.f_samp)
        # print(f"  TIMESIGNAL data[{self.shape}].t_start={self.t_start}  "
        #      f"obj._new_tstart={getattr(obj, '_new_tstart', '???')} "
        #      f"obj.t_start={getattr(obj, 't_start', '???')} ")

    @property
    def t(self):
        """Return the time vector with a shape=(nsamples,)"""
        t = linspace_lru(float(self.t_start), float((self.n_samples - 1) / self.f_samp + self.t_start),
                         int(self.n_samples),
                         endpoint=True)
        if self.n_samples > 1:
            assert np.abs((1 / self.f_samp) - (t[1] - t[0])) / (1 / self.f_samp) < 0.0001, \
                f"shape={self.shape} f_samp={self.f_samp} dt={t[1] - t[0]} 1/fsamp={1 / self.f_samp}"
        return t

    @property
    def n_samples(self):
        return self.shape[-1]

    @property
    def duration(self):
        """
        gives the sampled time of the signal
        :return:
        """
        return self.n_samples / self.f_samp

    @property
    def t_center(self):
        """
        time at the center of the signal
        :return:
        """
        return self.t_start + self.duration / 2

    @property
    def signal_energy(self):
        sq = np.square(self)
        m = np.mean(sq, -1)
        rms = np.sqrt(m)
        return rms

    def resample(self, factor=None, method="fft", time_shift=0.0, t_start_new=None, f_samp_new=None, fill_value="mean",
                 kind="linear"):
        """

        :param f_samp_new:
        :param time_shift:
        :param factor: fsamp_new= factor * fsamp_old
        :param method: "fft" or "interpolation"
        :return:


        >>> t = np.linspace(0, 10*np.pi, 500)
        >>> s = [[np.sin(t), np.cos(t)], [np.sin(2*t)*4 , np.cos(2*t)*3] ]
        >>> ts = TimeSignal(s, 1.0, 0)
        >>> ts2 = ts.resample(2.0)
        >>> ts_ = ts2.resample(0.5)
        >>> bool(np.mean(np.abs(ts_ - ts)) < 1e-3)
        True
        >>> all(ts.t == ts_.t)
        True
        >>> ts2 = ts.resample(2.0, method="interpolation")
        >>> ts_ = ts2.resample(0.5, method="interpolation")
        >>> bool(np.mean(np.abs(ts_-ts)) < 1e-3)
        True
        >>> all(ts.t == ts_.t)
        True
        >>> ts3 = ts.resample(1, method="interpolation", t_start_new=4)
        >>> bool(np.all(ts3[..., ts3.t == 10] == ts[..., ts.t == 10]))
        True
        >>> ts__ = ts2.resample(f_samp_new=1.0, method="interpolation")
        >>> bool(np.mean(np.abs(ts__ - ts)) < 1e-3)
        True
        >>> all(ts.t == ts__.t)
        True


        """
        assert all([l > 0 for l in self.shape])

        if t_start_new is None:
            t_start_new = self.t_start
        assert np.isscalar(time_shift) and not (np.isnan(time_shift) or np.isnan(time_shift))

        if f_samp_new is None:
            nsamp_new = int(np.ceil(self.n_samples * factor))
            f_samp_new = nsamp_new / self.duration
        else:
            nsamp_new = int(np.ceil(self.n_samples / self.f_samp * f_samp_new))

        ts_new = TimeSignal(np.empty(self.shape[:-1] + (nsamp_new,), dtype=self.dtype), f_samp_new,
                            t_start_new)

        if f_samp_new == self.f_samp and time_shift == 0.0 and t_start_new == self.t_start:
            ts_new[...] = self
        elif method == "fft":
            if time_shift != 0.0 or t_start_new != self.t_start:
                raise NotImplementedError
            ts_new[...] = scipy.signal.resample(self, nsamp_new, axis=-1)
        elif method == "interpolation":

            onedfunc = lambda data: scipy.interpolate.interp1d(self.t + time_shift, data, axis=-1, copy=False,
                                                               assume_sorted=True, kind=kind,
                                                               fill_value=fill_value if not fill_value == "mean" else np.mean(
                                                                   data, axis=-1), bounds_error=False)(ts_new.t)
            ts_new[...] = np.apply_along_axis(onedfunc, -1, self)
        else:
            raise NotImplementedError()
        return ts_new

    def resample_tvect(self, tvect, time_shift=0.0, fill_value="mean", kind="linear"):
        """
        resamples the signal to a given time vector using scipy.interpolate.interp1d

        :param fill_value: either a value according to scipy.interpolate.interp1d, or "mean" to fill with the mean value
                of the respective signals
        :param time_shift: introduce a time shift
        :param tvect: new time vector
        :return:


        >>> ts1 = TimeSignal(np.empty((4, 500)), 10, 0.0)
        >>> ts2 = TimeSignal(np.empty((4, 500)), 10, 0.15)
        >>> for ts in ts1, ts2:
        ...     ts[0, :] = np.sin(ts.t)
        ...     ts[1, :] = np.cos(ts.t)
        ...     ts[2, :] = np.sin(2 * ts.t) * 4
        ...     ts[3, :] = np.cos(2 * ts.t) * 3
        >>> ts1_2 = ts1.resample_tvect(ts2.t)
        >>> bool(np.mean(np.abs(ts1_2 - ts2)[...,ts2.t < ts1.t[-1]]) < 8e-3)
        True
        >>> all(ts1_2.t == ts2.t)
        True
        >>> np.allclose(ts1.resample_tvect(np.pi), [[0], [-1], [0], [3]], atol=0.05)
        True
        """
        tvect = np.array(tvect)
        try:  # check if tvect is just a scalar
            len(tvect)
        except TypeError:
            tvect = [tvect]

        ts_new = TimeSignal.from_t(np.empty(self.shape[:-1] + (len(tvect),), dtype=self.dtype), tvect)

        onedfunc = lambda data: scipy.interpolate.interp1d(self.t + time_shift, data, axis=-1, copy=False,
                                                           assume_sorted=True, kind=kind,
                                                           fill_value=fill_value if not fill_value == "mean" else np.mean(
                                                               data, axis=-1), bounds_error=False)(ts_new.t)
        ts_new[...] = np.apply_along_axis(func1d=onedfunc, axis=-1, arr=self)

        return ts_new

    def hilbert(self) -> 'TimeSignal':
        return TimeSignal(scipy.signal.hilbert(self, axis=-1), self)

    def abs(self) -> 'TimeSignal':
        return TimeSignal(np.abs(self), self)

    def angle(self) -> 'TimeSignal':
        return TimeSignal(np.angle(self), self)

    def unwrap(self, **kwargs) -> 'TimeSignal':
        return TimeSignal(np.unwrap(self, axis=-1, **kwargs), self)

    def filtfilt(self, b, a, axis=-1, **kwargs) -> 'TimeSignal':
        """

        :param b: filter coefficient
        :param a: filter coefficient
        :param axis: filter along which axis (default:-1)
        :return: the filtered signal

        #>>> x = TimeSignal(np.arange(10)**2,1.0,0)
        #>>> x[4:7]
        #TimeSignal([16, 25, 36])
        #>>> xnew = x.filtfilt([1.0/3, 1.0/3, 1.0/3],[1], padtype=None)
        #>>> xnew[5] == np.mean(x[4:7])

        """
        return TimeSignal(scipy.signal.filtfilt(b, a, x=self, axis=axis, **kwargs).astype(np.float32), self.f_samp,
                          self.t_start)

    def filt(self, b, a, axis=-1, **kwargs) -> 'TimeSignal':
        """

        :param b: filter coefficient
        :param a: filter coefficient
        :param axis: filter along which axis (default:-1)
        :return: the filtered signal

        #>>> x = TimeSignal(np.arange(10)**2,1.0,0)
        #>>> x[4:7]
        #TimeSignal([16, 25, 36])
        #>>> xnew = x.filtfilt([1.0/3, 1.0/3, 1.0/3],[1], padtype=None)
        #>>> xnew[5] == np.mean(x[4:7])

        """
        return TimeSignal(scipy.signal.lfilter(b, a, x=self, axis=axis, **kwargs).astype(np.float32), self.f_samp,
                          self.t_start)

    def median_filter(self, **kwargs) -> 'TimeSignal':
        """
        scipy.ndimage.median_filter
        :param kwargs:
        :return:
        """
        return TimeSignal(scipy.ndimage.median_filter(self, **kwargs), self.f_samp, self.t_start)

    def nd_convolve(self, *args, **kwargs) -> 'TimeSignal':
        """
        scipy.ndimage.filters.convolve
        :param kwargs:
        :return:
        """
        return TimeSignal(scipy.ndimage.convolve(self, *args, **kwargs), self.f_samp, self.t_start)

    def fft(self):
        """
        gives the complex FFT coefficients of the signal
        :return freq, sig_fft complex FFT coefficients
        """
        N = self.shape[-1]
        sig_fft = np.fft.fft(self, axis=-1)
        freq = np.fft.fftfreq(N, d=1 / self.f_samp)
        return freq, sig_fft

    def crop_signal(self, t_begin=0, t_end=0) -> 'TimeSignal':
        """
        Removes the samples of all dimensions (except the last) according to sampling frequency and corrects the start
        time of the signal
        :param t_begin: time to cut the signal from the beginning
        :param t_end: time to crop
        :return: cropped TimeSignal

        >>> ts = TimeSignal(np.arange(100), 1, 0)
        >>> ts_short = ts.crop_signal(t_begin=10, t_end=10)
        >>> ts_short # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        TimeSignal(data=[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
        ...34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57
        ...58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81
        ...82 83 84 85 86 87 88 89], f_samp=1.0, t_start=10.0)
        >>> ts = TimeSignal(np.arange(10), 1, 0)
        >>> ts_short = ts.crop_signal(t_begin=5)
        >>> ts_short
        TimeSignal(data=[5 6 7 8 9], f_samp=1.0, t_start=5.0)
        """
        return self.timeslice(self.t_start + t_begin, self.t[-1] - t_end)

    def timeslice(self, t_min=-np.inf, t_max=np.inf) -> 'TimeSignal':
        try:
            idx_start = np.flatnonzero(self.t >= t_min)[0]
            idx_stop = np.flatnonzero(self.t <= t_max)[-1]
            s = slice(idx_start, idx_stop + 1)
            t_start = self.t[idx_start]
        except IndexError:
            s = slice(0, 0)
            t_start = self.t_start
        return TimeSignal(self[..., s], self.f_samp, t_start)

    def strip_zeros(self) -> 'TimeSignal':
        """
        strips zeros from the beginning and the end
        >>> ts = TimeSignal(np.arange(-10, 15), 1, 0)
        >>> ts[np.abs(ts) > 5] = 0
        >>> ts_strip = ts.strip_zeros()
        >>> ts_strip[0] == -5
        True
        >>> ts_strip[-1] == 5
        True
        >>> ts_strip.t_start == 5
        True
        >>> bool(ts[ts.t == 8.] == ts_strip[ts_strip.t == 8.])
        True
        >>> ts2 = TimeSignal(np.zeros(10), 10, 1)
        >>> ts2_strip = ts2.strip_zeros()
        >>> ts2_strip.shape[0] == 0
        True
        >>> ts2_strip.t_start == 1
        True

        :return: new signal without zeros
        :rtype: TimeSignal
        """
        dims = tuple(range(self.ndim - 1))
        # TODO Lars, "np.any" may be redundant and may be removed
        nz = np.flatnonzero(np.any(self, axis=dims))
        try:
            s = slice(nz[0], nz[-1] + 1)
            t_start = self.t[s][0]
        except IndexError:
            s = slice(0, 0)
            t_start = self.t_start
        return TimeSignal(self[..., s], self.f_samp, t_start)

    def pad_with_zeros(self, before=0, after=0, value=0.0) -> 'TimeSignal':
        """
        :param before:
        :param after:
        :param value:
        :return:

        >>> before = 5
        >>> after = 10
        >>> padvalue = -0.0
        >>> unpaddeddata = TimeSignal(np.arange(10, dtype=np.float), f_samp=0.5, t_start=1.0)
        >>> paddeddata = unpaddeddata.copy().pad_with_zeros(before, after, value=padvalue)
        >>> paddeddata # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
         TimeSignal(data=[-0. -0. -0. -0. -0.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9. -0. -0. -0.
        ...-0. -0. -0. -0. -0. -0. -0.], f_samp=0.5, t_start=-9.0)
        >>> paddeddata.f_samp
        0.5
        >>> paddeddata.t_start == 1.0 - before / 0.5
        True
        >>> before = 3
        >>> after = 0
        >>> paddeddata2 = unpaddeddata.copy().pad_with_zeros(before, after, value=padvalue)
        >>> paddeddata2 # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        TimeSignal(data=[-0. -0. -0.  0.  1.  2.  3.  4.  5.  6.  7.  8.  9.], f_samp=0.5, t_start=-5.0)

        """

        newshape = np.array(self.shape, copy=True)
        newshape[-1] += before + after
        newpat = np.full(newshape, dtype=self.dtype, fill_value=value)
        newpat[..., before:newshape[-1] - after] = self[..., :]
        newpat = TimeSignal(newpat, self.f_samp,
                            self.t_start - before / self.f_samp)
        return newpat

    def window_energy_based(self, energy_lim=None, duration_lim=None, sample_lim=None, inplace=True):
        """Trim the signal to minimize the lost energy fraction per sample (EFPS) under the given constrains.

        window the signal: trim it from left and right so that the lost energy fraction per sample (EFPS) is minimized
        and the following conditions are met:

            * the retained energy fraction is between energy_lim[0] and energy_lim[1]
            * the duration is between duration_lim[0] and duration_lim[1]
            * the number of samples is between sample_lim[0] and sample_lim[1]

        if inplace is True: the trimmed samples are replaced with zero
        if inplace is False: the trimmed samples are replaced with zero

        :return: a tuple with the TimeSignal and a dictionary containing information about the trimming.
        :rtype: (TimeSignal, dict)


        >>> s = TimeSignal(1 - np.abs(np.arange(-1, 1, 0.1)), f_samp=10, t_start=1)
        >>> strim, info = s.window_energy_based(energy_lim=[0.5, 0.9], inplace=False)
        >>> strim2, info2 = s.window_energy_based(sample_lim=[5, 5], inplace=False)
        >>> bool(strim2.n_samples == 5)
        True
        >>> strim2.t_start == s.t[info2["n_trim"][0]]
        True
        >>> strim3, info3 = s.window_energy_based(sample_lim=[5, 5], inplace=True)
        >>> bool(np.count_nonzero(strim3 > 0) == 5)
        True
        """
        info = {}
        final_sample_lim = np.array([-np.inf, np.inf])

        def _update_final_sample_lim(values):
            final_sample_lim[0] = max(values[0], final_sample_lim[0])
            final_sample_lim[1] = min(values[1], final_sample_lim[1])

        ret = self
        N = self.shape[-1]
        n_samp_cut = np.arange(N, dtype=float)
        n_samp_cut_all = n_samp_cut[:, np.newaxis] + n_samp_cut[np.newaxis, :]

        energy_frac = np.sum((self - np.mean(self, keepdims=True, axis=-1)) ** 2, axis=tuple(range(self.ndim - 1)))
        energy_frac /= np.sum(energy_frac)
        ef_left = np.zeros_like(n_samp_cut)  # ef_left[n] is the fraction of energy that is lost,
        # when cutting n samples from the left
        ef_left[1:] = np.cumsum(energy_frac[:-1:])

        ef_right = np.zeros_like(n_samp_cut)
        ef_right[1:] = np.cumsum(energy_frac[:0:-1])  # ef_right[n] is the fraction of energy that is lost,
        # when cutting n samples from the right

        ef_all = ef_left[:, np.newaxis] + ef_right[np.newaxis, :]  # epfs_all[left_nsamp, right_nsamp]

        # enforce energy limits
        if energy_lim is not None:
            ef_all[1 - ef_all < energy_lim[0]] = np.inf
            ef_all[1 - ef_all > energy_lim[1]] = np.inf

        efps_all = ef_all / n_samp_cut_all

        # apply conditions
        efps_all[n_samp_cut_all >= N] = np.inf  # never cut more that the whole signal
        efps_all[0, 0] = 1e10  # always cut something, except there is no other option

        if duration_lim is not None:
            _update_final_sample_lim(np.array(duration_lim) * self.f_samp)

        if sample_lim is not None:
            _update_final_sample_lim(np.array(sample_lim))

        # enforce sample/duration limits
        efps_all[n_samp_cut_all > N - final_sample_lim[0]] = np.inf
        efps_all[n_samp_cut_all < N - final_sample_lim[1]] = np.inf

        # find minimum efps
        left_trim, right_trim = np.unravel_index(np.argmin(efps_all, axis=None), efps_all.shape)

        info["n_trim"] = left_trim, right_trim
        info["retained_energy_frac"] = 1 - ef_all[left_trim, right_trim]
        info["efps"] = efps_all[left_trim, right_trim]

        if info["efps"] == np.inf:  # no min found
            raise ValueError("window_energy_based: could not find a valid window with the given limits...")
        if inplace:
            if left_trim:
                self[..., 0:left_trim] = 0
            if right_trim:
                self[..., -right_trim:] = 0
        else:
            ret = self[..., left_trim:-right_trim if right_trim > 0 else self.n_samples]

        return ret, info

    def time_reverse(self) -> 'TimeSignal':
        """
        mirrors the signal at t=0, so tnew = -t
        :return: time reversed TimeSignal

        >>> ts = TimeSignal(np.arange(11), f_samp=1, t_start=-5)
        >>> ts_tr = ts.time_reverse()
        >>> repr(ts) == repr(ts_tr.time_reverse())
        True

        """
        return TimeSignal(self[..., ::-1], self.f_samp, -(self.t_start + self.duration))

    # def swapaxes(self, axis1, axis2) -> 'TimeSignal':
    #     time_axis = self.ndim
    #     if axis1 == time_axis or axis2 == time_axis:
    #         warnings.warn("axis1 or axis2 includes the time dimension! That is probably a mistake")
    #     return TimeSignal(np.swapaxes(self, axis1, axis2), self.f_samp, self.t_start)

    def convolve(self, kernel) -> 'TimeSignal':
        """
        Signal convolution regarding the time (samples) axis
        :param pattern: TimeSignal to convolve with
        :return:

        This implementation messes up the timing, so use convolve_sane instead...

        >>> s = TimeSignal(np.arange(10).reshape(2,5), 10e6, 15e-6)
        >>> p = TimeSignal([1, 3, 2], 10e6, 5e-6)
        >>> new = s.convolve(p)
        >>> print(np.array(new)) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[...5. 11. 17.]
         [...35. 41. 47.]]
        >>> new.t_start
        2e-05

        """
        if kernel.f_samp is not None:
            assert self.f_samp == kernel.f_samp  ### same sampling frequencies
        assert len(kernel.shape) == 1  ### only one dimensional pattern allowed
        assert self.shape[-1] >= kernel.shape[-1]  ### for valid convolution

        results = np.zeros(self.shape[:-1] + (self.shape[-1] - kernel.shape[-1] + 1,))
        ### iterate over all dimensions, but not the samples
        for idxs in np.ndindex(self.shape[:-1]):
            results[idxs] = np.convolve(self[idxs], kernel, mode='valid')

        return TimeSignal(results, self.f_samp, self.t_start + kernel.t_start)

    def convolve_sane(self, kernel, mode="valid", benchmark=False) -> 'TimeSignal':
        """
        Signal convolution regarding the time (samples) axis
        :param kernel: TimeSignal to convolve with
                if kernel is an 1d array then ...

                if the kernel has a shape matching self except the last dimension ... sum
        :param mode: see np.convolve
        :return:

        >>> s = TimeSignal(np.arange(10).reshape(2,5), 10e6, 15e-6)
        >>> p = TimeSignal([1, 3, 2], 10e6, 5e-6)
        >>> new = s.convolve_sane(p)
        >>> print(np.array(new)) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [[...5. 11. 17.]
         [...35. 41. 47.]]
        >>> new.t_start
        2.025e-05
        """

        # todo: use gpus:
        # see reikna and 
        #  http://hgomersall.github.io/pyFFTW/sphinx/tutorial.html#monkey-patching-3rd-party-libraries
        def _apply_kernel(N_samp_new, data, kernel, mode):
            """

            :param data: shape  ( rep, sum, t)
            :param kernel:  (  sum, t)
            :param mode: see np.convolve

            :return:  results: shape (rep, tnew)
            """

            results = np.zeros((reps, N_samp_new), dtype=np.float64)
            for rep in range(data.shape[0]):
                for k_idx in range(kernel.shape[0]):
                    # results[rep, :] += np.convolve(data[rep, k_idx, :], kernel[k_idx, :], mode=mode)
                    results[rep, :] += scipy.signal.fftconvolve(data[rep, k_idx, :], kernel[k_idx, :], mode=mode)
            return results

        # @numba.njit(parallel=True, cache=True)
        # def _apply_kernel_numba(N_samp_new, data, kernel, mode):
        #     """
        #
        #     :param data: shape  ( rep, sum, t)
        #     :param kernel:  (  sum, t)
        #     :param mode: see np.convolve
        #
        #     :return:  results: shape (rep, tnew)
        #     """
        #
        #     results = np.zeros((reps, N_samp_new))
        #     if mode == "same":
        #         offset = np.floor(kernel.shape[-1] / 2)
        #     elif mode == "full":
        #         offset = 0
        #     elif mode == "valid":
        #         offset = kernel.shape[-1] - 1
        #     assert kernel.shape[-1] <= data.shape[-1]
        #
        #     for rep in range(data.shape[0]):
        #         for k_idx in range(kernel.shape[0]):
        #             for n in range(N_samp_new):
        #                 for m in range(kernel.shape[-1]):
        #                     idx = int(offset + n - m)
        #                     if 0 <= idx < data.shape[-1]:
        #                         results[rep, n] += data[rep, k_idx, idx] * kernel[k_idx, m]
        #     return results

        if kernel.f_samp is not None:
            if not (self.f_samp == kernel.f_samp):  # same sampling frequencies
                warnings.warn("fsamp of signal ({} Hz) differs vs. kernel ({})".format(self.f_samp, kernel.f_samp))

        if mode == 'full':
            N_samp_new = self.shape[-1] + kernel.shape[-1] - 1
        elif mode == 'same':
            N_samp_new = max(self.shape[-1], kernel.shape[-1])
        elif mode == "valid":
            N_samp_new = max(self.shape[-1], kernel.shape[-1]) - min(self.shape[-1], kernel.shape[-1]) + 1

        if kernel.ndim == 1:  # apply the same kernel to all signals
            reps = int(np.prod(self.shape[:-1]))
            finalshape = self.shape[:-1] + (N_samp_new,)
        else:  # sum over matching kernels
            reps = int(np.prod(self.shape[:-2]))
            finalshape = self.shape[:-2] + (N_samp_new,)

        kernel_2d = np.array(kernel, copy=False, ndmin=2)
        data_3d = self.reshape((reps, kernel_2d.shape[0], self.shape[-1]))  # (reps,sum,t)

        results = _apply_kernel(N_samp_new, data_3d, kernel_2d, mode)
        if benchmark:
            results1 = _apply_kernel(N_samp_new, data_3d, kernel_2d, mode)
            results2 = _apply_kernel_numba(N_samp_new, data_3d, kernel_2d, mode)

            t = [
                timeit.timeit(functools.partial(f, N_samp_new=N_samp_new, data=data_3d, kernel=kernel_2d, mode=mode),
                              number=3)
                for f in [_apply_kernel, _apply_kernel_numba]]
            maxdiff = np.max(np.abs(results1 - results2))

            shift = None

            for s in range(-15, 15):
                if np.all(results1[:, 20:20 + 10] == results2[:, 20 + s:20 + s + 10]):
                    shift = s

            print(
                f"data({self.shape}) kernel={kernel.shape} mode={mode} "
                f"shift={shift} tnp ={t[0]}s tnumba={t[1]}s speedup={t[0] / t[1]:10.2f} maxdiff={maxdiff}")
            # assert maxdiff == 0,                print(f"MAXDIFF={maxdiff}, mode={mode} \nrnp={results1}\nrnumba={results2}")
        results = results.reshape(finalshape)

        # the convolution will shift thw center of our signal by the center time of the kernel
        new_t_start = self.t_center + kernel.t_center - N_samp_new / self.f_samp / 2
        return TimeSignal(results, self.f_samp, new_t_start)

    def timeshift_to_match(self, other: "TimeSignal"):
        """
        calculates a time shift to match this signal with another signal by using cross correlation
        :param other: the signal to match
        :return: t_shift, sig_new

        >>> interactive = False
        >>> f = 10
        >>> t_shift = 0.7
        >>> s1 = TimeSignal(np.empty((2, 200, )), f_samp=f*10, t_start=-1)
        >>> s2 = s1.copy()
        >>> s1[...] = scipy.signal.gausspulse(s1.t, fc=f)
        >>> s2[...] = scipy.signal.gausspulse(s2.t + t_shift, fc=f) * 1.5
        >>> s2[np.abs(s2) < 0.01] = 0
        >>> s2 = s2.strip_zeros()
        >>> t_shift_est, s3, xcorr = s1.timeshift_to_match(s2)
        >>> bool(np.all(np.abs(t_shift - t_shift_est) < 0.01))
        True
        >>> if interactive:
        ...     import matplotlib.pyplot as plt
        ...     fig = plt.figure()
        ...     ax = fig.add_subplot(2, 1, 1)
        ...     p = s1.plot(ax)
        ...     p = s2.plot(ax)
        ...     p = s3.plot(ax)
        ...     ax2 = fig.add_subplot(2, 1, 2)
        ...     xcorr.plot(ax2)
        ...     plt.show()
        """
        xcorr = self.convolve_sane(other.time_reverse())  # type:TimeSignal
        _, t_shift = xcorr.find_peak()

        sig_new = self.resample_tvect(other.t, time_shift=-t_shift)
        return t_shift, sig_new, xcorr

    def find_peak(self, interpolation="gaussian"):
        """

        :param interpolation:  either None , "gaussian" or "centroid"
        :return: peak_samp, t_peak

        >>> interactive = False
        >>> s = TimeSignal(np.empty((4, 100)), f_samp=1, t_start=1)
        >>> t_peak = np.array([1, 12.1, 43.5, 83.66])
        >>> s[...] = 100 - (s.t[np.newaxis, :] - t_peak[:, np.newaxis]) ** 2
        >>> peak_samp_est, t_peak_est = s.find_peak(interpolation="gaussian")
        >>> bool(np.all(np.abs(t_peak - t_peak_est) < 0.6))
        True
        >>> peak_samp_est, t_peak_est = s.find_peak(interpolation="centroid")
        >>> bool(np.all(np.abs(t_peak - t_peak_est) < 0.6))
        True
        >>> f= 1
        >>> t_peak=0.1
        >>> s = TimeSignal(np.empty((2, 200, )), f_samp=f*10, t_start=-1)
        >>> s[...] = scipy.signal.gausspulse(s.t - t_peak , fc=f)
        >>> peak_samp_est, t_peak_est = s.find_peak(interpolation="hilbert")
        >>> bool(np.all(np.abs(t_peak - t_peak_est) < 0.2 * 1 / s.f_samp))
        True
        >>> if interactive:
        ...     import matplotlib.pyplot as plt
        ...     fig = plt.figure()
        ...     ax = fig.add_subplot(1, 1, 1)
        ...     p = s.plot(ax)
        ...     plt.show()

        """
        peak_samp_int = np.array(np.argmax(self, axis=-1), dtype=int)

        c = np.take_along_axis(self, np.expand_dims(peak_samp_int, axis=-1), axis=-1)[..., 0]
        cl = np.take_along_axis(self, np.expand_dims(np.clip(peak_samp_int - 1, 0, None),
                                                     axis=-1), axis=-1)[..., 0]
        cr = np.take_along_axis(self, np.expand_dims(np.clip(peak_samp_int + 1, None, self.n_samples - 1),
                                                     axis=-1), axis=-1)[..., 0]

        peak_samp = np.array(peak_samp_int, dtype=float)
        if interpolation == "gaussian":
            peak_samp_diff = (np.log(cl) - np.log(cr)) / (2 * np.log(cl) - 4 * np.log(c) + 2 * np.log(cr))
            if np.any(np.isinf(peak_samp_diff)) or np.any(np.isnan(peak_samp_diff)):
                interpolation = "centroid"
                logger.warning("Falling back to centroid interpolation, because negative values were encountered ...")
            else:
                peak_samp += peak_samp_diff

        if interpolation == "centroid":
            peak_samp = np.array(((peak_samp - 1) * cl + peak_samp * c + (peak_samp + 1) * cr) / (cl + c + cr))

        if interpolation == "hilbert":  # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.362.2217&rep=rep1&type=pdf
            hilb = np.imag(self.hilbert())
            hilb_l = np.take_along_axis(hilb, np.expand_dims(np.clip(peak_samp_int - 1, 0, None),
                                                             axis=-1), axis=-1)[..., 0]
            hilb_r = np.take_along_axis(hilb, np.expand_dims(np.clip(peak_samp_int + 1, None, self.n_samples - 1),
                                                             axis=-1), axis=-1)[..., 0]
            # find zero crossing
            peak_samp += -1 - 2 * hilb_l / (hilb_r - hilb_l)

        t_peak = self.t_start + peak_samp / self.f_samp

        return peak_samp, t_peak

    # def estimate_phasediff_hilbert_angle(self, sigB, t_ref=None, mean_time=None):
    #     """
    #
    #     :param sigA:
    #     :param sigB:
    #     :param t_ref:
    #     :param mean_time:
    #     :return:
    #
    #
    #     >>> ts1 = TimeSignal(np.empty((500, )), 10, 0)
    #     >>> ts2 = TimeSignal(np.empty((500, )), 10, 0)
    #     >>> f = 2
    #     >>> deltat = 0.22
    #     >>> ts1[...] = np.sin(ts1.t * f * 2 * np.pi)
    #     >>> ts2[...] = np.sin((ts2.t - deltat) * f * 2 * np.pi)
    #     >>> phase_diff, phase_diff_mean, freq_mean = ts1.estimate_phasediff_hilbert_angle(ts2)
    #     >>> phase_diff_mean / 2 / np.pi / freq_mean
    #     >>> np.isclose(phase_diff_mean / 2 / np.pi / freq_mean, deltat)
    #     True
    #     >>> np.isclose(freq_mean, f)
    #     True
    #
    #     """
    #     if mean_time is None:
    #         mean_time = (-np.inf, np.inf)
    #
    #     sigA_phase = self.hilbert().angle().unwrap()
    #     freqA = np.diff(sigA_phase) / 2 / np.pi * self.f_samp
    #     sigB_phase = sigB.hilbert().angle().unwrap()
    #     freqB = np.diff(sigB_phase) / 2 / np.pi * sigB.f_samp
    #     phase_diff = sigA_phase - sigB_phase  # type:TimeSignal
    #
    #     if t_ref is not None:
    #         phase_ref = phase_diff.resample_tvect(t_ref)
    #         offset = np.rint(phase_ref / 2 / np.pi) * 2 * np.pi
    #         phase_diff -= offset
    #
    #     # calc mean
    #     phase_diff_mean = float(np.mean(phase_diff.timeslice(*mean_time)))
    #     freq_mean = float(np.mean(freqA.timeslice(*mean_time) + freqB.timeslice(*mean_time))) / 2
    #     return phase_diff, phase_diff_mean, freq_mean

    def est_autocorrelation_phasor(self):
        return TimeSignal.from_t(self[..., slice(1, None)] * self.conj()[..., slice(None, -1)],
                                 self.t[0:-1] + 0.5 / self.f_samp)

    def est_phasediff_hilbert(self, sigB, mean_time=None):
        """

        :param sigA:
        :param sigB:
        :param t_ref:
        :param mean_time:
        :return:


        >>> ts1 = TimeSignal(np.empty((500, )), 10, 0)
        >>> ts2 = TimeSignal(np.empty((500, )), 10, 0)
        >>> f = 2
        >>> deltat = 0.22
        >>> ts1[...] = np.sin(ts1.t * f * 2 * np.pi)
        >>> ts2[...] = np.sin((ts2.t - deltat) * f * 2 * np.pi)
        >>> phase_diff, phase_diff_mean, freq_mean = ts1.est_phasediff_hilbert(ts2)
        >>> bool(np.isclose(phase_diff_mean / 2 / np.pi / freq_mean, deltat))
        True
        >>> bool(np.isclose(freq_mean, f))
        True

        """
        assert np.allclose(self.t, sigB.t)
        if mean_time is None:
            mean_time = (-np.inf, np.inf)

        sigA = self.hilbert()
        sigB = sigB.hilbert()

        prod = sigA * sigB.conj()

        phase_diff = prod.angle()

        # calc mean
        diff_phasor = np.nansum(prod.timeslice(*mean_time), axis=-1)
        ac_phasor_A = np.nansum(sigA.est_autocorrelation_phasor().timeslice(*mean_time), axis=-1)
        ac_phasor_B = np.nansum(sigB.est_autocorrelation_phasor().timeslice(*mean_time), axis=-1)
        phase_diff_mean = np.angle(diff_phasor)
        freq_mean = np.angle(ac_phasor_A + ac_phasor_B) / 2 / np.pi * self.f_samp
        return phase_diff, phase_diff_mean, freq_mean

    def get_quant_signal(self, no_qu_steps=3, use_pdm=True):
        if no_qu_steps is None:
            return self
        if use_pdm:
            if no_qu_steps != 3:
                raise NotImplementedError
            return self.pulse_density_modulation()
        else:
            return TimeSignal(get_quant_signal(self, no_qu_steps=no_qu_steps), self.f_samp, self.t_start)

    def pulse_density_modulation(self):
        """

        implements PDM (https://en.wikipedia.org/wiki/Pulse-density_modulation) for a signal in the range -1...1

        >>> s = TimeSignal(np.empty(100), 10, 0)
        >>> s[...] = np.sin(s.t) * (s.t < 5) *0.8
        >>> sq = s.pulse_density_modulation()
        >>> sreq = sq.filtfilt(np.ones(3) / 3,1)
        >>> bool(np.all(sq[sq.t > 6] == 0))
        True
        >>> bool(np.sum((sreq - s) ** 2) / np.sum(s ** 2) < 0.05)
        True
        >>> interactive = False
        >>> if interactive:
        ...     import matplotlib.pyplot as plt
        ...     fig = plt.figure()
        ...     ax = fig.add_subplot(1, 1, 1)
        ...     p = ax.plot(s)
        ...     p = ax.plot(sq)
        ...     p = ax.plot(sreq)
        ...     plt.grid(True)
        ...     plt.show()

        :return:
        """

        ret = np.zeros_like(self)
        error = np.zeros(self.shape[:-1])
        thresh = 2 / 3
        for i in range(self.shape[-1] - 1):
            diff = self[..., i] - error
            ret[..., i] = (diff > thresh) + (diff < -thresh) * -1  # quantize to -1, 0 or 1
            error = ret[..., i] - self[..., i] + error
        return TimeSignal(ret, self.f_samp, self.t_start)

    def resample_window_quant(self, quant_gain=1.0, quant_steps=3, use_pdm=True, n_samp=None,
                              targ_f_samp=None):
        """
        :param n_samp: limits the number of samples by cropping the signal from both sides
        :param use_pdm: use pulse density modulation
        :param targ_f_samp: desired sampling frequency
        :param quant_gain: float, quantization gain for TX pattern
        :param quant_steps: int, count of quantization steps. Currently only 3 steps possible for PAUDV

        :return  quantized, resampled and windowed pattern

        """

        tx_pattern = self

        # resample
        if targ_f_samp is not None:
            factor = targ_f_samp / tx_pattern.f_samp
            tx_pattern = tx_pattern.resample(factor=factor, method="interpolation")

        # window
        if n_samp is not None:
            n_samp = min(n_samp, tx_pattern.n_samples)
            tx_pattern, info = tx_pattern.window_energy_based(
                sample_lim=[n_samp, n_samp], inplace=False)

        # quantization
        if quant_steps is not None:
            tx_pattern /= np.std(tx_pattern, axis=-1, keepdims=True) * 2 ** 0.5  # bring the rms to sqrt(2),
            # that is the same as a sinosoidal signal with an amplitude of 1
            tx_pattern *= quant_gain
            tx_pattern = tx_pattern.get_quant_signal(no_qu_steps=quant_steps, use_pdm=use_pdm)

        return tx_pattern

    def plot(self, ax=None, **kwargs):
        return ax.plot(self.t, self.T, **kwargs)

    def plot2d(self, ax=None, **kwargs):
        extent = [self.t[0], self.t[-1], 0, self.shape[0]]  # xmin, xmax, ymin, ymax
        ax.set_xlabel('t / s')
        return ax.imshow(self, cmap=None, aspect='auto', interpolation='none', extent=extent, **kwargs)

    @classmethod
    def from_t(cls, data, t):
        t = np.array(t)
        assert t.ndim == 1
        f_samp = 1 / (t[1] - t[0]) if len(t) > 1 else 1.0
        t_start = t[0] if len(t) > 0 else 0
        ts = TimeSignal(data, f_samp=f_samp, t_start=t_start)
        assert np.all(np.abs(t - ts.t) <= np.abs(0.001 * t))
        assert data.shape[-1] == t.shape[-1]
        return ts

    def __getslice__(self, start, stop):
        """This solves a subtle bug, where __getitem__ is not called, and all
        the dimensional checking not done, when a slice of only the first
        dimension is taken, e.g. a[1:3]. From the Python docs:
           Deprecated since version 2.0: Support slice objects as parameters
           to the __getitem__() method. (However, built-in types in CPython
           currently still implement __getslice__(). Therefore, you have to
           override it in derived classes when implementing slicing.)
        """
        return self.__getitem__(slice(start, stop))

    # https://stackoverflow.com/questions/31282764/when-subclassing-a-numpy-ndarray-how-can-i-modify-getitem-properly
    def __getitem__(self, index):
        # handle slicing with correct t_start
        try:
            _new_tstart = None
            if isinstance(index, (slice, int, np.ndarray)):
                if self.ndim == 1:
                    _new_tstart = self.t[index]
            else:
                _new_tstart = self.t[index[-1]]
            if _new_tstart is not None and not np.isscalar(_new_tstart):
                _new_tstart = _new_tstart[..., 0]
            self._new_tstart = float(_new_tstart) if _new_tstart is not None else None
        except (IndexError, AttributeError, TypeError, ValueError):
            pass

        # print(f"  TIMESIGNAL getslice data[{self.shape}].t_start={self.t_start} index={index}")
        ret = super(TimeSignal, self).__getitem__(index)
        self._new_tstart = None
        return ret

        # overwrite __reduce__ and __setstate__ allow (un)pickling with additional attributes in np.ndarray

    # http://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.f_samp, self.t_start,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # Set the additional attributes
        self.f_samp = float(state[-2])
        self.t_start = float(state[-1])
        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-2])

    @classmethod
    def __from_data_and_aux__(cls, data, aux):
        return TimeSignal(data, aux["f_samp"], aux["t_start"])

    def __to_data_and_aux__(self):
        return np.array(self, copy=False), {"f_samp": self.f_samp, "t_start": self.t_start}

    def __str__(self):
        return "TimeSignal(data={}, f_samp={}, t_start={})".format(super(TimeSignal, self).__str__(), self.f_samp,
                                                                   self.t_start)

    def __repr__(self):
        return str(self)

    @staticmethod
    def stack(arrays, axis=0, out=None):
        f_samp = None
        t_start = None
        for a in arrays:
            if f_samp is None:
                f_samp = a.f_samp
                t_start = a.t_start
            else:
                assert f_samp == a.f_samp
                assert t_start == a.t_start
        if out is not None:
            raise NotImplementedError

        return TimeSignal(np.stack(arrays, axis=axis, out=out), f_samp=f_samp, t_start=t_start)

    @staticmethod
    def concatenate(arrays, axis=0, out=None):
        f_samp = None
        t_start = None
        try:
            for a in arrays:
                if f_samp is None:
                    f_samp = a.f_samp
                    t_start = a.t_start
                else:
                    assert f_samp == a.f_samp
                    assert t_start == a.t_start
        except AttributeError:
            raise TypeError(f"The arrays lack .f_samp or .t_start: {[type(e) for e in arrays]}")

        if out is not None:
            raise NotImplementedError
        return TimeSignal(np.concatenate(arrays, axis=axis, out=out), f_samp=f_samp, t_start=t_start)

"""
@distributed.protocol.dask_serialize.register(TimeSignal)
def serialize(ts: TimeSignal) -> Tuple[Dict, List[bytes]]:
    data_header, data_frames = distributed.protocol.serialize(np.asarray(ts))
    header = {"f_samp": ts.f_samp, "t_start": ts.t_start, "data": data_header}
    return header, data_frames


@distributed.protocol.dask_deserialize.register(TimeSignal)
def deserialize(header: Dict, frames: List[bytes]) -> TimeSignal:
    data = distributed.protocol.deserialize(header["data"], frames)
    return TimeSignal(data, header["f_samp"], header["t_start"])


@dask.base.normalize_token.register(TimeSignal)
def tokenize_TimeSignal(x):
    return (np.asarray(x), x.f_samp, x.t_start)
"""