from general.signals.time_signal import TimeSignal
import numpy as np
import os
import importlib
from scipy import io as sio
#import mat73
#import h5py


def import_signal_data_matlab(folder:str, file:str, return_pos_data = False):
    """
    
    :param folder: path to folder containing the measurement file 
    :param file: Measurement file name without file extension
    :return: TimeSignal array: [meas_idx_1, regime, rx_el, meas_idx_2, samples]
            positions: [meas_idx_1, coordinate_idx, meas_idx_2]
    """

    if not os.path.exists(os.path.join(folder, file + ".mat")):
        # save measurement data as .mat file with matlab engine
        matlab_mod = importlib.import_module("matlab.engine")
        eng = matlab_mod.start_matlab()
        eng.save_meas_data_as_matfile(os.path.join(folder, file), nargout=0)

    # Load .mat file with scipy
    measurement_data = sio.loadmat(os.path.join(folder, file + ".mat"), mat_dtype = True)["ret"]  # measurement_data[meas_idx,field_of_data(=0 gives signal_data)][samples, meas_idx_in_regime, regime, rx_channel]
    #measurement_data = mat73.loadmat(os.path.join(folder, file + ".mat"))["ret"]
    f_samp = measurement_data[0, 8][0, 0]

    # rearrange np.array: [meas_idx_1, regime, rx_el, meas_idx_2, samples]
    measurement_data_rearranged = []
    positions = []
    for meas_idx_1 in range(np.shape(measurement_data)[0]):
        signal_data = measurement_data[meas_idx_1, 0].astype(np.float32)
        signal_data = np.moveaxis(signal_data, 0, -1)
        signal_data = np.moveaxis(signal_data, 0, -2)
        measurement_data_rearranged.append(signal_data)         # Todo: Add t_start from Matlab measurement file
        positions.append(measurement_data[meas_idx_1, 6])

    if return_pos_data:
        return TimeSignal(np.array(measurement_data_rearranged), f_samp), np.array(positions)
    else:
        return TimeSignal(np.array(measurement_data_rearranged), f_samp)
