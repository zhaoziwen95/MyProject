from matplotlib import pylab as plt
from main_dev.PYTHON.general.tools.import_measurement_data_matlab import import_signal_data_matlab
import numpy as np
from main_dev.PYTHON.general.signals.time_signal import TimeSignal

# Pfad zu den Messdaten angeben
path = "/home/ziwen/Desktop/data"
file = "2021_05_21__100834_calib1"

# Messdatei einlesen als "TimeSignal"-Objekt (numpy-array)
time_signal = import_signal_data_matlab(path, file)

# Nicht benötigte Dimensionen weglassen
time_signal2= TimeSignal(np.squeeze(np.array(time_signal)), f_samp=time_signal.f_samp)

# Signal von Sensor 0 (Sender) zu Sensor 2 (Empfänger) plotten
plt.plot(time_signal2[0,2,1000:])
plt.show()