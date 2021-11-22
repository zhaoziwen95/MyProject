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
begin_time=1000
end_time=12000

ax1=plt.subplot(4,4,1)
plt.plot(time_signal2[0,0,begin_time:end_time])
ax2=plt.subplot(4,4,2)
plt.plot(time_signal2[0,1,begin_time:end_time])
ax1=plt.subplot(4,4,3)
plt.plot(time_signal2[0,2,begin_time:end_time])
ax2=plt.subplot(4,4,4)
plt.plot(time_signal2[0,3,begin_time:end_time])
ax1=plt.subplot(4,4,5)
plt.plot(time_signal2[1,0,begin_time:end_time])
ax2=plt.subplot(4,4,6)
plt.plot(time_signal2[1,1,begin_time:end_time])
ax1=plt.subplot(4,4,7)
plt.plot(time_signal2[1,2,begin_time:end_time])
ax2=plt.subplot(4,4,8)
plt.plot(time_signal2[1,3,begin_time:end_time])
ax1=plt.subplot(4,4,9)
plt.plot(time_signal2[2,0,begin_time:end_time])
ax2=plt.subplot(4,4,10)
plt.plot(time_signal2[2,1,begin_time:end_time])
ax1=plt.subplot(4,4,11)
plt.plot(time_signal2[2,2,begin_time:end_time])
ax2=plt.subplot(4,4,12)
plt.plot(time_signal2[2,3,begin_time:end_time])
ax1=plt.subplot(4,4,13)
plt.plot(time_signal2[3,0,begin_time:end_time])
ax2=plt.subplot(4,4,14)
plt.plot(time_signal2[3,1,begin_time:end_time])
ax1=plt.subplot(4,4,15)
plt.plot(time_signal2[3,2,begin_time:end_time])
ax2=plt.subplot(4,4,16)
plt.plot(time_signal2[3,3,begin_time:end_time])
plt.show()