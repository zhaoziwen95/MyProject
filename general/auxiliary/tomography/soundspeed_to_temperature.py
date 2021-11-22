import numpy as np


def sound_speed_of_water(temperature):
    # temperature in kelvin
    T = temperature - 273
    c = 1.402385e3 + 5.038813*T - 5.799136e-2*T**2 + \
        3.287156e-4*T**3 - 1.398845e-6*T**4 + 2.787860e-9*T**5
    return c


def temperature_from_soundspeed(sound_speed):
    a1 = 5.799136e-2
    a2 = 5.038813
    a3 = 1.402385e3
    T = -(a2/(2*a1)) + np.sqrt((a2/a1)**2/4 - (a3 - sound_speed)/a1)
    return T

def temperature_from_soundspeed_linear_approx(sound_speed):
    return 0.541*sound_speed - 784.562