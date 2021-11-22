import numpy as np
from scipy import special as sp

from PAUDV.tools.wave_propagation_modeling.math import hankel_first_kind_first_derivative, spherical_hankel_first_kind, \
    get_value_of_legendre_polynomial



def scattering_coefficient_single_particle(order, freqs, density_bulk, density_particle, cp_1, cp_2, cs_2, r_particle):
    '''
    compute scattering coefficient for single particle for inviscid fluid and elastic sphere particle
    from: Brill, D., and G. Gaunaurd. "Resonance theory of elastic waves ultrasonically scattered from an elastic sphere."
    The Journal of the Acoustical Society of America 81.1 (1987): 1-21.

    Args:
        order: scattering order (o --> monopole, 1 --> dipole ...)
        freqs: frequencies to compute the scattering coefficient for
        density_bulk: density of bulk/continous phase
        density_particle: density of particles
        cp_1: long. soundspeed of bulk
        cp_2: long. soundspeed of particle
        cs_2: shear soundspeed of particle
        r_particle: radius of particle

    Returns: A_n: vector of scattering coefficients of order n for all freqs

    '''

    A_n = []
    for f in freqs:

        k_d1 = 2*np.pi*f / cp_1
        k_d2 = 2*np.pi*f / cp_2
        k_s2 = 2*np.pi*f / cs_2

        x_d1 = k_d1 * r_particle   # x_d in medium 1
        x_d2 = k_d2 * r_particle
        x_s2 = k_s2 * r_particle

        # calculate coefficients for inviscid fluid and elastic spheres
        d_11 = x_d1 * hankel_first_kind_first_derivative(order, x_d1)
        d_31 = -spherical_hankel_first_kind(order, x_d1)
        d_41 = 0

        d_13 = -x_d2 * sp.spherical_jn(order, x_d2, derivative=True)
        d_33 = -(density_particle/density_bulk) * 1/x_s2**2 * ((2*order*(order + 1 )-x_s2**2)*sp.spherical_jn(order,x_d2) - 4*x_d2*sp.spherical_jn(order, x_d2, derivative=True))
        d_43 = -(density_particle/density_bulk) * 1/x_s2**2 * (x_d2*sp.spherical_jn(order, x_d2, derivative=True) - sp.spherical_jn(order, x_d2))

        d_14 = -order * (order + 1) * sp.spherical_jn(order, x_s2)
        d_34 = -(density_particle/density_bulk) * (2*order*(order+1)/x_s2**2)*(x_s2 * sp.spherical_jn(order, x_s2, derivative=True) - sp.spherical_jn(order, x_s2))
        d_44 = -(density_particle/density_bulk) * 1/x_s2**2 * ((order * (order +1) - x_s2**2/2 - 1)*sp.spherical_jn(order, x_s2) - x_s2*sp.spherical_jn(order, x_s2, derivative=True))


        delta_134 = np.array([[d_11, d_13, d_14],
                                [d_31, d_33, d_34],
                                [d_41, d_43, d_44]])
        det_delta_134 = np.linalg.det(delta_134)
        delta_1star34 = np.array([[np.conj(d_11), d_13, d_14],
                                    [np.conj(d_31), d_33, d_34],
                                    [np.conj(d_41), d_43, d_44]])
        det_delta_1star34 = np.linalg.det(delta_1star34)
        A_n.append(0.5 * (1 + det_delta_134/det_delta_1star34))

    return np.array(A_n)


def scattering_amplitude(freqs, density_bulk, density_particle, cp_1, cp_2, cs_2, r_particle, angles = (0,), max_order = 10):
    '''
    computes the scattering amplitude up to the sc attering order of max_order for all freqs and angles
    Args:
        freqs: frequency vector
        density_bulk: density of bulk material/continous phase
        density_particle: density of particles
        cp_1: long. soundspeed of bulk
        cp_2: long. soundspeed of particle
        cs_2: shear soundspeed of particle
        r_particle: radius of particle
        angles: angles to compute the scattering amplitude for
        max_order: max order of scattering, scattering amplitude takes all orders up to max_order into account

    Returns: nd.array with 3 Dimensions: scattering_amplitude[order, frequency, angle]

    '''

    scattering_amplitude = np.zeros((max_order, len(freqs), len(angles)), dtype=complex)   # angles in radiant
    k = 2*np.pi*freqs / cp_1
    for i, angle in enumerate(angles):
        for order in range(max_order):
            scattering_amplitude[order, :, i] = 1/(1j*k)*(2*order + 1) * scattering_coefficient_single_particle(order, freqs, density_bulk, density_particle, cp_1, cp_2, cs_2, r_particle) * get_value_of_legendre_polynomial(order, np.cos(angle))  # angle in degree!!!

    return scattering_amplitude

def effective_medium_soundspeed_attenuation_for_suspension(freqs, density_bulk, density_particle, cp_1, cp_2, cs_2, r_particle, volume_fraction, max_order = 10):

    '''
    calculation of soundspeed and attenuation for one scatterer size with the effective medium model taken from:
    Peters, François, and Luc Petit. "Propagation of ultrasound waves in concentrated suspensions."
    Acta Acustica united with Acustica 86.5 (2000): 838-846.
    Valid for suspensions of solid particles in fluid
    Args:
        freqs: frequency vector
        c_bulk: soundspeed of pressure wave in bulk material/continous phase
        density_bulk: density of bulk material/continous phase
        density_particle: density of particles
        cp_1: long. soundspeed of bulk
        cp_2: long. soundspeed of particle
        cs_2: shear soundspeed of particle
        r_particle: radius of particle
        volume_fraction: volume fraction of particles in the suspension
        angles: angles to compute the scattering amplitude for
        max_order: max order of scattering, scattering amplitude takes all orders up to max_order into account

    Returns:

    '''
    k = 2*np.pi*freqs / cp_1
    scattering_amps = scattering_amplitude(freqs, density_bulk, density_particle, cp_1, cp_2, cs_2, r_particle, angles = (0, np.pi), max_order = max_order)
    V_particle = np.pi * (2*r_particle)**3 / 6
    n_0 = volume_fraction / V_particle
    kappa = k * np.sqrt((1 + (2 * np.pi * n_0 * np.sum(scattering_amps[:,:,0],axis=0)) / k ** 2) ** 2 - ((2 * np.pi * n_0 * np.sum(scattering_amps[:,:,1],axis=0)) / k ** 2) ** 2)

    soundspeed = 2*np.pi*freqs/np.real(kappa)
    attenuation = np.imag(kappa)

    return soundspeed, attenuation

def effective_medium_soundspeed_attenuation_for_suspension_with_particle_distribution(freqs, density_bulk, density_particle, cp_1, cp_2, cs_2, r_particle, volume_fraction, distribution, max_order = 10):
    k = 2*np.pi*freqs / cp_1
    helper_sum = np.sum(distribution/np.amax(distribution) * 4/3 * np.pi * r_particle**3)
    n_0 = distribution/np.amax(distribution) / helper_sum * volume_fraction      # particle_density calculated from the particle distribution as number of particles per volume

    sum_scat_amp_0 = 0
    sum_scat_amp_pi = 0
    for i in range(len(r_particle)):
        print("calculation for particle size: " + str(r_particle[i]) + "m")
        scattering_amps = scattering_amplitude(freqs, density_bulk, density_particle, cp_1, cp_2, cs_2, r_particle[i], angles = (0, np.pi), max_order = max_order)
        sum_scat_amp_0 += n_0[i] * np.nansum(scattering_amps[:,:,0],axis=0)
        sum_scat_amp_pi += n_0[i] * np.nansum(scattering_amps[:,:,1],axis=0)
    kappa = k * np.sqrt((1 + (2 * np.pi *  sum_scat_amp_0) / k ** 2) ** 2 - ((2 * np.pi * sum_scat_amp_pi) / k ** 2) ** 2)

    soundspeed = 2*np.pi*freqs/np.real(kappa)
    attenuation = np.imag(kappa)

    return soundspeed, attenuation

def get_extinction_length(attenuation):
    return 1/(2*attenuation)
