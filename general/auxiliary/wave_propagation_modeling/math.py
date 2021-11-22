from scipy import special as sp


def spherical_hankel_first_kind(order, value):
    return sp.spherical_jn(order, value) + 1j * sp.spherical_yn(order, value)

def get_value_of_legendre_polynomial(order, value):
    return sp.lpn(order, value)[0][order]

def hankel_first_kind_first_derivative(order, value):
    hankel_gradient = sp.spherical_jn(order, value, derivative=True) + 1j * sp.spherical_yn(order, value, derivative=True)
    return hankel_gradient