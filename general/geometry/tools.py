import warnings
import numpy as np
from PAUDV.geometry import Orientation, Line


def get_coordinates_cross_section(point1, point2, length=None, steps=400, type='axial'):
    """
    Calculates line between point1 and point2; for type='lateral' it calculates the orthogonal line to the
    line between point1 and point2 in point2

    :param point1: Point object
    :param point2: Point object
    :param length:  length of section
    :param steps: number of discretization steps
    :param type: 'axial' or 'lateral' cross section
    :return:
    """
    firsthalf = point2 - point1
    # 2d hack
    if type == 'lateral':
        firsthalf = Orientation(np.pi / 2, 0, 0).transform(firsthalf)
        assert length is not None
        #warnings.warn('No length defined for lateral cross section!')

    if length is not None:
        firsthalf = firsthalf.normalize() * length / 2
        line = Line(point2 - firsthalf, point2 + firsthalf)
    else:
        line = Line(point2 - firsthalf, point2)

    points = line.discretize(steps)
    return points


def sample_crooked_line(a, b, weights, *, num, seed=None):
    """Create crooked line between two points.

    Creates `num` points that deviate randomly in the orthogonal direction
    from a line between `a` and `b`

    Parameters
    ----------
    a : (N,) sequence[float]
        Starting point of the line.
    b : (N,) sequence[float]
        Stoping point of the line.
    weights : (N,) sequence[float]
        Wheight used for each dimension when randomly shifting points on the line
        between `a` and `b`.
    num : int
        Number of points on the crooked line. The edges `a` and `b` are included.
    seed : int, optional
        The seed to use.

    Returns
    -------
    way_points : (`num`, N) numpy.ndarray
        An of generated way points of length. `a` and `b` are included.
    """
    if not len(a) == len(b) == len(weights):
        raise ValueError("a, b  and w must have the same length")

    zipped = list(zip(a, b))
    coords = [np.linspace(a, b, num, endpoint=True) for a, b in zipped]

    generator = np.random.RandomState(seed)
    widths = generator.random_sample(num)
    widths -= widths[1:-1].min()
    widths /= widths[1:-1].max()
    widths -= 0.5

    # Construct way points
    way_points = [space + widths * weight for space, weight in zip(coords, weights)]
    way_points = np.array([coords for coords in zip(*way_points)])

    # Replace edges with unshifted points
    way_points[0, :] = a
    way_points[-1, :] = b
    assert way_points.shape == (num, len(a))
    return way_points


def _define_circle(p0, p1, p2):
    # https://stackoverflow.com/a/20315063
    points = np.array([p0, p1, p2], dtype=float)
    p01 = np.linalg.norm(points[1] - points[0])
    p02 = np.linalg.norm(points[2] - points[0])
    p12 = np.linalg.norm(points[2] - points[1])
    # Half triangle circumfence
    s = (p02 + p12 + p01) / 2
    # Area of the triangle between 3 points (Heron's formula)
    area = np.sqrt(s * (s - p02) * (s - p12) * (s - p01))
    # Radius of the circle circumscribing the triangle
    radius = p02 * p12 * p01 / 4 / area
    # Center in barycentric coordinates
    b_center = np.array(
        [
            p12 ** 2 * (p02 ** 2 + p01 ** 2 - p12 ** 2),
            p02 ** 2 * (p12 ** 2 + p01 ** 2 - p02 ** 2),
            p01 ** 2 * (p12 ** 2 + p02 ** 2 - p01 ** 2),
        ]
    )
    # Transform into cartesian coordinates
    center = points.T.dot(b_center) / b_center.sum()

    return center, radius


def sample_arc(p0, p1, p2, num=50):
    """Sample part of a circle defined by three points.

    Parameters
    ----------
    p0 : (float, float)
        Starting point of the arc.
    p1 : (float, float)
        Point on the arc defining its curvature.
    p2 : (float, float)
        End point of the arc.
    num: int
        Number of sample points to generate on the arc. The edges are included.

    Returns
    -------
    way_points : ndarray
        An array of shape (`num`, 2) containing the sample points on the arc.

    Examples
    --------
    >>> sample_arc((1, 1), (2, 2), (2, 1), num=4)
    array([[1., 1.],
           [1., 2.],
           [2., 2.],
           [2., 1.]])
    >>> sample_arc((2, 1), (2, 2), (1, 1), num=4)
    array([[2., 1.],
           [2., 2.],
           [1., 2.],
           [1., 1.]])
    >>> sample_arc((2, 1), (1.5, 0.9), (1, 1), num=5)
    array([[2.        , 1.        ],
           [1.75495098, 0.92524512],
           [1.5       , 0.9       ],
           [1.24504902, 0.92524512],
           [1.        , 1.        ]])
    """
    points = np.array([p0, p1, p2], dtype=float)
    center, radius = _define_circle(*points)

    # Map points to unit circle
    normed_points = points.copy()
    normed_points[:, 0] -= center[0]
    normed_points[:, 1] -= center[1]
    normed_points /= radius

    # Compute angles of points on unit circle
    angles = np.arctan2(normed_points[:, 1], normed_points[:, 0])
    normed_angles = angles - angles[0]
    normed_angles[normed_angles < 0] += 2 * np.pi

    if normed_angles[1] < normed_angles[2]:
        # Arc turns counter-clockwise
        start_angle = angles[0]
        stop_angle = angles[0] + normed_angles[2]
    else:
        # Arc turns clockwise
        start_angle = angles[0]
        stop_angle = angles[0] - 2 * np.pi + normed_angles[2]

    assert abs(stop_angle - start_angle) <= 2 * np.pi

    way_angles = np.linspace(start_angle, stop_angle, num, endpoint=True)
    way_points = np.array(
        [
            np.cos(way_angles) * radius + center[0],
            np.sin(way_angles) * radius + center[1],
        ]
    )
    way_points = way_points.transpose()

    return way_points
