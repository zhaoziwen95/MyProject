import ctypes
import numbers
import zlib

import math
import numpy as np

#from PAUDV.geometry.grid_tools import get_grid_pitch


class Point(object):
    """
    geometric primitive of a point, defined with its cartesian coordinates

    >>> p1=Point(-0.1e-3,0.2,0)
    >>> p2=Point(-0.1e-3,0.2,0)
    >>> p3=Point(1, 0, 0)
    >>> p1 == p2
    True
    >>> p1 + p2 == p2 + p1
    True
    >>> (p1 - p2) == Point(0,0,0)
    True
    >>> p1 == p3
    False
    >>> p3.distance()==1
    True


    """

    tol = 1e-10  # one Angstroem

    def __init__(self, x=None, y=None, z=None, np_array: np.ndarray = None):
        """
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :param z: z coordinate of the point
        :param np_array: numpy array containing coordinates (if given x, y, z are ignored)
        """
        if np_array is None:
            for i in [x, y, z]:
                assert i is not None
            self.pos = np.array([x, y, z], dtype=np.float64)
        else:
            assert isinstance(np_array, np.ndarray)
            assert np_array.shape == (3,)
            self.pos = np_array

    @property
    def x(self):
        return self.pos[0]

    @x.setter
    def x(self, value):
        self.pos[0] = value

    @property
    def y(self):
        return self.pos[1]

    @y.setter
    def y(self, value):
        self.pos[1] = value

    @property
    def z(self):
        return self.pos[2]

    @z.setter
    def z(self, value):
        self.pos[2] = value

    def __getitem__(self, index):
        return self.pos[index]

    def __setitem__(self, index, item):
        self.pos[index] = item

    def __add__(self, other):
        if type(other) is tuple and len(other) == 3:
            other = Point(other[0], other[1], other[2])
        if type(other) is not Point:
            raise TypeError("Tried to add two Points, but one is not of type Point.")
        return Point(np_array=other.pos + self.pos)

    def distance(self, other=None):
        """
        Shortest distance between this Point and another Point. If other is None distance to Point(0, 0, 0) is returned.
        :param other: Point object or None
        """
        if other is None:
            return np.linalg.norm(self.pos)
        elif isinstance(other, Point):
            return np.linalg.norm(self.pos - other.pos)

    def normalize(self):
        """
        Normalizes the vector from origin to Point (return unit vector in direction of this Point).
        """
        if self.distance() == 0:
            return self
        else:
            return self / self.distance()

    def __sub__(self, other):
        return Point(np_array=self.pos - other.pos)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Point(np_array=self.pos * other)
        else:
            raise ValueError('No "mul" with element type {} defined!'.format(type(other).__name__))

    def __neg__(self):
        return Point(np_array=self.pos * -1)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return Point(np_array=self.pos / other)
        else:
            raise ValueError('No "div" with element type {} defined!'.format(type(other).__name__))

    def __str__(self):
        # return 'Pos({:0.3g} m, {:0.3g} m, {:0.3g} m)'.format(self.pos[0], self.pos[1], self.pos[2])
        return 'Pos({:0.3g} mm, {:0.3g} mm, {:0.3g} mm)'.format(self.pos[0] * 1000, self.pos[1] * 1000,
                                                                self.pos[2] * 1000)

    def __repr__(self):
        return 'Point({:0.6g}, {:0.6g}, {:0.6g})'.format(self.pos[0], self.pos[1], self.pos[2])

    def __eq__(self, other):
        return np.all(np.abs(self.pos - other.pos) < self.tol)

    def __lt__(self, other):
        for less, equal in zip(self.pos < other.pos, self.pos == other.pos):
            if less:
                return True
            else:
                if equal:
                    continue
                else:
                    return False
        return False

    def scalar(self, other: "Point"):
        """
        Scalar product with Point other
        :param other: Point to calculate scalar product for
        """
        if isinstance(other, Point):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise ValueError('No "scalar" with element type {} defined!'.format(type(other)))

    def __hash__(self):
        return ctypes.c_uint32(zlib.adler32(self.pos.tobytes())).value


class Line(object):
    """
    line in 3d space
    """

    def __init__(self, point1: Point, point2: Point):
        """
        :param point1: end point 1 of the line
        :param point2: end point 2 of the line
        """
        self.point1 = point1
        self.point2 = point2

    def length(self) -> float:
        """
        Length of this line.
        """
        return self.point1.distance(self.point2)

    def discretize(self, n_points):
        """
        returns a list in points that form a discretized line.
        :param n_points: number of points of the discretized version
        :return: list of points
        :rtype   list[Point]
        """
        assert isinstance(n_points, int)
        if n_points == 1:
            return [(self.point1 + self.point2) / 2]
        elif n_points > 1:
            diff = (self.point2 - self.point1) / (n_points - 1)
            return [self.point1 + diff * index for index in range(n_points)]
        else:
            raise ValueError("Cannot discretize line in {} points.".format(n_points))

    def get_pitch(self, n_points):
        """
        Returns the pitch vector for a discretization with n_points for the given line.
        :param n_points:
        :return:
        """
        if n_points == 1:
            return (self.point2 + self.point1) / 2
        if n_points > 1:
            return (self.point2 - self.point1) / (n_points - 1)
        else:
            raise ValueError("n_points need to be >= 1!")

    def transform(self, rot, shift=Point(0, 0, 0)):
        """
        Transforms this line using a rotation (described by Orientation matrix rot) and a subsequent translation
        (described by Point shift). Returns a new Line object that resembles this Line after the
        transformation.
        :param rot: orientation matrix for rotation
        :param shift: shift vector
        """
        assert isinstance(shift, Point)
        return Line(rot.transform(self.point1) + shift, rot.transform(self.point2) + shift)

    def distance(self, other):
        """
        Shortest distance between this Line and a Point.
        :param other: Point object
        """
        assert isinstance(other, Point)
        # http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        return np.linalg.norm(np.cross((self.point2 - self.point1).pos, (other - self.point1).pos)) / \
               (self.point1.distance(self.point2))

    def angle_between(self, other):
        """

        :param other:
        :return:

        >>> l1 = Line(Point(1,1,1), Point(1,1,0))
        >>> l2 = Line(Point(1,1,1), Point(0,1,1))
        >>> l1.angle_between(l2) == np.pi / 2
        True
        >>> l2.angle_between(l1) == np.pi / 2
        True
        >>> l3 = Line(Point(1,1,10), Point(1,1,11))
        >>> l3.angle_between(l1) == np.pi
        True

        """
        v1 = self.point2 - self.point1
        v2 = other.point2 - other.point1
        return np.arccos(np.inner(v1.pos, v2.pos))

    # def contains(self, other):
    #     assert isinstance(other, Point)
    #     #http://geomalgorithms.com/a02-_lines.html
    #     return np.cross((self.point2 - self.point1).pos, (other - self.point1).pos) == 0

    def __eq__(self, other):
        return self.point1 == other.point1 and self.point2 == other.point2

    def __str__(self):
        return "Line({} - {})".format(str(self.point1), str(self.point2))

    def __hash__(self):
        return hash(self.point1) + hash(self.point2)


class Plane(object):
    """
    planar surface in 3d space
    """

    def __init__(self, origin, point1, point2):
        """
        :param origin: point on the planar surface
        :param point1: point on the planar surface (vector from origin to point1 spans the surface, linear independent of vector from origin to point2)
        :param point2: point on the planar surface (vector from origin to point2 spans the surface, linear independent of vector from origin to point1)
        """
        self.origin = origin
        self.point1 = point1
        self.point2 = point2

        # calculate vector of the surface normal

    @property
    def normal_vect(self):
        return Point(
            np_array=np.cross(self.point1.pos - self.origin.pos, self.point2.pos - self.origin.pos)).normalize()

    # for compatibility, as normal_vect was a variable before
    @normal_vect.setter
    def normal_vect(self, val):
        pass

    @property
    def d(self):
        return self.normal_vect.x * self.origin.x + \
               self.normal_vect.y * self.origin.y + \
               self.normal_vect.z * self.origin.z

    # for compatibility, as d was a variable before
    @d.setter
    def d(self, val):
        pass

    def distance(self, other) -> float:
        """
        Shortest distance between this Plane and a Point resp. another Plane.
        :param other: Point or Plane object
        """
        if isinstance(other, Point):
            return abs(self.oriented_distance_to_point(other))
        elif isinstance(other, Plane):
            angle_between = np.arccos(np.inner(self.normal_vect.pos, other.normal_vect.pos))
            if angle_between > np.pi / 2:
                angle_between -= np.pi / 2
            elif angle_between < -np.pi / 2:
                angle_between += np.pi / 2
            if abs(angle_between) < 1e-6:
                return abs(self.d - other.d)
            else:
                return 0
        else:
            raise ValueError("Distance not defined for objects of type {} and {}".format(type(self).__name__,
                                                                                         type(other).__name__))

    def flip_orientation(self):
        """
        Flips the orientation of this Plane by flipping the direction of the normal vector and mirroring on of the
        given points (point1).
        """
        self.point1 = self.origin - (self.point1 - self.origin)

    def oriented_distance_to_point(self, point: Point) -> float:
        """
        Oriented shortest distance between this Plane and the Point point. Oriented distance means that the absolute
        value represents the shortest distance (see distance) and the sign represents the side of 3d space parted by the
        Plane point is in. A positive oriented distance means that point lies in the half space of the Plane in which
        the normal vector (normal_vect) points.
        :param point: Point to calculate distance to
        """
        return self.oriented_distance_to_points(point.x, point.y, point.z)

    def oriented_distance_to_points(self, x, y, z) -> float:
        """
        Oriented shortest distance between this Plane and the Point point. Oriented distance means that the absolute
        value represents the shortest distance (see distance) and the sign represents the side of 3d space parted by the
        Plane point is in. A positive oriented distance means that point lies in the half space of the Plane in which
        the normal vector (normal_vect) points.
        :param x,y,z: coordinates of points
        """
        return self.normal_vect.x * x + \
               self.normal_vect.y * y + \
               self.normal_vect.z * z - \
               self.d

    def perpendicular(self, point: Point):
        """
        Returns perpendicular point of the given point and this plane (point is mirrored at this Plane).
        :param point:
        """
        return point - self.normal_vect * self.oriented_distance_to_point(point)

    def __hash__(self):
        return hash(self.origin) + hash(self.point1) + hash(self.point2)

    def __setstate__(self, state):
        """
        compatibility handling for changed arguments
        :param state:
        :return:
        """
        if "vect1" in state and "vect2" in state:
            state["point1"] = state["vect1"]
            state["point2"] = state["vect2"]
            del state["vect1"]
            del state["vect2"]

        self.__dict__.update(state)


class Parallelogram(object):
    """
    surface which resembles a parallelogram area.
    """

    def __init__(self, point1: Point, point2: Point, point3: Point):
        """
        :param point1: start point of the parallelogram
        :param point2: point of vector from point 1 for first dimension of the parallelogram (point2 != point3)
        :param point3: point of vector from point 1 for second dimension of the parallelogram (point2 != point3)
        """
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3

    def discretize(self, n_points_dim1, n_points_dim2=None):
        """
        Returns a discretized version of a plane. Return either a
        list of points(if n_points_dim2 >=1) or a list of lines (n_points_dim2 == None).
        :param n_points_dim1:
        :param n_points_dim2:
        :return: list of points or list of lines
        """
        assert isinstance(n_points_dim1, int)

        if n_points_dim1 == 1:
            diff_dim1 = (self.point2 - self.point1) / 2
            lines = [Line(self.point1 + diff_dim1, self.point3 + diff_dim1)]
        elif n_points_dim1 > 1:
            diff = (self.point2 - self.point1) / (n_points_dim1 - 1)
            lines = [Line(self.point1 + diff * index, self.point3 + diff * index) for index in range(n_points_dim1)]
        else:
            raise ValueError("Cannot discretize parallelogram in {} points.".format((n_points_dim1, n_points_dim2)))

        if n_points_dim2 is None:
            return lines
        elif n_points_dim2 >= 1:
            assert isinstance(n_points_dim2, int)
            ret = []
            for line in lines:
                ret.extend(line.discretize(n_points_dim2))
            return ret
        else:
            raise ValueError("Cannot discretize parallelogram in {} points.".format((n_points_dim1, n_points_dim2)))

    def get_pitch(self, n_points_dim1, n_points_dim2=None):
        """
        Returns pitch vectors for the given discretization and the given plane.
        :param n_points_dim1:
        :param n_points_dim2:
        """
        if n_points_dim1 is None:
            raise ValueError("n_points_dim1 is None!")
        if n_points_dim2 is None:
            return Line(self.point1, self.point2).get_pitch(n_points_dim1)
        else:
            return (Line(self.point1, self.point2).get_pitch(n_points_dim1),
                    Line(self.point1, self.point3).get_pitch(n_points_dim2))

    def transform(self, rot, shift=Point(0, 0, 0)):
        """
        Transforms this parallelogram using a rotation (described by Orientation matrix rot) and a subsequent translation
        (described by Point shift). Returns a new Parallelogram object that resembles this Parallelogram after the
        transformation.
        :param rot: orientation matrix for rotation
        :param shift: shift vector
        """
        assert isinstance(shift, Point)
        return Parallelogram(rot.transform(self.point1) + shift, rot.transform(self.point2) + shift,
                             rot.transform(self.point3) + shift)

    def __str__(self):
        return 'Parallelogram({}, {}, {})'.format(str(self.point1), str(self.point2), str(self.point3))

    def __repr__(self):
        return 'Parallelogram({}, {}, {})'.format(repr(self.point1), repr(self.point2), repr(self.point3))

    @classmethod
    def centered_around(cls, center, vec1, vec2):
        return Parallelogram(center - vec1 - vec2, center + vec1 - vec2, center - vec1 + vec2)

    @property
    def plane(self) -> Plane:
        """
        Returns Plane in which this Parallelogram lies.
        """
        return Plane(self.point1, self.point2, self.point3)

    def discretize_to_grid(self, grid):
        """
        gives you an idx to a grid that represents the nearest grid points to the parallelogram
        :param grid:
        :return: indices
        """
        dx, dy, dz = get_grid_pitch(grid)

        diff1 = (self.point2 - self.point1)
        diff2 = (self.point3 - self.point1)
        n1 = 1
        n2 = 1
        fact = 2

        for idim, d in enumerate((dx, dy, dz)):
            if d != 0:
                n1 += (diff1.pos[idim] / d) ** 2
                n2 += (diff2.pos[idim] / d) ** 2

        points = self.discretize(int(np.sqrt(n1) * fact), int(np.sqrt(n2) * fact))

        # list of points to a nd array
        coords = np.empty((len(points), 3))
        for i, p in enumerate(points):
            coords[i, :] = p.pos

        indices = []
        for dim in range(3):
            slce = [0, 0, 0]
            slce[dim] = slice(None)
            slce = tuple(slce)
            dimsize = grid[dim][slce].size
            i = np.searchsorted(grid[dim][slce], coords[:, dim])
            i[i < 0] = 0
            i[i > dimsize - 1] = dimsize - 1
            indices.append(i)

        # mark the selected grid points; automatically deduplicate them
        idx = np.zeros_like(grid[0], dtype=np.bool)
        idx[tuple(indices)] = 1
        return idx

    def __eq__(self, other: 'Parallelogram'):
        return self.point1 == other.point1 and self.point2 == other.point2 and self.point3 == other.point3

    def __hash__(self):
        return hash(self.point1) + hash(self.point2) + hash(self.point3)


class Orientation(object):
    """
    orientation matrix with fixed axis (roll-pitch-yaw)
    1. phi around x-axis 2. chi around y-axis, 3. psi around z-axis
    """

    def __init__(self, psi, chi, phi):
        """
        :param psi: rotation angle around z-axis
        :param chi: rotation angle around y-axis
        :param phi: rotation angle around x-axis
        """
        self.psi = psi
        self.chi = chi
        self.phi = phi

        sin_psi = math.sin(self.psi)
        cos_psi = math.cos(self.psi)
        self.transformation_mat = np.array([[cos_psi, -sin_psi, 0],
                                            [sin_psi, cos_psi, 0],
                                            [0, 0, 1]])
        sin_chi = math.sin(self.chi)
        cos_chi = math.cos(self.chi)
        self.transformation_mat = np.dot(self.transformation_mat, np.array([[cos_chi, 0, sin_chi],
                                                                            [0, 1, 0],
                                                                            [-sin_chi, 0, cos_chi]]))
        sin_phi = math.sin(self.phi)
        cos_phi = math.cos(self.phi)
        self.transformation_mat = np.dot(self.transformation_mat, np.array([[1, 0, 0],
                                                                            [0, cos_phi, -sin_phi],
                                                                            [0, sin_phi, cos_phi]]))

    def transform(self, point: Point) -> Point:
        """
        Transform point using this orientation matrix.
        :param point: Point to transform using this orientation matrix.
        """
        return Point(np_array=np.dot(point.pos, self.transformation_mat))

    def __str__(self):
        return 'Ori({}, {}, {})'.format(self.psi, self.chi, self.phi)

    def __hash__(self):
        return hash(self.psi) + hash(self.chi) + hash(self.phi)
