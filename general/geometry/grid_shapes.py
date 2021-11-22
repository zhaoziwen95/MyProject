import functools
import logging

import matplotlib.path as mpltPath
import numpy as np
import skimage.draw
import scipy.interpolate as interp
import scipy.ndimage
import scipy.signal as s
#import skimage.draw

from PAUDV.geometry import Parallelogram, Point, Orientation, Plane
from PAUDV.geometry.grid_tools import find_nearest_gridpoint, get_pos_list_for_grid_mask, get_grid_pitch, \
    interpolate_from_3dgrid
from PAUDV.publish.plot import save_plot

logger = logging.getLogger(__name__)


def rect_grid_selector(grid, xlim=None, ylim=None, zlim=None):
    assert len(grid) == 3
    mask = np.ones_like(grid[0], dtype="bool")
    if xlim is not None:
        mask &= xlim[0] <= grid[0]
        mask &= xlim[1] >= grid[0]
    if ylim is not None:
        mask &= ylim[0] <= grid[1]
        mask &= ylim[1] >= grid[1]
    if zlim is not None:
        mask &= zlim[0] <= grid[2]
        mask &= zlim[1] >= grid[2]
    return mask


def cuboid_grid_selector(grid, center_pos=Point(0, 0, 0), ori=Orientation(0, 0, 0), width=None, height=None,
                         depth=None):
    """

    :param grid:
    :param center_pos: the center point of the cuboid
    :param ori:  the orientation
    :param width x
    :param height z
    :param depth: y
    :return: mask

    if the dimensions are not specified the cuboid extends into infinity

    >>> a = np.arange(-1, 1, 0.1)
    >>> grid = np.meshgrid(a, a, a, indexing="ij")
    >>> mask = cuboid_grid_selector(grid, width=1, height=1, depth=1, ori=Orientation(0, 0, 0))
    >>> np.all(mask[:, :, 5] == False)
    True
    >>> np.all(mask[6:15, 6:15, 10] == True)
    True


    #>>> mask[:,:,10]

    """
    assert len(grid) == 3
    mask = np.ones_like(grid[0], dtype="bool")
    planes = []
    if width is not None:
        planes.append((Point(-width / 2, 0, 0), Point(-width / 2, 1, 0), Point(-width / 2, 0, 1)))
        planes.append((Point(width / 2, 0, 0), Point(width / 2, 0, 1), Point(width / 2, 1, 0)))
    if height is not None:
        planes.append((Point(0, 0, -height / 2), Point(1, 0, -height / 2), Point(0, 1, -height / 2)))
        planes.append((Point(0, 0, height / 2), Point(0, 1, height / 2), Point(1, 0, height / 2)))
    if depth is not None:
        planes.append((Point(0, -depth / 2, 0), Point(0, -depth / 2, 1), Point(1, -depth / 2, 0)))
        planes.append((Point(0, depth / 2, 0), Point(1, depth / 2, 0), Point(0, depth / 2, 1)))

    for p1, p2, p3 in planes:
        p = Plane(center_pos + ori.transform(p1), center_pos + ori.transform(p2), center_pos + ori.transform(p3))
        partmask = p.oriented_distance_to_points(grid[0], grid[1], grid[2]) >= 0
        np.logical_and(mask, partmask, out=mask)
    return mask


def cuboid_grid_selector_2p(grid, start=Point(0, 0, 0), stop=Point(1, 0, 0), halfwidth_vec=Point(0, 1, 0),
                            halfheight_vec=Point(0, 0, 1)):
    """

    :param grid:
    :param start:
    :param stop:
    :param halfwidth_vec: relative vector spanning the width of the coboid
    :param halfheight_vec: relative vector spanning the height of the coboid
    :return:

    >>> a = np.arange(-1, 1, 0.1)
    >>> grid = np.meshgrid(a, a, a, indexing="ij")
    >>> start = Point(-0.5, 0, 0)
    >>> stop = Point(0.5, 0, 0)
    >>> halfwidth_vec = Point(0, 0.5, 0)
    >>> halfheight_vec = Point(0, 0, 0.5)
    >>> mask = cuboid_grid_selector_2p(grid,start=start, stop=stop, halfwidth_vec=halfwidth_vec, halfheight_vec=halfheight_vec)
    >>> np.all(mask[:, :, 5] == False)
    True
    >>> np.all(mask[6:15, 6:15, 10] == True)
    True
    >>> mask2 = cuboid_grid_selector_2p(grid, start=stop, stop=start, halfwidth_vec=halfwidth_vec, halfheight_vec=halfheight_vec)
    >>> np.all(mask == mask2)
    True

    """
    assert len(grid) == 3
    mask = np.ones_like(grid[0], dtype="bool")

    pair1 = [Plane(start, start + halfwidth_vec, start + halfheight_vec),
             Plane(stop, stop + halfwidth_vec, stop - halfheight_vec)]

    pair2 = [Plane(start + halfwidth_vec, stop + halfwidth_vec, start + halfwidth_vec + halfheight_vec),
             Plane(start - halfwidth_vec, stop - halfwidth_vec, start - halfwidth_vec - halfheight_vec)]

    pair3 = [Plane(start + halfheight_vec, stop + halfheight_vec, start + halfheight_vec - halfwidth_vec),
             Plane(start - halfheight_vec, stop - halfheight_vec, start - halfheight_vec + halfwidth_vec)]

    for pair in [pair1, pair2, pair3]:
        if pair[0].oriented_distance_to_point(pair[1].origin) < 0:  # let them face each other
            pair[0].flip_orientation()
        if pair[1].oriented_distance_to_point(pair[0].origin) < 0:  # let them face each other
            pair[1].flip_orientation()

        for p in pair:
            partmask = p.oriented_distance_to_points(grid[0], grid[1], grid[2]) >= 0
            np.logical_and(mask, partmask, out=mask)
    return mask


def spherical_grid_selector(grid, center, radius):
    return ((grid[0] - center[0]) ** 2 +
            (grid[1] - center[1]) ** 2 +
            (grid[2] - center[2]) ** 2) < radius ** 2


def ellipsoidal_grid_selector(grid, radii, center=(0, 0, 0)):
    """

    :param grid:
    :param radii: radius in x,y,z, could be np.inf
    :param center:
    :return: mask
    """
    dist = np.zeros_like(grid[0])
    for dim in range(3):
        dist += (grid[dim] - center[dim]) ** 2 / radii[dim] ** 2
    return dist <= 1


def wavy_ellipsoid_selector(grid, radii, center=(0, 0, 0), ampl=0.1, freq=1):
    """Create mask of an ellipsoid with wavy surface.

    Parameters
    ----------
    grid : tuple[np.array]
        3 equally shaped arrays representing the coordinates in each dimension
        (as returned by np.meshgrid).
    radii : tuple[int]
        The average radius of the ellipsoid in each dimension.
    center : tuple[int], optional
        Coordinates of the center of the ellipsoid.
    ampl : int or float
        A factor that controls the ampliute of the ellipsoids "waviness".
    freq : int or float
        The freqency in 1/rad of the ellipsoids "waviness".

    Returns
    -------
    mask : np.array
        A boolean array whose True values match the coordinate points inside
        the ellipsoid.

    Notes
    -----
    The waviness of the ellipsoids surface is a harmonic function of the
    inclination and azimuth relative to its center. Its angular velocity is
    calculated as 2 · pi · `freq`. Therefore this "constant" angular velocity
    is distorted for ellipsoids that aren't perfect spheres.
    """
    if not len(grid) == 3:
        raise ValueError("grid must have 3 elements")
    if not len(radii) == 3:
        raise ValueError("radii must have 3 elements")
    if not len(center) == 3:
        raise ValueError("center must have 3 elements")

    distance = np.zeros_like(grid[0], dtype="float64")
    for dim in range(len(grid)):
        distance += (grid[dim] - center[dim]) ** 2 / radii[dim] ** 2

    inclination = np.arccos((grid[2] - center[2]) / distance)
    azimuth = np.arctan2(grid[1] - center[1], grid[0] - center[0])

    angular_velocity = 2 * np.pi * freq
    distance += np.cos(inclination * angular_velocity) * ampl
    distance += np.cos(azimuth * angular_velocity) * ampl

    return distance <= 1


def cubic_tiled_spheres_grid_selector(grid, pitch, center=(0, 0, 0), diameter=None):
    """

    :param grid:
    :param pitch:
    :param center:
    :return:


    >>> grid = np.meshgrid(np.linspace(-1,1,300), np.linspace(-1,1,300), [0,],indexing="ij")
    >>> mask = cubic_tiled_spheres_grid_selector(grid, pitch=0.1, diameter=0.05)
    >>> if False:
    ...     from PAUDV.publish.plot import plot_image_from_grid
    ...     import matplotlib.pyplot as plt
    ...     fig = plt.figure(dpi=100)
    ...     ax = fig.add_subplot(1, 1, 1)
    ...     plot_image_from_grid(ax,grid[0][:,:,0],grid[1][:,:,0], mask[:,:,0])
    ...     plt.show()
    >>> np.all(mask[::-1,...] == mask)
    True
    >>> np.all(mask[...,::-1] == mask)
    True

    """
    if diameter is None:
        diameter = pitch
    return ((np.abs(np.mod(grid[0] - center[0] + pitch / 2, pitch)) - pitch / 2) ** 2 +
            (np.abs(np.mod(grid[1] - center[1] + pitch / 2, pitch)) - pitch / 2) ** 2 +
            (np.abs(np.mod(grid[2] - center[2] + pitch / 2, pitch)) - pitch / 2) ** 2
            ) <= (diameter / 2) ** 2


# def trapez_grid_selector(grid, trapez : Trapez = None):
#
#     assert trapez is not None
#
#     zlim_low = trapez.point1.z
#     zlim_high = trapez.point1.z + trapez.thick
#     a, b, c, d = trapez.coordinates()
#     m = (c.y-a.y)/(c.x-a.x)
#     assert len(grid) == 3
#     mask = np.ones_like(grid[0], dtype="bool")
#
#     mask &= trapez.point1.x <= grid[0]
#     mask &= trapez.point2.x >= grid[0]
#
#     mask &= m*grid[0]+a.y >= grid[1]
#     mask &= -m*grid[0]+b.y <= grid[1]
#
#     mask &= zlim_low <= grid[2]
#     mask &= zlim_high >= grid[2]
#
#     return mask

def thermal_expansion(mask, grid, delta_T, thermal_expansion_coeff, dims=(0, 1, 2), fixed_coords=Point(0, 0, 0)):
    """
        mask must be a ahape without holes

    :param mask:
    :param dim:
    :param grid:
    :param delta_T: temperature difference field (3d) in K
    :param thermal_expansion_coeff:
    :param fixed_coords: Point(x,y,z) the planes with the given (x,y,z) coordinates will be held in place
    :return:

          >>> grid = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,200), [0], indexing="ij")
          >>> mask = np.logical_and(np.abs(grid[0]) < 0.5, np.abs(grid[1]) < 0.5 )
          >>> delta_T = np.abs(grid[0])
          >>> mask = thermal_expansion(mask, grid, delta_T=delta_T, thermal_expansion_coeff=1,
          ...                          dims=(0,1,), fixed_coords=Point(0, 0, 0))
          >>> if False:
          ...   import matplotlib.pyplot as plt
          ...   from PAUDV.publish.plot import plot_image_from_grid
          ...
          ...   fig = plt.figure(dpi=100)
          ...   ax = fig.add_subplot(1, 1, 1)
          ...   ax.set_aspect("equal")
          ...   ax.set_xlabel('x / m')
          ...   ax.set_ylabel('y / m')
          ...   plot_image_from_grid(ax, grid[0][:,:,0], grid[1][:,:,0], mask[:,:,0])
          ...   save_plot(fig, "doctest_thermal_expansion", formats=("png",))
    """

    assert len(grid) == 3
    assert np.all(delta_T.shape == grid[0].shape)

    grid_pitch = get_grid_pitch(grid)
    ref_idx = find_nearest_gridpoint(grid, fixed_coords)

    for dim in dims:
        strain = grid_pitch[dim] * thermal_expansion_coeff * delta_T * mask

        # find index of the fixpoint line
        ref_idx_ = [slice(None), slice(None), slice(None)]
        ref_idx_[dim] = slice(ref_idx[dim], ref_idx[dim] + 1)

        # the movement of the grids coordinates
        delta = np.cumsum(strain, axis=dim)
        delta -= delta[tuple(ref_idx_)]

        new_grid = np.copy(grid)
        new_grid[dim] = grid[dim] - delta  # delta is subtracted, because interpolate_from_3dgrid is done inversely
        logger.info("thermal_expansion: in {}. dim:  {:6.4f} um".format(dim, 1e6 * np.max(np.abs(delta))))

        mask = interpolate_from_3dgrid(mask, grid, new_grid[0], new_grid[1], new_grid[2],
                                       interpolation_method="nearest",
                                       fill_value=0).astype(bool)
    return mask


def generic_rod_grid_selector(grid, boundary_func_y=None, boundary_func_z=None, prox: Point = None,
                              dist: Point = None, ):
    """
          Only for the x-y-plane!
          :param delta_T: temperature difference in K
          :param boundary_func_z:An arbitrary function that defines the outline of the waveguide
          :param boundary_func_y:An arbitrary function that defines the outline of the waveguide
          :param grid
          :param prox: start point of the waveguide. The point resides on the symmetry line of the waveguide
          :param dist: Defines the endpoint of the waveguide. As for now the point must reside on the x-axis.

                                  ............ <- Line defined by boundary_func
             y           ........:           |
             ^           |                   |
             |-->x       o---Symmetry line---x <- dist
                         |                   |
                        lp                   |
                                             ld

          >>> grid = np.meshgrid(np.linspace(-0.1,250e-3,100), np.linspace(-160e-3,160e-3,200), [0], indexing="ij")
          >>> func = lambda x: 0.2 * x + 64e-3
          >>> mask = generic_rod_grid_selector(grid, boundary_func_y=func, boundary_func_z=func, prox=Point(0,0,0), dist=Point(7,0,0))
          >>> print(bool(mask[28][60]))
          False
          >>> print(bool(mask[29][60]))
          True
          >>> print(bool(mask[99][29]))
          True
          >>> if False:
          ...   import matplotlib.pyplot as plt
          ...   from PAUDV.publish.plot import plot_image_from_grid
          ...   fig = plt.figure(dpi=100)
          ...   ax = fig.add_subplot(1, 1, 1)
          ...   ax.set_aspect("equal")
          ...   ax.set_xlabel('x / m')
          ...   ax.set_ylabel('y / m')
          ...   plot_image_from_grid(ax, grid[0][:,:,0], grid[1][:,:,0], mask[:,:,0])
          ...   plt.show()

          """
    assert len(grid) == 3
    mask = np.ones_like(grid[0], dtype="bool")

    assert prox.y == dist.y == 0 and prox.z == dist.z == 0, "ERROR: Symmetry line of waveguide must be in x-y-plane!"
    xlim_low = prox.x
    xlim_up = dist.x
    delta_x = 0
    assert xlim_low >= 0

    mask &= xlim_low <= grid[0]
    mask &= xlim_up + delta_x >= grid[0]

    if boundary_func_y is not None:
        y = boundary_func_y(grid[0])
        mask &= y >= grid[1]
        mask &= -y <= grid[1]

    if boundary_func_z is not None:
        z = boundary_func_z(grid[0])
        mask &= z >= grid[2]
        mask &= -z <= grid[2]

    return mask


def cone2d_grid_selector(grid, m1, m2, m3, m4):
    """
    This method returns a binary matrix which represents a 2d-frustum of a cone.
    Therefor the intersecting set between two parallelograms are used.
    :param m1: Start point of the first parallelogram
    :param m2: Start point of the second parallelogram
    :param m3: One point of the first parallelogram corresponding to the small radius
    :param m4: One point of the second parallelogram corresponding to the small radius
    :return:
    """
    return Parallelogram(m1, m2, m3).discretize_to_grid(grid) & Parallelogram(m2, m1, m4).discretize_to_grid(grid)


def random_particle_v_field_patch(grid, seed, density, x_lim, y_lim,
                                  v_field_pos_x, v_field_pos_y, v_field_x, v_field_y, dt, t_increment):
    """
    Random material patch for particles (size 1 pixel) with a given particle density whose movement is specified by
    the velocity grids v_field_x and v_field_y on the grid positions v_field_pos and a movement time dt and a time
    increment t_increment.
    :param grid: grid to create mask for
    :param seed: seed for the random number generator
    :param density: density or concentration of the distributed particles
    :param x_lim: list with lower and upper limit for the x coordinate of the area with particles
    :param y_lim: list with lower and upper limit for the y coordinate of the area with particles
    :param v_field_pos_x: x coordinate vector for the velocity fields v_field_x and v_field_y
    :param v_field_pos_y: y coordinate vector for the velocity fields v_field_x and v_field_y
    :param v_field_x: velocity field in x direction
    :param v_field_y: velocity field in y direction
    :param dt: time for the movement of the particles according to the velocity field
    :param t_increment: time increment
    :return: mask with distributed particles
    """

    # interpolate velocity fields for x and y coordinates
    func_v_field_x = interp.RectBivariateSpline(v_field_pos_x, v_field_pos_y, v_field_x)
    func_v_field_y = interp.RectBivariateSpline(v_field_pos_x, v_field_pos_y, v_field_y)

    # create the basic mask
    mask = np.random.RandomState(seed).random_sample(size=grid[0].shape) <= density

    mask &= np.logical_and(grid[0] >= x_lim[0], grid[0] <= x_lim[1])
    mask &= np.logical_and(grid[1] >= y_lim[0], grid[1] <= y_lim[1])
    particle_pos_list = get_pos_list_for_grid_mask(grid, mask)

    # move particles in the mask according to the velocity field
    n_increments = int(np.ceil(dt / t_increment))

    for i in range(n_increments):
        old_part_pos = np.copy(particle_pos_list)
        particle_pos_list[:, 0] += func_v_field_x(old_part_pos[:, 0], old_part_pos[:, 1], grid=False)
        particle_pos_list[:, 1] += func_v_field_y(old_part_pos[:, 0], old_part_pos[:, 1], grid=False)

    # create empty mask
    mask = np.zeros_like(grid[0])

    # create mask from position list again
    for part_idx in range(particle_pos_list.shape[0]):
        part_pos = particle_pos_list[part_idx]

        # handle particle moved outside of the scope (outside of x_lim, y_lim)
        # for now: just delete these particles
        grid_dx = abs(grid[0][1, 0, 0] - grid[0][0, 0, 0])
        grid_dy = abs(grid[1][0, 1, 0] - grid[1][0, 0, 0])
        if x_lim[0] - grid_dx / 2 > part_pos[0] or \
                x_lim[1] + grid_dx / 2 < part_pos[0] or \
                y_lim[0] - grid_dy / 2 > part_pos[1] or \
                y_lim[1] + grid_dy / 2 < part_pos[1]:
            continue

        part_grid_pos_idx = find_nearest_gridpoint(grid, [part_pos[0], part_pos[1], 0])
        mask[part_grid_pos_idx[0], part_grid_pos_idx[1], part_grid_pos_idx[2]] = 1

    return mask


def random_particle_patch_distribution(grid, seed, d_particle_list, material_concentration_list,
                                       overall_material_concentration, xlim, ylim):
    d_particle_list = np.array(d_particle_list)
    material_concentration_list = np.array(material_concentration_list)

    mask = np.squeeze(np.zeros_like(grid[0], dtype="bool"))
    overall_area = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])

    dx = grid[0][1, 0] - grid[0][0, 0]
    dy = grid[1][0, 1] - grid[1][0, 0]

    N_particles = 4 * material_concentration_list * overall_material_concentration * overall_area / (
            np.pi * d_particle_list ** 2)
    N_particles = np.array(N_particles, dtype="int")

    # create random vector
    r = np.random.RandomState(seed)
    rand_vec = r.random_sample((len(d_particle_list), np.sum(N_particles) * 50))

    for i in range(len(d_particle_list)):

        Num_used_particles = 0
        new_particle_mask = np.zeros_like(mask, dtype=float)
        # distribute particles until Num_particles is reached
        for j in range(int(len(rand_vec[i]) / 2)):

            if Num_used_particles == N_particles[i]:
                break

            x_coord = rand_vec[i, j] * (xlim[1] - xlim[0]) + xlim[0]
            y_coord = rand_vec[i, -j] * (ylim[1] - ylim[0]) + ylim[0]

            # distance_vector = np.sqrt((y_coord - np.array(y_coord_list))**2 + (x_coord - np.array(x_coord_list))**2)
            # if np.any(distance_vector < d_particle) or x_coord > xlim[1]-d_particle/2 or x_coord < xlim[0]+d_particle/2 or y_coord > ylim[1]-d_particle/2 or y_coord < ylim[0]+d_particle/2:
            #    continue
            idx_of_particle_in_grid = find_nearest_gridpoint(grid, Point(x_coord, y_coord, 0))
            new_particle_mask[idx_of_particle_in_grid[0], idx_of_particle_in_grid[1]] = True

            Num_used_particles += 1

        else:
            if Num_used_particles < N_particles[i]:
                logger.warning(
                    "Not enough space to distribute all particles! Maybe concentration is too high !?")

        # make size of particle
        sigma_x = np.float(d_particle_list[i] / (2 * dx))
        size = int(sigma_x * 1.5)
        weights_x = [0.0] * (2 * size + 1)
        weights_x[size] = 1  # kernel center
        for k in range(1, size + 1):
            tmp = np.exp(-0.5 * np.square(k / sigma_x))
            weights_x[size + k] = tmp
            weights_x[size - k] = tmp

        sigma_y = np.float(d_particle_list[i] / (2 * dy))
        size = int(sigma_y * 1.5)
        weights_y = [0.0] * (2 * size + 1)
        weights_y[size] = 1  # kernel center
        for k in range(1, size + 1):
            tmp = np.exp(-0.5 * np.square(k / sigma_y))
            weights_y[size + k] = tmp
            weights_y[size - k] = tmp

        for k in range(np.size(mask, 0)):
            new_particle_mask[k, :] = np.convolve(new_particle_mask[k, :], weights_x, mode='same')

        for k in range(np.size(mask, -1)):
            new_particle_mask[:, k] = np.convolve(new_particle_mask[:, k], weights_y, mode='same')

        threshold = np.exp(-0.5)  # threshold for sigma = x
        new_particle_mask = new_particle_mask > threshold
        mask += new_particle_mask

    return mask


def random_particle_patch(grid, seed, shift, d_particle, planar_random_material_concentration, xlim, ylim):
    """
    deprecated, use random_particles instead...
    :param grid: total grid of the environment
    :param seed: parameter which defines the distribution of scatterer (Note for reproducibility)
    :param d_particle: diameter of the scatterer
    :param planar_random_material_concentration: describes concentration of random material as: (area of rdm material)/(overall_area)
    :param x_l: lower limit of the distributed area in x direction
    :param x_u: upper limit of the distributed area in x direction
    :param y_l: lower limit of the distributed area in y direction
    :param y_u: upper limit of the distributed area in x direction
    :return: mask that includes an area with distributed binary numbers
    """
    mask = np.zeros_like(grid[0], dtype="bool")

    # calculate number of particels
    overall_area = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])
    Num_particles = int(planar_random_material_concentration * overall_area / (np.pi * d_particle ** 2 / 4))

    # create random vector
    r = np.random.RandomState(seed)
    rand_vec = r.random_sample((Num_particles * 50))

    x_coord_list = []
    y_coord_list = []
    Num_used_particles = 0
    # distribute particles until Num_particles is reached
    for i in range(int(len(rand_vec) / 2)):
        x_coord = rand_vec[i] * (xlim[1] - xlim[0]) + xlim[0]
        y_coord = rand_vec[-i] * (ylim[1] - ylim[0]) + ylim[0]
        # check whether there are other particles closer as particle diameter
        distance_vector = np.sqrt((y_coord - np.array(y_coord_list)) ** 2 + (x_coord - np.array(x_coord_list)) ** 2)
        if np.any(distance_vector < d_particle) or x_coord > xlim[1] - d_particle / 2 or x_coord < xlim[
            0] + d_particle / 2 or y_coord > ylim[1] - d_particle / 2 or y_coord < ylim[0] + d_particle / 2:
            continue
        idx_of_particle_in_grid = find_nearest_gridpoint(grid, Point(x_coord, y_coord, 0))
        mask[idx_of_particle_in_grid[0], idx_of_particle_in_grid[1]] = True
        Num_used_particles += 1
        if Num_used_particles == Num_particles:
            break
    else:
        if Num_used_particles < Num_particles:
            logger.warning(
                "Not enough space to distribute all particles! Maybe concentration is too high !?")

    dx = grid[0][1, 0] - grid[0][0, 0]
    dy = grid[1][0, 1] - grid[1][0, 0]

    if dx != dy:  # test grid spacing in x and y direction
        logger.warning(
            "random_material_patch assumes dx==dy of the grid, that is not the case! The particles may be nonspherical.")

    shift_x = shift[0] / dx
    shift_y = shift[1] / dy

    mask = np.roll(mask, np.int(np.round(shift_x)), axis=0)  # shift in x direction
    mask = np.roll(mask, np.int(np.round(shift_y)), axis=1)  # shift in y direction

    # prompt rounding errors
    if np.abs((np.round(shift_x) - shift_x) / shift_x) > 0.5:
        logger.warning("remarkable rounding errors during the shifting of the scatterer area")

    if np.abs((np.round(shift_y) - shift_y) / shift_y) > 0.5:
        logger.warning("remarkable rounding errors during the shifting of the scatterer area")

    # realise the particle size using the Gaussian distribution
    # convolution using a Gaussian kernel
    sigma = np.float(d_particle / (2 * dx))
    size = int(sigma * 1.5)
    weights = [0.0] * (2 * size + 1)
    weights[size] = 1  # kernel center
    for ii in range(1, size + 1):
        tmp = np.exp(-0.5 * np.square(ii / sigma))
        weights[size + ii] = tmp
        weights[size - ii] = tmp

    mask = mask[:, :, 0]
    mask_blurred = np.zeros(np.shape(mask))

    # convolution of the kernel and the mask along each axis (here 2D)
    for ii in range(np.size(mask, 0)):
        mask_blurred[ii, :] = np.convolve(mask[ii, :], weights, mode='same')

    for ii in range(np.size(mask, -1)):
        mask_blurred[:, ii] = np.convolve(mask_blurred[:, ii], weights, mode='same')

    threshold = np.exp(-0.5)  # threshold for sigma = x
    mask_new = mask_blurred > threshold

    return mask_new


def random_material_patch(grid, d_particle, density, shift=(0, 0), seed=42, x_l=None, x_u=None, y_l=None, y_u=None):
    """
    !!!! deprecated, use random_particles instead...

    This function defines a binary grid that contains an area with randomly distributed binary values
    :param grid: total grid of the environment
    :param seed: parameter which defines the distribution of scatterer (Note for reproducibility)
    :param shift: vector that defines the shifting of the scatterer area (shape(xshift,yshift))
    :param d_particle: diameter of the scatterer
    :param density: parameter to control the density of material
    :param x_l: lower limit of the distributed area in x direction
    :param x_u: upper limit of the distributed area in x direction
    :param y_l: lower limit of the distributed area in y direction
    :param y_u: upper limit of the distributed area in x direction
    :return: mask that includes an area with distributed binary numbers
    """

    # determine a rectangle including scatterer
    r = np.random.RandomState(seed)
    mask = r.random_sample(grid[0].shape) >= density
    dx = grid[0][1, 0] - grid[0][0, 0]
    dy = grid[1][0, 1] - grid[1][0, 0]

    if not np.isclose(dx, dy, rtol=1e-2):  # test grid spacing in x and y direction
        logger.warning(
            "random_material_patch assumes dx==dy of the grid, that is not the case! "
            "The particles may be nonspherical. (dx = {}, dy = {})".format(dx, dy))

    if x_l is not None:
        mask &= grid[0] > x_l
    if x_u is not None:
        mask &= grid[0] < x_u
    if y_l is not None:
        mask &= grid[1] > y_l
    if y_u is not None:
        mask &= grid[1] < y_u
    mask = np.float32(mask)

    shift_x = shift[0] / dx
    shift_y = shift[1] / dy

    mask = np.roll(mask, np.int(np.round(shift_x)), axis=0)  # shift in x direction
    mask = np.roll(mask, np.int(np.round(shift_y)), axis=1)  # shift in y direction

    # prompt rounding errors
    if shift_x != 0 and np.abs((np.round(shift_x) - shift_x) / shift_x) > 0.5:
        logger.warning("remarkable rounding errors during the shifting of the scatterer area")

    if shift_y != 0 and np.abs((np.round(shift_y) - shift_y) / shift_y) > 0.5:
        logger.warning("remarkable rounding errors during the shifting of the scatterer area")

    # realise the particle size using the Gaussian distribution
    # convolution using a Gaussian kernel
    sigma = np.float(d_particle / (2 * dx))
    size = int(sigma * 1.5)
    weights = [0.0] * (2 * size + 1)
    weights[size] = 1  # kernel center
    for ii in range(1, size + 1):
        tmp = np.exp(-0.5 * np.square(ii / sigma))
        weights[size + ii] = tmp
        weights[size - ii] = tmp

    mask = mask[:, :, 0]
    mask_blurred = np.zeros(np.shape(mask))

    # convolution of the kernel and the mask along each axis (here 2D)
    for ii in range(np.size(mask, 0)):
        mask_blurred[ii, :] = np.convolve(mask[ii, :], weights, mode='same')

    for ii in range(np.size(mask, -1)):
        mask_blurred[:, ii] = np.convolve(mask_blurred[:, ii], weights, mode='same')

    threshold = np.exp(-0.5)  # threshold for sigma = x
    mask_new = mask_blurred > threshold

    return mask_new


def random_particles(grid, d_particle=(0, 0, 0), density=0.01, shift=(0, 0, 0), seed=42):
    """
    This function defines a binary grid that contains an area with randomly distributed binary values
    :param grid: a 3d grid
    :param seed: parameter which defines the distribution of scatterer (for reproducibility)
    :param shift: vector that defines the shifting of the scatterer area in meter (xshift, yshift, zshift)
    :param d_particle: diameter of the scatterer (x,z,y) in meter
    :param density: parameter to control the density of the particles (probability in the range of 0-1)
    :return: mask

    >>> grid = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 200), [0], indexing="ij")
    >>> d_particle=(0.2, 0.3, np.inf)
    >>> seed=42
    >>> mask = random_particles(grid, d_particle=(0.2, 0.3, np.inf), density=0.001, seed=seed)
    >>> mask_dots = random_particles(grid, density=0.001, seed=seed)
    >>> np.all(np.logical_or(mask, mask_dots) == mask)
    True
    >>> if False:
    ...   import matplotlib.pyplot as plt
    ...   from PAUDV.publish.plot import plot_image_from_grid
    ...
    ...   fig = plt.figure(dpi=100)
    ...   ax = fig.add_subplot(1, 1, 1)
    ...   ax.set_aspect("equal")
    ...   ax.set_xlabel('x / m')
    ...   ax.set_ylabel('y / m')
    ...   plot_image_from_grid(ax, grid[0][:, :, 0], grid[1][:,:,0], mask[:,:,0])
    ...   save_plot(fig, "doctest_random_particles", formats=("png",))
    """

    # determine a rectangle including scatterer
    r = np.random.RandomState(seed)
    mask = r.random_sample(grid[0].shape) >= 1.0 - density

    deltagrid = get_grid_pitch(grid)
    kernel_pix = [1, 1, 1]
    vector = [[0], [0], [0]]

    for dim in range(3):
        if deltagrid[dim] > 0:
            shift_pix = shift[dim] / deltagrid[dim]
            shift_pix_int = np.int(np.round(shift_pix))
            if shift_pix_int:
                mask = np.roll(mask, shift_pix_int, axis=dim)  # shift in direction dim

            # prompt rounding errors
            if shift_pix != 0 and np.abs((shift_pix_int - shift_pix) / shift_pix) > 0.5:
                logger.warning("significant rounding errors during the shifting of the scatterer area")

        if deltagrid[dim] > 0:
            kernel_pix[dim] = int(np.round(d_particle[dim] / deltagrid[dim]))
            kernel_pix[dim] = min(kernel_pix[dim], grid[0].shape[dim])

        if kernel_pix[dim] > 2:
            vector[dim] = np.linspace(-1, 1, kernel_pix[dim], endpoint=True)

    kern_grid = np.meshgrid(*vector, indexing="ij")
    footprint = (kern_grid[0] ** 2 + kern_grid[1] ** 2 + kern_grid[2] ** 2) <= 1
    return scipy.ndimage.maximum_filter(mask, footprint=footprint, mode="reflect")


def circular_ring(grid, r_inner, r_outer, origin=(0, 0)):
    dists = np.sqrt((grid[0] - origin[0]) ** 2 + (grid[1] - origin[1]) ** 2)
    return np.logical_and(dists <= r_outer, dists >= r_inner)


def circular_ring_with_chamfer(grid, r_inner, r_outer, origin, depth_chamfer, angle_to_chamfer=3 / 2 * np.pi):
    mask = circular_ring(grid, r_inner, r_outer, origin)
    # set region of chamfer to zero
    # TODO: make chamfer at different angles
    mask[grid[1] < origin[0] - (r_outer - depth_chamfer)] = 0
    return mask


# TODO Lars, deprecate in favor of polygon_selector
def polygon_2D(grid, points):
    """
    This function defines a binary grid that contains a 2D polynom
    :param grid: a 3d grid
    :param points: List of points of the polynom, example: [(x0,y0),(x1,y1),(x2,y2)]
    :return: mask
    """
    mask = np.ones_like(grid[0], dtype="bool")
    path = mpltPath.Path(points)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            mask[x][y][0] = path.contains_point((grid[0][x][y][0], grid[1][x][y][0]))

    return mask


def polygon_selector(grid, polygon):
    """Draw a polygon inside a grid.

    Parameters
    ----------
    grid : (ndarray. ndarray, ndarray)
        Coordinate grid for 3 dimensions as returned by
        ``np.meshgrid(..., indexing="ij")``. The polygon is drawn in the first
        two.
    polygon : (N, 2) ndarray
        X and Y position of N points that make up the polygon.

    Returns
    -------
    mask : ndarray
        A boolean array with the same shape as the arrays in `grid` indicating the
        polygon.
    """
    if not polygon.shape[1] == 2:
        raise ValueError("polygon points had 3 dimension, only first 2 are supported")

    shape = grid[0].shape[:2]

    x, y = polygon.T.copy()
    # Shift points in case grid starts with negative coordinates,
    # draw.polygon assumes origin (0, 0)
    x -= grid[0].min()
    y -= grid[1].min()
    # and scale physical unit to index
    x *= shape[0] / (grid[0].max() - grid[0].min())
    y *= shape[1] / (grid[1].max() - grid[1].min())

    xx, yy = skimage.draw.polygon(x, y, shape)
    mask = np.zeros_like(grid[0], dtype=bool)
    mask[xx, yy, :] = 1
    return mask


def bumpy_interface_rect(grid, xlim, ylim, zlim=None, bumpyness_amplitude=2e-3, bumpyness_lateralsize=2e-3,
                         bumpy_side=None, seed=None):
    """

    Args:
        grid: arbitrary grid
        xlim: (xmin, xmax)
        ylim: (ymin, ymax)
        zlim: (zmin,zmax) --> not really implemented yet, only use for 2D!
        bumpyness_amplitude: amplitude of bumps from the respective lim value
        bumpyness_lateralsize: something like the pitch of the bumbs in lateral direction
        bumpy_side: Point which defines the side of the rect with bump, e.g. Point(1,0,0) right side with bumps,
        Point(0,-1,0) lower side of the rectangle with bumps
        seed: if seed is None


    Returns: binary mask

    """
    assert len(grid) == 3
    assert type(bumpy_side) is Point

    # make rectangle, if no bumpy side is given!
    mask = rect_grid_selector(grid, xlim, ylim, zlim)

    # apply bumpy interface
    if bumpy_side is None:
        return mask
    elif abs(bumpy_side.x) == 1:
        n_bumps = (ylim[1] - ylim[0]) / bumpyness_lateralsize
        num_gridpoints = np.shape(grid[0])[1]
    elif abs(bumpy_side.y) == 1:
        n_bumps = (xlim[1] - xlim[0]) / bumpyness_lateralsize
        num_gridpoints = np.shape(grid[0])[0]

    r = np.random.RandomState(seed)
    rdm_vector = (r.random_sample((n_bumps,)) - 0.5) * 2 * bumpyness_amplitude
    bumpy_vector = s.resample(rdm_vector, num_gridpoints)

    # 4 cases for one of 4 sides with bumps...
    if bumpy_side.x == 1:
        for linenumber in range(np.shape(grid[0])[1]):
            if grid[1][0, linenumber] < ylim[1] and grid[1][0, linenumber] > ylim[0]:
                mask[:, linenumber] = (xlim[1] + bumpy_vector[linenumber] >= grid[0][:, 0]) & (xlim[0] < grid[0][:, 0])
    elif bumpy_side.x == -1:
        for linenumber in range(np.shape(grid[0])[1]):
            if grid[1][0, linenumber] < ylim[1] and grid[1][0, linenumber] > ylim[0]:
                mask[:, linenumber] = (xlim[0] - bumpy_vector[linenumber] <= grid[0][:, 0]) & (xlim[1] > grid[0][:, 0])
    elif bumpy_side.y == 1:
        for linenumber in range(np.shape(grid[0])[0]):
            if grid[0][linenumber, 0] < xlim[1] and grid[0][linenumber, 0] > xlim[0]:
                mask[linenumber, :] = (ylim[1] + bumpy_vector[linenumber] >= grid[1][0, :]) & (ylim[0] < grid[1][0, :])
    elif bumpy_side.y == -1:
        for linenumber in range(np.shape(grid[0])[0]):
            if grid[0][linenumber, 0] < xlim[1] and grid[0][linenumber, 0] > xlim[0]:
                mask[linenumber, :] = (ylim[0] - bumpy_vector[linenumber] <= grid[1][0, :]) & (ylim[1] > grid[1][0, :])

    return mask


def logical_op_grid_selector(grid, op=None, func0=None, func1=None, func2=None, **kwargs):
    """
    this function calculates 
    :param grid: a 3d grid
    :param func0: grid selector function
    :param func1: grid selector function
    :param func2: grid selector function
    :param funcN: grid selector function
    :return: mask


    >>> grid = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 200), [0], indexing="ij")
    >>> func0= functools.partial(rect_grid_selector, xlim=[0,1])
    >>> func1= functools.partial(rect_grid_selector, xlim=[0.1, 0,6], ylim=[0.1, 0.6])
    >>> mask1 = func1(grid)
    >>> mask0 = func0(grid)
    >>> mask = logical_op_grid_selector(grid, op=np.logical_and, func0=func0, func1=func1)
    >>> np.all(np.logical_and(mask0, mask1) == mask)
    True
    >>> maskmultiple = logical_op_grid_selector(grid, op=np.logical_and, func0=func0, func1=func1, func2=func1,
    ...                                             func3=func1, func4=func0)
    >>> np.all(maskmultiple == mask)
    True

    """
    mask = op(func0(grid), func1(grid))
    for f in [func2, ] + list(kwargs.values()):
        if f is not None:
            mask = op(mask, f(grid))
    return mask


def logical_and_grid_selector(grid, func0=None, func1=None, func2=None, **kwargs):
    """ see logical_op_grid_selector() """
    return logical_op_grid_selector(grid, op=np.logical_and, func0=func0, func1=func1, func2=func2, **kwargs)


def logical_or_grid_selector(grid, func0=None, func1=None, func2=None, **kwargs):
    """ see logical_op_grid_selector() """
    return logical_op_grid_selector(grid, op=np.logical_or, func0=func0, func1=func1, func2=func2, **kwargs)


def logical_xor_grid_selector(grid, func0=None, func1=None, func2=None, **kwargs):
    """ see logical_op_grid_selector() """
    return logical_op_grid_selector(grid, op=np.logical_xor, func0=func0, func1=func1, func2=func2, **kwargs)
