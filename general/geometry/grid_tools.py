import itertools
from math import sqrt

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator


def get_grid_pitch(grid):
    """
    Extracts the respective grid steps from a 3d grid.

    :param grid:
    :return: dx,dy,dz


     >>> g = np.meshgrid(np.arange(0, 10, .5), np.arange(0, 10, 2), np.arange(0, 10, 5), indexing="ij")
     >>> get_grid_pitch(g)
     (0.5, 2, 5)

    """
    assert len(grid) == 3
    assert grid[0].ndim == 3
    if grid[0].shape[0] > 1:
        dx = grid[0][1, 0, 0] - grid[0][0, 0, 0]
    else:
        dx = 0
    if grid[1].shape[1] > 1:
        dy = grid[1][0, 1, 0] - grid[1][0, 0, 0]
    else:
        dy = 0
    if grid[2].shape[2] > 1:
        dz = grid[2][0, 0, 1] - grid[2][0, 0, 0]
    else:
        dz = 0

    return dx, dy, dz


def get_mask_area(grid, mask):  # 2d only
    assert len(mask.shape) == 2
    count = np.count_nonzero(mask)
    deltax = np.diff(grid[0][0:2, 0, 0])
    deltay = np.diff(grid[1][0, 0:2, 0])
    areael = deltax * deltay
    return areael * count


def find_level_around(grid, v, pos, level):
    # only 2d for now

    idx = find_nearest_gridpoint(grid, pos)

    mask = np.zeros_like(grid[0], dtype=np.int8)
    lastmask = np.copy(mask)

    mask[idx] = 1

    while True:
        # extend the mask
        mask = scipy.ndimage.filters.convolve(mask, np.ones((6, 6, 1), dtype=np.int8), mode='nearest')
        mask[mask > 0] = 1

        # crop back if v is lower than level
        mask[v < level] = 0

        # no changes? finished!
        if np.all(lastmask == mask):
            break
        else:
            lastmask = np.copy(mask)

        if False:
            from matplotlib import pyplot as plt
            plt.imshow(mask[:, :, 0])
            plt.show()

    return mask


def find_nearest_gridpoint(grid, point):
    """
    Returns grid point indices for a given point in a given grid.
    Correct indexing assumed (e.g. indexing="ij" for numpy.meshgrid).
    :param grid:
    :param point: 3 coordinate vector (Point or single numpy array)
    """
    dx, dy, dz = get_grid_pitch(grid)

    if dx == 0:
        idx_x = 0
    else:
        idx_x = np.int(np.round((point[0] - grid[0][0, 0, 0]) / dx))
        if idx_x >= grid[0].shape[0]:
            idx_x = grid[0].shape[0] - 1
        elif idx_x < 0:
            idx_x = 0

    if dy == 0:
        idx_y = 0
    else:
        idx_y = np.int(np.round((point[1] - grid[1][0, 0, 0]) / dy))
        if idx_y >= grid[1].shape[1]:
            idx_y = grid[1].shape[1] - 1
        elif idx_y < 0:
            idx_y = 0

    if dz == 0:
        idx_z = 0
    else:
        idx_z = np.int(np.round((point[2] - grid[2][0, 0, 0]) / dz))
        if idx_z >= grid[2].shape[2]:
            idx_z = grid[2].shape[2] - 1
        elif idx_z < 0:
            idx_z = 0

    return idx_x, idx_y, idx_z


# def find_nearest_gridpoint(grid, point):
#     d = (grid[0] - point.x) ** 2 + (grid[1] - point.y) ** 2 + (grid[2] - point.z) ** 2
#     idx = np.where(d == d.min())
#     return idx


def interpolate_from_3dgrid(data, grid, xi, yi, zi, interpolation_method="nearest", **kwargs):
    """
    
    :param data: 
    :param grid:  
    :param xi: vector of x coordinates
    :param yi: 
    :param zi: 
    :param interpolation_method: either "nearest" or "linear"
    :return: the interpolated data
    """
    assert len(grid) == 3

    # UGLY work around for https://github.com/scipy/scipy/issues/5890
    for dim in range(len(grid)):
        if grid[dim].shape[dim] == 1 and interpolation_method == "linear":
            # increase the length for this dimension to 2
            reps = np.ones_like(grid[dim].shape)
            reps[dim] = 2
            data = np.tile(data, reps)
            grid = [np.tile(g, reps) for g in grid]
            s = [slice(None) for d in range(len(grid))]
            s[dim] = 1
            grid[dim][s] += 1

    interp = RegularGridInterpolator((grid[0][:, 0, 0], grid[1][0, :, 0], grid[2][0, 0, :]),
                                     data, interpolation_method, bounds_error=False, **kwargs)
    return interp((xi, yi, zi))


def get_cross_section_at_line(grid, val, line, interpolation_method="nearest"):
    """

    :param grid:
    :param val:
    :param line:
    :return: coords, values

    >>> from PAUDV.geometry import Line
    >>> from PAUDV.geometry import Point
    >>> grid = np.meshgrid(np.arange(-10, 10, .1), np.arange(-10, 10, 0.1), (0,), indexing="ij")
    >>> f = grid[0] ** 2 + grid[1] ** 2
    >>> line = Line(Point(-10,-10,0), Point(10,10,0))
    >>> coords, values1 = get_cross_section_at_line(grid, f, line, interpolation_method="nearest")
    >>> coords, values2 = get_cross_section_at_line(grid, f, line, interpolation_method="linear")
    >>> ref = coords[:,0] ** 2 + coords[:,1] ** 2
    >>> np.nanmean((values1 - ref) ** 2) < 1.0
    True
    >>> np.nanmean((values2 - ref) ** 2) < 1.0
    True
    >>> if False:
    ...     from matplotlib import pyplot as plt
    ...     plt.plot(values1)
    ...     plt.plot(coords[:,0]**2 + coords[:,1]**2)
    ...     plt.show()

    """
    dx, dy, dz = get_grid_pitch(grid)
    if dx == 0:
        xpix = 0
    else:
        xpix = (abs((line.point1.x - line.point2.x) / (dx)))
    if dy == 0:
        ypix = 0
    else:
        ypix = (abs((line.point1.y - line.point2.y) / (dy)))
    if dz == 0:
        zpix = 0
    else:
        zpix = (abs((line.point1.z - line.point2.z) / (dz)))

    fact = 2
    n = int((sqrt(xpix ** 2 + ypix ** 2 + zpix ** 2) + 1) * fact)
    points = line.discretize(n)
    coords = pointslist_to_ndarray(points)
    vi = interpolate_from_3dgrid(val, grid, coords[:, 0], coords[:, 1], coords[:, 2], interpolation_method)
    return coords, vi


def pointslist_to_ndarray(points):
    """

    :param points: list of points
    :return: ndarray shape== (pointno, dimension)
    """
    coords = np.empty((len(points), 3))
    for i, p in enumerate(points):
        coords[i, :] = p.pos
    return coords


def get_cross_section_at_points(points, grid, data, interpolation_method="nearest"):
    """
    This method returns the values from data that are specified by the points. The data is organized in the meshgrid
    grid.

    :param points: list of Point objects
    :param grid: meshgrid
    :param data: all possible data correspondatant to the grid
    :return: (vi, dists) vi -> interpolated values, dist -> accumulated distance from the center of the path
    """
    xi = [p.x for p in points]
    yi = [p.y for p in points]
    zi = [p.z for p in points]

    dists = np.zeros((len(points),))
    for i in range(1, len(points)):
        dists[i] = points[i].distance(points[i - 1]) + dists[i - 1]

    dists -= (dists[-1] - dists[0]) / 2  # zero is in the middle

    vi = interpolate_from_3dgrid(data, grid, xi, yi, zi, interpolation_method)
    return vi, dists


def get_pos_list_for_grid_mask(grid, mask):
    """
    Returns a list of the positions for points marked by a binary mask on the grid.
    :param grid: 3d grid (e.g. from np.meshgrid indexing="ij")
    :param mask: binary grid (np.ndarray with dtype==np.bool)
    :return:
    >>> grid = np.meshgrid(np.arange(-10, 10, 2), np.arange(-10, 10, 2), (0,), indexing="ij")
    >>> mask = np.zeros_like(grid[0], dtype=np.bool)
    >>> mask[0, 0] = 1
    >>> mask[9, 5] = 1
    >>> get_pos_list_for_grid_mask(grid, mask)
    array([[-10., -10.,   0.],
           [  8.,   0.,   0.]])

    """
    if mask.dtype is not np.bool:
        mask = mask.astype(np.bool)

    x_idx_list, y_idx_list, z_idx_list = np.where(mask == True)
    pos_list = np.zeros([len(x_idx_list), 3])
    for idx, pos_idx in enumerate(zip(x_idx_list, y_idx_list, z_idx_list)):
        pos_list[idx, 0] = grid[0][pos_idx[0], 0, 0]
        pos_list[idx, 1] = grid[1][0, pos_idx[1], 0]
        pos_list[idx, 2] = grid[2][0, 0, pos_idx[2]]
    return pos_list


def get_tomographic_path_over_grid(pos1, pos2, grid):
    """
    This function calculates the path length in every pixel of the grid that is on the line between pos1 and pos2.
    Only 2D!
    The algorithms is based on the Siddons algorithm: Jacobs, Filip, et al.
    "A fast algorithm to calculate the exact radiological path through a pixel or voxel space."
    Journal of computing and information technology 6.1 (1998): 89-94.

    :param pos1 Point 1
    :param pos2 Point 2
    :param grid meshgrid with x,y,z (with z = 0)

    :return   l[0...N_x, 0...N_y] : 2D array with pathlength of line in every pixel

    """

    # Number of pixels in x and y direction
    N_x = np.shape(grid[0])[0]
    N_y = np.shape(grid[0])[1]

    # contruct array for pathlengths
    l = np.squeeze(np.zeros((N_x - 1, N_y - 1)))
    # overall lengths of line
    dist = np.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)

    # pitch of pixels
    d_x = abs(grid[0][1, 0, 0] - grid[0][0, 0, 0])
    d_y = abs(grid[1][0, 1, 0] - grid[1][0, 0, 0])

    # Are the points meaningful?
    if dist == 0:
        print("Attention: The two given points are equal!")
        return None

    # case for vertical line (not part of siddons algorithms)
    elif pos1.x == pos2.x:
        if pos1.x < np.amin(grid[0][:, 0]) or pos1.x > np.amax(grid[0][:, 0]):
            print("Ray does not intersect pixel space!")
            return None
        if pos2.y < pos1.y:
            p1 = pos2
            p2 = pos1
        else:
            p1 = pos1
            p2 = pos2

        aux0 = grid[0][:, 0] - p1.x
        aux0[aux0 > 0] = np.nan
        idx_x = np.nanargmin(abs(aux0))  # calculate x coordinate of vertical line
        if idx_x >= np.shape(l)[0]:
            idx_x = idx_x - 1  # QUICK AND DIRTY TODO: check condition wether outer bound is hit
        if p1.y <= np.amin(grid[1][0]):
            idx_y_start = 0
            l[idx_x, idx_y_start] = d_y
        else:
            aux1 = (grid[1][0] - p1.y)  # make sure the next upper grid point is chosen
            aux1[aux1 > 0] = np.nan
            idx_y_start = np.nanargmin(abs(aux1))
            l[idx_x, idx_y_start] = d_y - (p1.y - grid[1][0, idx_y_start])
        if p2.y >= np.amax(grid[1][0]):
            idx_y_end = N_y - 1
            l[idx_x, idx_y_end - 1] = d_y
        else:
            aux2 = (grid[1][0] - p2.y)
            aux2[aux2 < 0] = np.nan
            idx_y_end = np.nanargmin(aux2)
            l[idx_x, idx_y_end - 1] = d_y - (grid[1][0, idx_y_end] - p2.y)
        l[idx_x, idx_y_start + 1:idx_y_end - 1] = d_y * np.ones(idx_y_end - idx_y_start - 2)
        return l
    # case for horizontal line (not part of siddons algorithms)
    elif pos1.y == pos2.y:
        if pos1.y < np.amin(grid[1][0]) or pos1.y > np.amax(grid[1][0]):
            print("Ray does not intersect pixel space!")
            return None
        if pos2.x < pos1.x:
            p1 = pos2
            p2 = pos1
        else:
            p1 = pos1
            p2 = pos2

        aux0 = grid[1][0] - p1.y
        aux0[aux0 > 0] = np.nan
        idx_y = np.nanargmin(abs(aux0))  # calculate y coordinate of vertical line
        if idx_y >= np.shape(l)[1]:
            idx_y = idx_y - 1  # QUICK AND DIRTY TODO: check condition wether outer bound is hit
        if p1.x <= np.amin(grid[0][:, 0]):
            idx_x_start = 0
            l[idx_x_start, idx_y] = d_x
        else:
            aux1 = (grid[0][:, 0] - p1.x)  # make sure the next upper grid point is chosen
            aux1[aux1 > 0] = np.nan
            idx_x_start = np.nanargmin(abs(aux1))
            l[idx_x_start, idx_y] = d_x - (p1.x - grid[1][0, idx_x_start])
        if p2.x >= np.amax(grid[0][:, 0]):
            idx_x_end = N_x - 1
            l[idx_x_end - 1, idx_y] = d_x
        else:
            aux2 = (grid[0][:, 0] - p2.x)
            aux2[aux2 < 0] = np.nan
            idx_x_end = np.nanargmin(aux2)
            l[idx_x_end - 1, idx_y] = d_x - (grid[1][0, idx_x_end] - p2.x)
        l[idx_x_start + 1:idx_x_end - 1, idx_y] = d_x * np.ones(idx_x_end - idx_x_start - 2)
        return l

    # lower left corner of pixel space
    b_x = np.amin(np.squeeze(grid[0]))
    b_y = np.amin(np.squeeze(grid[1]))

    # pixel indices from (0,0)...(N_x - 1, N_y - 1)
    # SIDDONS ALGORITHM:
    i = np.arange(0, N_x)
    j = np.arange(0, N_y)

    alpha_x = ((b_x + i * d_x) - pos1.x) / (pos2.x - pos1.x)
    alpha_y = ((b_y + j * d_y) - pos1.y) / (pos2.y - pos1.y)

    alpha_x_min = np.nanmin((alpha_x[0], alpha_x[N_x - 1]))
    alpha_x_max = np.nanmax((alpha_x[0], alpha_x[N_x - 1]))

    alpha_y_min = np.nanmin((alpha_y[0], alpha_y[N_y - 1]))
    alpha_y_max = np.nanmax((alpha_y[0], alpha_y[N_y - 1]))

    alpha_min = np.nanmax((alpha_x_min, alpha_y_min))
    alpha_max = np.nanmin((alpha_x_max, alpha_y_max))

    if alpha_min >= alpha_max:
        print("Ray does not intersect pixel space!")
        return None

    # cases for x/i
    if pos1.x < pos2.x:
        if alpha_min == alpha_x_min:
            i_min = 1
        else:
            phi_x = (pos1.x + alpha_min * (pos2.x - pos1.x) - b_x) / d_x
            i_min = int(np.ceil(phi_x))

        if alpha_max == alpha_x_max:
            i_max = N_x - 1
        else:
            phi_x = (pos1.x + alpha_max * (pos2.x - pos1.x) - b_x) / d_x
            i_max = int(np.floor(phi_x))
    else:
        if alpha_min == alpha_x_min:
            i_max = N_x - 2
        else:
            phi_x = (pos1.x + alpha_min * (pos2.x - pos1.x) - b_x) / d_x
            i_max = int(np.floor(phi_x))

        if alpha_max == alpha_x_max:
            i_min = 0
        else:
            phi_x = (pos1.x + alpha_max * (pos2.x - pos1.x) - b_x) / d_x
            i_min = int(np.ceil(phi_x))

    # cases for y/j
    if pos1.y < pos2.y:
        if alpha_min == alpha_y_min:
            j_min = 1
        else:
            phi_y = (pos1.y + alpha_min * (pos2.y - pos1.y) - b_y) / d_y
            j_min = int(np.ceil(phi_y))

        if alpha_max == alpha_y_max:
            j_max = N_y - 1
        else:
            phi_y = (pos1.y + alpha_max * (pos2.y - pos1.y) - b_y) / d_y
            j_max = int(np.floor(phi_y))
    else:
        if alpha_min == alpha_y_min:
            j_max = N_y - 2
        else:
            phi_y = (pos1.y + alpha_min * (pos2.y - pos1.y) - b_y) / d_y
            j_max = int(np.floor(phi_y))

        if alpha_max == alpha_y_max:
            j_min = 0
        else:
            phi_y = (pos1.y + alpha_max * (pos2.y - pos1.y) - b_y) / d_y
            j_min = int(np.ceil(phi_y))

    if pos1.x < pos2.x:
        alpha_x = alpha_x[i_min:i_max + 1]
    else:
        # alpha_x = alpha_x[range(i_max, i_min,-1)]
        alpha_x = alpha_x[range(i_min, i_max + 1, 1)]

    if pos1.y < pos2.y:
        alpha_y = alpha_y[j_min:j_max + 1]
    else:
        # alpha_y = alpha_y[range(j_max, j_min,-1)]
        alpha_y = alpha_y[range(j_min, j_max + 1, 1)]

    alpha_new = np.hstack((alpha_min, alpha_x, alpha_y))
    if alpha_x_min < 0 and alpha_y_min < 0:
        alpha_new = np.hstack((alpha_new, (0,)))
    if alpha_x_max > 1 and alpha_y_max > 1:
        alpha_new = np.hstack((alpha_new, (1,)))

    alpha_xy = np.unique(np.round(alpha_new, decimals=5))

    alpha_xy = alpha_xy[alpha_xy >= 0]
    alpha_xy = alpha_xy[alpha_xy <= 1]

    for m in range(1, np.shape(alpha_xy)[0]):
        arg_phi_i = (alpha_xy[m] + alpha_xy[m - 1]) / 2
        i_m = np.floor((pos1.x + arg_phi_i * (pos2.x - pos1.x) - b_x) / d_x)

        arg_phi_j = (alpha_xy[m] + alpha_xy[m - 1]) / 2
        j_m = np.floor((pos1.y + arg_phi_j * (pos2.y - pos1.y) - b_y) / d_y)

        # print(j_m, ", " , (pos1.y + arg_phi_j*(pos2.y-pos1.y) - b_y)/d_y)
        try:
            l[i_m, j_m] = (alpha_xy[m] - alpha_xy[m - 1]) * dist
        except:
            print("whats going on here?")

    return l


def regrid_data(x, y, v):
    """
    heuristically regrids data

    :param x: x coordinates ndarray shape==(l,)
    :param y: y coordinates ndarray shape==(l,)
    :param v: data value ndarray shape==(...,l,)
    :return: (xgrid, ygrid, vgrid) reshaped to (xlen,ylen)


    >>> xgrid, ygrid = np.meshgrid(np.arange(10), np.arange(20), indexing='ij')
    >>> vgrid= xgrid **2 + ygrid ** 3
    >>> x = xgrid.flatten()
    >>> y = ygrid.flatten()
    >>> v = vgrid.flatten()
    >>> (xgridnew, ygridnew, vgridnew) = regrid_data(x, y, v)
    >>> np.all(xgridnew == xgrid)
    True
    >>> np.all(vgrid == vgridnew)
    True
    >>> x = xgrid.T.flatten()
    >>> y = ygrid.T.flatten()
    >>> v = vgrid.T.flatten()
    >>> (xgridnew, ygridnew, vgridnew) = regrid_data(x, y, v)
    >>> np.all(xgridnew == xgrid)
    True
    >>> np.all(vgrid == vgridnew)
    True
    >>> vgrid = np.random.random((7,) + xgrid.shape)
    >>> x = xgrid.flatten()
    >>> y = ygrid.flatten()
    >>> v = vgrid.reshape((7,-1))
    >>> (xgridnew, ygridnew, vgridnew) = regrid_data(x, y, v)
    >>> np.all(xgridnew == xgrid)
    True
    >>> np.all(vgrid == vgridnew)
    True

    """

    x = np.array(x).flatten()
    y = np.array(y).flatten()
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    v_additional_dims = v.shape[:-1]

    xdiff = np.flatnonzero(np.diff(x))
    ydiff = np.flatnonzero(np.diff(y))

    if len(ydiff) > 0 and ydiff[0] > 0:  # x is the fastest moving dimension
        xshape = ydiff[0] + 1
        yshape = x.size // xshape
        xgrid = np.swapaxes(x.reshape((yshape, xshape)), -1, -2)
        ygrid = np.swapaxes(y.reshape((yshape, xshape)), -1, -2)
        vgrid = np.swapaxes(v.reshape(v_additional_dims + (yshape, xshape)), -1, -2)
        # print("x ist the fastest moving dimension x={} y={}".format(x, y))

    else:
        yshape = xdiff[0] + 1
        xshape = x.size // yshape
        xgrid = x.reshape((xshape, yshape))
        ygrid = y.reshape((xshape, yshape))
        vgrid = v.reshape(v_additional_dims + (xshape, yshape))
        # print("y ist the fastest moving dimension x={} y={}".format(x, y))

    assert np.all(xgrid[0, 0] == xgrid[0, :])
    assert np.all(ygrid[0, 0] == ygrid[:, 0])
    return xgrid, ygrid, vgrid


def scan_grid(x_rough_max, x_rough_min,
              y_rough_max, y_rough_min,
              x_rough_samples, y_rough_samples,
              x_fine_length, y_fine_length,
              x_fine_samples, y_fine_samples,
              ini_points):
    """
    This function generates a 2-D scan area divided in a rough grid and areas with a finer grid. The fine grid is generated
    considering the initial points.
    :param x_rough_max: maximum x coordinate of the entire scan area
    :param x_rough_min: minimum x coordinate of the entire scan area
    :param y_rough_max: maximum y coordinate of the entire scan area
    :param y_rough_min: minimum y coordinate of the entire scan area
    :param x_rough_samples: number of x-values (rough grid)
    :param y_rough_samples: number of y-values (rough grid)
    :param x_fine_length: expansion of fine grid in x-direction of one examined point
    :param y_fine_length: expansion of fine grid in y-direction of one examined point
    :param x_fine_samples: number of x-values (fine grid)
    :param y_fine_samples: number of y-values (fine grid)
    :param ini_points: points around which the fine grid is arranged
    :return:

    >>> from PAUDV.geometry.primitives import Point
    >>> ini_points = [Point(0, 2, 0), Point(-4, 2, 0)]
    >>> x_rough_max = 10
    >>> x_rough_min = -10
    >>> y_rough_max = 10
    >>> y_rough_min = -10
    >>> x_rough_samples = 21
    >>> y_rough_samples = 21

    >>> x_fine_length = 1
    >>> y_fine_length = 1
    >>> x_fine_samples = 10
    >>> y_fine_samples = 10
    >>> (pointlist, grid) = scan_grid(x_rough_max=x_rough_max, x_rough_min=x_rough_min,
    ...                                 y_rough_max=y_rough_max, y_rough_min=y_rough_min,
    ...                                 x_rough_samples=x_rough_samples, y_rough_samples=y_rough_samples,
    ...                                 x_fine_length=x_fine_length, y_fine_length=y_fine_length,
    ...                                 x_fine_samples=x_fine_samples, y_fine_samples=y_fine_samples,
    ...                                 ini_points=ini_points)
    >>> grid[0][:,0]
    array([-10.        ,  -9.        ,  -8.        ,  -7.        ,
            -6.        ,  -5.        ,  -4.77777778,  -4.55555556,
            -4.33333333,  -4.11111111,  -4.        ,  -3.88888889,
            -3.66666667,  -3.44444444,  -3.22222222,  -3.        ,
            -2.        ,  -1.        ,  -0.77777778,  -0.55555556,
            -0.33333333,  -0.11111111,   0.        ,   0.11111111,
             0.33333333,   0.55555556,   0.77777778,   1.        ,
             2.        ,   3.        ,   4.        ,   5.        ,
             6.        ,   7.        ,   8.        ,   9.        ,  10.        ])
    >>> np.all(np.diff(grid[0], axis=1)==0)
    True
    >>> np.all(np.diff(grid[1], axis=0)==0)
    True
    """

    from PAUDV.geometry.primitives import Point

    x_rough = np.linspace(x_rough_min, x_rough_max, x_rough_samples).tolist()
    y_rough = np.linspace(y_rough_min, y_rough_max, y_rough_samples).tolist()

    x_fine = []
    y_fine = []
    for point in ini_points:
        x_fine.append(np.linspace(point.pos[0] - x_fine_length, point.pos[0] + x_fine_length, x_fine_samples))
        y_fine.append(np.linspace(point.pos[1] - y_fine_length, point.pos[1] + y_fine_length, y_fine_samples))

    # flat the fine lists
    x_fine = list(itertools.chain(*x_fine))
    y_fine = list(itertools.chain(*y_fine))

    x_total = x_rough + x_fine  # combine rough and fine grid points
    x_total = np.sort(np.array(x_total))  # sort the points
    # avoid doublings
    sel_idx = np.pad((np.abs(np.diff(x_total)) >= (x_fine_length / x_fine_samples)), (1, 0), mode="constant",
                     constant_values=True)
    x_total = x_total[sel_idx]

    y_total = y_rough + y_fine
    y_total = np.sort(np.array(y_total))
    sel_idx = np.pad((np.abs(np.diff(y_total)) >= (y_fine_length / y_fine_samples)), (1, 0), mode="constant",
                     constant_values=True)
    y_total = y_total[sel_idx]

    grid = np.meshgrid(x_total, y_total, indexing="ij")  # build scan grid
    pointlist = [Point(x, y, 0) for x, y in zip(grid[0].ravel(), grid[1].ravel())]

    return pointlist, grid

#
# def regrid_data_nd(*args, data=None):
#     """
#     heuristically regrids data
#
#     :param x: x coordinates ndarray shape==(l,)
#     :param y: y coordinates ndarray shape==(l,)
#     :param z:
#     :param ...
#     :param data ndarray shape==(l,)
#     :return: (xgrid, ygrid, zgrid, ...)
#
#
#     >>> xgrid, ygrid = np.meshgrid(np.arange(10), np.arange(20), indexing='ij')
#     >>> vgrid= xgrid **2 + ygrid ** 3
#     >>> x = xgrid.flatten()
#     >>> y = ygrid.flatten()
#     >>> v = vgrid.flatten()
#     >>> regrid_data2(x, y, data=v)
#
#     """
#
#     dims = []
#     for dimidx, vect in enumerate(args):
#         diffidx = np.flatnonzero(np.diff(vect))[0]
#         dims.append((dimidx, vect, diffidx))
#
#     dims.sort(key=lambda a: a[2])
#
#     shape = []
#     permute = []
#     for dimidx, vect, diffidx in dims:
#         print(dimidx, diffidx)
#         if diffidx > 0:
#             shape.append(diffidx + 1)
#
#     shape.append(-1)
#
#     np.reshape()
