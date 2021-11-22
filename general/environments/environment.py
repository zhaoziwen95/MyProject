import functools
import logging

import joblib
import numpy as np

from PAUDV.geometry import Point, Plane, Line
from PAUDV.geometry.grid_shapes import rect_grid_selector
from PAUDV.geometry.grid_tools import get_cross_section_at_line
from PAUDV.publish.plot import plot_image_from_grid
from PAUDV.soundfield.abs_environment import Environment
from PAUDV.soundfield.materials import Material, DEFAULT_MATERIAL_COLOUR_MAP

logger = logging.getLogger(__name__)


class SimpleEnvironment(Environment):
    """
    Environment model that describes a homogeneous material with a given sound speed.
    """

    def __init__(self, sound_speed: float):
        self.sound_speed = sound_speed

    def t_sound_travel(self, pos1: Point, pos2: Point) -> float:
        return pos1.distance(pos2) / self.sound_speed

    def to_complex_env(self):
        ce = ComplexEnv()
        ce.set_material(Material("unknown", self.sound_speed))
        return ce

    def __hash__(self):
        return int(self.sound_speed * 1e6)


class HomogFluidEnv(SimpleEnvironment):
    """
    Environment model that describes a homogeneous material (Material instance given).
    """

    def __init__(self, material: Material):
        self.material = material
        super().__init__(self.material.sound_speed)

    def to_complex_env(self):
        ce = ComplexEnv()
        ce.set_material(self.material)
        return ce

    def __hash__(self):
        return hash(self.material)

    def __str__(self):
        return "{} bulk({})".format(super().__str__(), str(self.material))


class HomogFluidWallEnv(HomogFluidEnv):
    """
    Environment model consisting of a bulk material and a wall_material with boundary_plane
    being the border between the two materials. Bulk material is in positive direction of normal_vect of
    the boundary plane.
    """

    break_point_diff = 1e-5
    max_n_iterations = 20

    def __init__(self, bulk_material: Material, wall_material: Material, boundary_plane: Plane):
        """
        :param bulk_material: material characteristics of the bulk material (fluid)
        :param wall_material: material characteristics of the wall material
        :param boundary_plane: plane between fluid and wall. Take care that the surface normal of the plane
                               (plane.normal_vect) points into the direction of the fluid.
        """
        self.wall_material = wall_material
        self.boundary_plane = boundary_plane

        super().__init__(bulk_material)

    @functools.lru_cache(maxsize=2 ** 16)
    def t_sound_travel(self, pos1: Point, pos2: Point) -> float:
        pos1_dist = self.boundary_plane.oriented_distance_to_point(pos1)
        pos2_dist = self.boundary_plane.oriented_distance_to_point(pos2)

        if pos1_dist >= 0 and pos2_dist >= 0:  # pos1 and pos2 both in bulk material
            return pos1.distance(pos2) / self.material.sound_speed
        elif pos1_dist <= 0 and pos2_dist <= 0:  # pos1 and pos2 both in wall material
            return pos1.distance(pos2) / self.wall_material.sound_speed
        else:  # pos1 and pos2 on opposite sides of the boundary
            if pos1_dist <= 0:
                wall_pos = pos1
                bulk_pos = pos2
            else:
                wall_pos = pos2
                bulk_pos = pos1

            point_of_incidence = self.get_point_of_incidence(wall_pos=wall_pos, bulk_pos=bulk_pos)

            if point_of_incidence is None:
                return None
            else:
                return bulk_pos.distance(point_of_incidence) / self.material.sound_speed + \
                       wall_pos.distance(point_of_incidence) / self.wall_material.sound_speed

    def to_complex_env(self):
        def wall_mask_func(grid):
            return self.boundary_plane.oriented_distance_to_points(grid[0], grid[1], grid[2]) < 0

        ce = ComplexEnv()
        ce.set_material(self.material)
        ce.set_material(self.wall_material, wall_mask_func)
        return ce

    def get_point_of_incidence(self, wall_pos: Point, bulk_pos: Point) -> Point:
        """
        Returns point of incidence for a refracted sound "ray" (geometrical acoustics).
        :param wall_pos:
        :param bulk_pos:
        :return:
        """
        wall_proj = self.boundary_plane.perpendicular(wall_pos)
        l_wall = abs(self.boundary_plane.distance(wall_pos))
        bulk_proj = self.boundary_plane.perpendicular(bulk_pos)
        l_bulk = abs(self.boundary_plane.distance(bulk_pos))
        distance = wall_proj.distance(bulk_proj)

        if distance == 0:
            return wall_proj
        else:
            def step(d_trans):
                """
                step of newton's method
                :param d_trans: old x_value
                :return: new x_value
                """
                c_sp = self.material.sound_speed ** 2 / self.wall_material.sound_speed ** 2

                return d_trans - \
                       (d_trans ** 4 * (l_wall ** 2 * l_bulk ** 2 * (1 - c_sp)) -
                        d_trans ** 3 * 2 * distance * l_wall ** 2 * l_bulk ** 2 * (1 - c_sp) +
                        d_trans ** 2 * (distance ** 2 * l_wall ** 2 * l_bulk ** 2 * (1 - c_sp) +
                                        l_wall ** 4 * l_bulk ** 2 -
                                        l_wall ** 2 * l_bulk ** 4 * c_sp
                                        ) +
                        d_trans * 2 * distance * l_wall ** 2 * l_bulk ** 4 * c_sp -
                        distance ** 2 * l_wall ** 2 * l_bulk ** 4 * c_sp
                        ) / \
                       (d_trans ** 3 * 4 * (l_wall ** 2 * l_bulk ** 2 * (1 - c_sp)) -
                        d_trans ** 2 * 6 * distance * l_wall ** 2 * l_bulk ** 2 * (1 - c_sp) +
                        d_trans * 2 * (distance ** 2 * l_wall ** 2 * l_bulk ** 2 * (1 - c_sp) +
                                       l_wall ** 4 * l_bulk ** 2 -
                                       l_wall ** 2 * l_bulk ** 4 * c_sp
                                       ) +
                        2 * distance * l_wall ** 2 * l_bulk ** 4 * c_sp
                        )

            d_poi = d_poi_start = distance * (l_bulk * self.material.sound_speed /
                                              (
                                                      l_bulk * self.material.sound_speed + l_wall * self.wall_material.sound_speed))
            d_poi_old = d_poi + 2 * self.break_point_diff
            n_iterations = 0

            while abs(d_poi - d_poi_old) > self.break_point_diff:
                d_poi_old = d_poi

                if n_iterations >= self.max_n_iterations:
                    logger.warning("Max number of iterations ({}) reached! {} > {}".format(self.max_n_iterations,
                                                                                           abs(d_poi - d_poi_old),
                                                                                           self.break_point_diff))
                    break
                n_iterations += 1

                d_poi = step(d_poi_old)

                if d_poi < 0:
                    # logger.warning("Refraction point calculation out of bounds."
                    #            "Last d_poi: {} Use {} instead.".format(d_poi, d_poi_start))
                    return None
                    # d_poi = d_poi_start
                    # break

                elif d_poi > distance:
                    # logger.warning("Refraction point calculation out of bounds."
                    #            "Last d_poi: {} Use {} instead.".format(d_poi, d_poi_start))
                    return None
                    # d_poi = d_poi_start
                    # break
                    # raise ValueError(
                    #     "Refraction point calculation out of bounds. Last d_poi: {}".format(d_poi))

            return bulk_proj + (wall_proj - bulk_proj) * d_poi / distance

    def __str__(self):
        return "{} wall({})".format(super().__str__(), str(self.wall_material))

    def __hash__(self):
        return int(joblib.hashing.hash([self.wall_material, self.boundary_plane, self.material]), base=16)


class HomogFluidThinWallEnv(HomogFluidEnv):
    """
    Environment model consisting of a homogeneous fluid (bulk material) and a simple representation
    of a thin wall (wall material). Refraction at the boundary between wall and fluid is not modelled
    (wall is modelled as a constant length (=wall_thickness) of the distance between pos1 and pos2 to
    be in wall material. -> usable for thin walls only).
    """

    def __init__(self, bulkmaterial: Material, wall_thickness: float, wall_material: Material):
        self.wall_thickness = wall_thickness
        self.wall_material = wall_material
        super().__init__(bulkmaterial)

    def t_sound_travel_with_refraction(self, pos1: Point, pos2: Point) -> float:
        """
        Calculates sound travel time taking refraction at the wall into account
        assume second position is array pos, wall in +y-dir
        :param pos1: position 1
        :param pos2: position 2
        """
        dist = pos1.distance(pos2)

        if dist < self.wall_thickness:
            return dist / self.wall_material.sound_speed
        else:
            diff = pos2 - pos1
            beta = np.arctan2(abs(diff.x), abs(diff.y))
            alpha = np.arcsin(np.sin(beta) * self.wall_material.sound_speed / self.sound_speed)

            dist_wall = self.wall_thickness / np.cos(alpha)
            x_diff = self.wall_thickness / np.sin(alpha)
            if diff.x >= 0:
                p = Point(diff.x - x_diff, diff.y - self.wall_thickness, 0)
            else:
                p = Point(diff.x + x_diff, diff.y - self.wall_thickness, 0)
            dist_medium = p.distance()

            # print("dist: {}, medium: {}, wall: {}, sum: {}".format(dist,
            #                                                        dist_medium,
            #                                                        dist_wall,
            #                                                        dist_medium + dist_wall))
            # t_time = dist_medium / self.sound_speed + dist_wall / self.wall_material.sound_speed
            # print("time_new: {}, time_old: {}".format(t_time, self.t_sound_travel(pos1, pos2)))

            return dist_medium / self.sound_speed + dist_wall / self.wall_material.sound_speed

    def t_sound_travel(self, pos1: Point, pos2: Point) -> float:
        """
        without refraction
        :param pos1: position 1
        :param pos2: position 2
        """

        dist = pos1.distance(pos2)
        if dist < self.wall_thickness:
            return dist / self.wall_material.sound_speed
        else:
            return (dist - self.wall_thickness) / self.sound_speed + \
                   self.wall_thickness / self.wall_material.sound_speed

    def to_complex_env(self):
        ce = ComplexEnv()
        ce.set_material(self.material)
        logging.warning("Ignoring wall for conversion to complex environment!")
        # ce.set_material(self.wall_material)
        return ce

    def __hash__(self):
        return joblib.hashing.hash([self.wall_thickness, self.material, self.wall_material], hash_name='sha1')


class GriddedEnv(Environment):
    """
    a generic environment model that is based on a grid with material parameters

    Attention: t_sound_travel() gives just a rough approximation of the TOF based on a straight line path
    """

    def __init__(self, grid):
        dtype = np.float32
        self.grid = grid
        self.materials = set()

        self.sound_speed = np.zeros_like(grid[0], dtype)
        self.shearwave_speed = np.zeros_like(grid[0], dtype)

        self.density = np.zeros_like(grid[0], dtype=dtype)
        self.c11 = np.zeros_like(grid[0], dtype=dtype)
        self.c12 = np.zeros_like(grid[0], dtype=dtype)
        self.c22 = np.zeros_like(grid[0], dtype=dtype)
        self.c44 = np.full_like(grid[0], 1e-30, dtype=dtype)
        self.eta_v = np.full_like(grid[0], 1e-30, dtype=dtype)
        self.eta_s = np.full_like(grid[0], 1e-30, dtype=dtype)
        self.nonphysical_sound_absorption = np.zeros_like(grid[0], dtype=dtype)
        #: Stores the material hash (as returned by `hash(material)`) for each
        #: point in the grid. The hash function returns a fixed-length integer
        #: (Py_ssize_t) corresponding to `np.intp`.
        self.material_hash = np.zeros_like(grid[0], dtype=np.intp)

        self.temperature = np.full_like(grid[0], np.nan, dtype=dtype)  # in Kelvin

    def t_sound_travel(self, pos1: Point, pos2: Point) -> float:
        l = Line(pos1, pos2)
        coords, vals = get_cross_section_at_line(self.grid, 1 / self.sound_speed, l)
        return l.length() * np.mean(vals)

    def set_material(self, m: Material, mask=None):
        if mask is None:
            mask = np.ones_like(self.grid[0], dtype=np.bool)
        else:
            mask = mask > 0  # make mask a boolean array
        self.sound_speed[mask] = m.sound_speed
        self.shearwave_speed[mask] = m.shearwave_speed
        self.density[mask] = m.density
        self.c11[mask] = m.c11
        self.c12[mask] = m.c12
        self.c22[mask] = m.c22
        self.c44[mask] = m.c44
        self.eta_v[mask] = m.eta_v
        self.eta_s[mask] = m.eta_s
        self.nonphysical_sound_absorption[mask] = m.nonphysical_sound_absorption
        self.material_hash[mask] = hash(m)
        self.materials.add(m)

    def set_material_rect(self, m: Material, xlim=None, ylim=None, zlim=None):
        self.set_material(m, rect_grid_selector(self.grid, xlim=xlim, ylim=ylim, zlim=zlim))

    def plot(self, ax, env_slice=(slice(None), slice(None), 0), mat_colour_map=None, coloured=True, hatched=False,
             **kwargs):
        super().plot(ax, **kwargs)
        import matplotlib.colors

        if mat_colour_map is None:
            mat_colour_map = DEFAULT_MATERIAL_COLOUR_MAP

        im = self.material_hash[env_slice]

        if not hatched:
            bgimg = np.zeros(im.shape + (3,))
            for mat, (color, hatch) in mat_colour_map.items():
                idx = im == hash(mat)
                x, y = idx.nonzero()
                bgimg[x, y, :] = matplotlib.colors.colorConverter.to_rgb(color)

            bg = plot_image_from_grid(ax, self.grid[0][env_slice], self.grid[1][env_slice], bgimg,
                                      **kwargs)

        else:
            bgimg = np.zeros(im.shape)
            i = 0
            hatches = []
            colors = []

            for mat, (color, hatch) in mat_colour_map.items():
                idx = im == hash(mat)
                x, y = idx.nonzero()
                if len(x) > 0:
                    bgimg[x, y] = i
                    hatches.append(hatch)
                    colors.append(color)
                    i += 1

            if not coloured:
                colors = 'none'
            bg = ax.contourf(self.grid[0][env_slice], self.grid[1][env_slice], bgimg, len(hatches), colors=colors,
                             hatches=hatches, levels=range(len(hatches)), extend='lower', **kwargs)
        return bg

    def plot3d(self, ax, env_slice=(slice(None), slice(None), 0), axis1vec=(0, 0), mat_colour_map=None, coloured=True,
               hatched=False,
               **kwargs):
        super().plot(ax, **kwargs)
        import matplotlib.colors

        if mat_colour_map is None:
            mat_colour_map = DEFAULT_MATERIAL_COLOUR_MAP

        im = self.material_hash[env_slice]

        if not hatched:
            bgimg = np.zeros(im.shape + (3,))
            for mat, (color, hatch) in mat_colour_map.items():
                idx = im == hash(mat)
                x, y = idx.nonzero()
                bgimg[x, y, :] = matplotlib.colors.colorConverter.to_rgb(color)

            bg = plot_image_from_grid(ax, self.grid[axis1vec[0]][env_slice], self.grid[axis1vec[1]][env_slice], bgimg,
                                      **kwargs)

        else:
            bgimg = np.zeros(im.shape)
            i = 0
            hatches = []
            colors = []

            for mat, (color, hatch) in mat_colour_map.items():
                idx = im == hash(mat)
                x, y = idx.nonzero()
                if len(x) > 0:
                    bgimg[x, y] = i
                    hatches.append(hatch)
                    colors.append(color)
                    i += 1

            if not coloured:
                colors = 'none'
            bg = ax.contourf(self.grid[axis1vec[0]][env_slice], self.grid[axis1vec[1]][env_slice], bgimg, len(hatches),
                             colors=colors,
                             hatches=hatches, levels=range(len(hatches)), extend='lower', **kwargs)
        return bg

    def to_complex_env(self):
        raise NotImplementedError

    def __hash__(self):
        return int(joblib.hashing.hash(self.material_hash), base=16)


def _simple_set_prop_func(ge: GriddedEnv, m: Material, mask_func):
    if mask_func is not None:
        ge.set_material(m, mask_func(ge.grid))
    else:
        ge.set_material(m)


class ComplexEnv(Environment):
    """
    a generic environment model
    specify the material through set_material calls

    """

    def __init__(self):
        self.materials = set()
        self._materials_stack = []

    def set_material(self, m: Material, mask_func=None):
        """
        example that draws a circle:

        env.set_material(gainsn, functools.partial(spherical_grid_selector, center=center, radius=radius))

        :param m:                   material
        :param mask_func:           a function that takes a grid and returns a mask
        :return:
        """
        if not isinstance(m, Material):
            logger.warning("I need a material!")

        self.set_parametric_properties(m, set_prop_func=functools.partial(_simple_set_prop_func, mask_func=mask_func))

    def set_parametric_properties(self, m: Material, set_prop_func):
        """

        :param m: a material
        :param set_prop_func: a function that takes a gridded environment and a material.
                                Its purpose is to modify the properties inside the gridded environment.
                                See GriddedEnv.set_material().

        :return:


        >>> m1 = Material("air", 1, 0.1)
        >>> m2 = Material("steel", 2, 0.2)
        >>> ce = ComplexEnv()
        >>> ce.set_material(m1)
        >>> def set_prop_func(ge: GriddedEnv, m: Material):
        ...     mask = ge.grid[0] > 0.1
        ...     T = ge.grid[0] / np.var(ge.grid[0]) + ge.grid[1] / np.var(ge.grid[1])
        ...     T = T [mask]
        ...     ge.sound_speed[mask] = m.sound_speed * (1 + T)
        ...     ge.shearwave_speed[mask] = m.shearwave_speed * (1 + T)
        ...     ge.density[mask] = m.density * (1 + T)
        ...     ge.c11[mask] = m.c11 * (1 + T)
        ...     ge.c12[mask] = m.c12
        ...     ge.c22[mask] = m.c22
        ...     ge.c44[mask] = m.c44
        ...     ge.eta_v[mask] = m.eta_v
        ...     ge.eta_s[mask] = m.eta_s
        ...     ge.material_hash[mask] = hash(m)
        >>> ce.set_parametric_properties(m2, set_prop_func)
        >>> grid = np.meshgrid(np.arange(-10e-3, 10e-3, 0.25e-3), np.arange(0e-3, 30e-3, 0.25e-3), np.arange(-2e-3, 2e-3, 0.25e-3), indexing='ij')
        >>> ge = ce.discretize(grid)
        """

        if not isinstance(m, Material):
            logger.warning("I need a material!")
        self._materials_stack.append((m, set_prop_func))
        self.materials.add(m)

    def set_material_rect(self, m: Material, xlim=None, ylim=None, zlim=None):
        maskfunc = functools.partial(rect_grid_selector, xlim=xlim, ylim=ylim, zlim=zlim)
        self.set_material(m, mask_func=maskfunc)

    def discretize(self, grid) -> GriddedEnv:
        ge = GriddedEnv(grid)
        for m, prop_func in self._materials_stack:
            if prop_func is not None:
                prop_func(ge, m)
            else:
                ge.set_material(m)
        return ge

    def t_sound_travel(self, pos1: Point, pos2: Point) -> float:
        raise NotImplementedError("discretize the ComplexEnv first!")

    def to_complex_env(self):
        return self

    def plot(self, ax, env_slice=(slice(None), slice(None), 0), mat_colour_map=None, coloured=True, hatched=False,
             **kwargs):
        logger.warning("No plotting for complex environment implemented yet, discretize environment first..")
        super().plot(ax, **kwargs)

        # if mat_colour_map is None:
        #     mat_colour_map = DEFAULT_MATERIAL_COLOUR_MAP
        #
        # for (m, xlim, ylim, zlim) in self._materials_stack:
        #     (color, hatch) = mat_colour_map[m]
        #
        #     if coloured:
        #         c = matplotlib.colors.colorConverter.to_rgb(color)
        #     else:
        #         c = "k"
        #
        #     if not hatched:
        #         hatch = ""
        #
        #     if xlim is None:
        #         xlim = [-10, 10]
        #     if ylim is None:
        #         ylim = [-10, 10]
        #     if zlim is None:
        #         zlim = [-10, 10]
        #
        #     mincord = [xlim[0], ylim[0], zlim[0]]
        #     maxcord = [xlim[1], ylim[1], zlim[1]]
        #     x, y = mincord[env_slice]
        #     w, h = maxcord[env_slice] - mincord[env_slice]
        #
        #     Rectangle((x, y), w, h, axis=ax, color=c, hatch=hatch)

        return None

    def __hash__(self):
        return int(joblib.hashing.hash(self._materials_stack), base=16)
