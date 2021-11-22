import logging
import os
from typing import Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PAUDV.data.data import Data
from PAUDV.data_proc.sig_proc.time_signal import TimeSignal
from PAUDV.geometry.grid_tools import get_cross_section_at_points, find_level_around, find_nearest_gridpoint, \
    get_grid_pitch

from PAUDV.geometry.tools import get_coordinates_cross_section
from PAUDV.publish.plot import plot_draw_array, plot_image_from_grid, update_image_from_grid, cmap_make_transp
from PAUDV.soundfield import SimpleEnvironment
from PAUDV.geometry.primitives import Point
from matplotlib import animation
from PAUDV.publish.plot import plot_image_from_grid
from PAUDV.soundfield.abs_environment import Environment
from PAUDV.soundfield.environment import GriddedEnv

__author__ = 'olg'
logger = logging.getLogger(__name__)


class Sndfield(object):
    def __init__(self, grid, p_rms=None, task=None, env=None, p=None):
        """
        :param grid:
            either
            * give a grid as in grid = np.meshgrid(x, y, z, indexing="ij" )
            * [default] let the grid be automatically determined
            * give a tuple of the number of gridpoint (xcount, ycount, zcount) for automatic meshing
        :param task: task object describing the
        :return:
        """
        self.task = task
        self.env = env
        if p_rms is not None:
            self.p = p_rms[..., np.newaxis]
        else:
            self.p = p
        self.focalspot = None
        self.grid = grid
        assert isinstance(self.grid, (list, tuple)) and isinstance(self.grid[0], np.ndarray) or (
                isinstance(self.grid, np.ndarray) and self.grid.shape[0] == 3)

    @property
    def p_rms(self):
        return np.sum(self.p ** 2, axis=-1) ** 0.5

    def find_max_around(self, pos, radius):
        # only 2d for now
        v = np.abs(np.copy(self.p_rms))
        v[(self.grid[0] - pos.x) ** 2 + (self.grid[1] - pos.y) ** 2 > radius ** 2] = - np.inf

        idx = np.unravel_index(v.argmax(), v.shape)
        vmax = v[idx]

        return Point(self.grid[0][idx], self.grid[1][idx], 0), vmax

    def find_focalspot(self, pos=None, radius=None, rellevel=0.5):
        """
        returns a mask of the focal area
        :param pos: where to start looking for the maximum
        :param radius: specify the circle where the max is searched
        :return:
        """
        if pos is None:
            pos = self.focus
        if radius is None:
            radius = 10e-3

        pos, p_max = self.find_max_around(pos, radius)
        self.focalspot = find_level_around(self.grid, np.abs(self.p_rms), pos, p_max * rellevel)
        return self.focalspot

    def plot_draw_array(self, ax, **kwargs):
        plot_draw_array(self.task, ax, **kwargs)

    def plot_draw_tof(self, ax, point=None, draw_tof_lines=True, draw_isochrones=True):
        # draw focus
        if point is None:
            point = self.focus

        ax.scatter(point.x, point.y, marker='x')

        tx_pos = self.task.paudv_settings.tx.array.get_center_pos()

        # draw line to focus
        if draw_tof_lines:
            l = [(tx_pos, "g")]
            for rx_array in self.task.paudv_settings.rx_all_arrays:
                l.append((rx_array.get_center_pos(), "r"))

            for p, c in l:
                ax.plot([p.x, point.x], [p.y, point.y], c=c)

        rx_pos = self.task.paudv_settings.rx[0].array.get_center_pos()

        if draw_isochrones:
            iso_mask = Sndfield.calc_isochrones(self.grid, self.task.aux["environment"], tx_pos, point, 1e-6, rx_pos)
            self.plot_draw_mask(ax, iso_mask, c="green", masks_alpha=0.7)

    def plot_draw_mask(self, ax, mask, c="black", masks_alpha=0.3):
        # 2d for now
        cmap = matplotlib.colors.ListedColormap([(0, 0, 0, 0),
                                                 matplotlib.colors.colorConverter.to_rgba(c, alpha=masks_alpha)])
        ax.imshow(mask[:, :, 0].transpose(), origin="lower",
                  extent=(self.grid[0][0, 0, 0], self.grid[0][-1, 0, 0], self.grid[1][0, 0, 0], self.grid[1][0, -1, 0]),
                  cmap=cmap, zorder=10)

    def plot_cross_section(self, ax, points, c="black", draw_fwhm=True, zero_in_the_middle=True,
                           draw_maximum=False, dx=None, draw_quality=False, fob=False, only_highest_lobe=True) -> tuple:
        """
        Plots a cross section of the p_rms from a simulated soundfield

        :param ax:
        :param points: list of Point objects
        :param c: color of the line
        :param draw_fwhm: Boolean for drawing the FWHM
        :param zero_in_the_middle: Boolean for putting zero to the middle of the x-axis
        :param fob: fwhm_other_base -> Half is based on difference max to min, not max to 0
        :param only_highest_lobe: Change whether all lobes or only the highest lobe is considered for FWHM calculation.
        :return: ax, dists, vi, fwhm
        """

        vi, dists = get_cross_section_at_points(points, self.grid, self.p_rms)

        if not zero_in_the_middle:
            dists = []
            for Pos in points:
                dists.append(np.sqrt(Pos.x ** 2 + Pos.y ** 2 + Pos.z ** 2))

        ax.plot(dists, vi, c=c)
        ax.set_xlabel("distance / m")
        ax.set_ylabel("p_rms / a.u.")

        if draw_fwhm:
            fwhm_idx, fwhm = self.get_FWHM(points=dists, data=vi, fob=fob, only_highest_lobe=only_highest_lobe)
            if fob == True:
                maxval = np.nanmax(vi)
                minval = np.nanmin(vi)
                maxval_diff = maxval - minval  # TODO: Orginal: ohne minval und maxvall_diff
                if fwhm is not None:
                    ax.plot([dists[fwhm_idx[0]], dists[fwhm_idx[1]]], [(maxval_diff / 2) + minval] * 2)
                    ax.text(dists[fwhm_idx[0]], (maxval_diff / 2) + minval, "  FWHM={:11.8g} mm".format(fwhm * 1000))
            elif fob == False:
                maxval = np.nanmax(vi)

                if fwhm is not None:
                    ax.plot([dists[fwhm_idx[0]], dists[fwhm_idx[1]]], [maxval / 2] * 2)
                    ax.text(dists[fwhm_idx[0]], maxval / 2, "  FWHM={:11.8g} mm".format(fwhm * 1000))

        if draw_maximum:
            maxval = np.nanmax(vi)
            m_idx = np.nanargmax(vi)
            ax.plot(dists[m_idx], maxval, '.', c='b')
            ax.text(dists[m_idx], maxval, "({:6.3e}, {:6.3e})".format(dists[m_idx], maxval))

        if draw_quality:
            fwhm_idx, fwhm = self.get_FWHM(points=dists, data=vi, fob=fob, only_highest_lobe=only_highest_lobe)
            maxval = np.nanmax(vi)
            if fwhm is not None:
                ax.text(dists[1], 0.9 * maxval, "Q={:6.3e}".format(maxval / fwhm))

        # draw energy of FWHM
        if dx is not None:
            fwhm_idx, fwhm = self.get_FWHM(points=dists, data=vi, fob=fob, only_highest_lobe=only_highest_lobe)
            maxval = np.nanmax(vi)
            if fwhm is not None:
                energy = np.cumsum((vi[fwhm_idx[0]:fwhm_idx[1]] ** 2) * dx)
                ax.text(dists[1], 0.8 * maxval, "E={:6.3e}".format(energy[-1]))

        return ax, dists, vi, fwhm

    def get_slice(self, var="z", idx=None):
        if var == "z":
            if idx is None:
                idx = int(np.round(self.grid[2].shape[2] / 2))
            grid = [g[:, :, [idx, ]] for g in self.grid]
            p_rms = self.p_rms[:, :, [idx, ]]
        else:
            raise NotImplementedError

        return Sndfield(grid, p_rms, self.task)

    @staticmethod
    def get_FWHM(points, data, fob=False, only_highest_lobe=True) -> Tuple[Tuple[int, int], float]:
        """Get the FWHM of the specified soundfield slice.

        :param points: list of Point objects, these are the coordinates used for distance calculation.
        :param data: 1D array of the sound energy p_rms
        :param fob: Change whether the half maximum is base on the on difference max to min or max to 0. (FWHM other base)
        :param only_highest_lobe: Change whether all lobes or only the highest lobe is considered for FWHM calculation.
        :return:  A tuple with a tuple with the left  and right index of the fwhm and the fwhm.
        >>> points = np.array([0, 1, 2, 3, 4])
        >>> data = np.array([0, 1, 10, 1, 0])
        >>> Sndfield.get_FWHM(points, data)
        ((1, 3), 2)
        >>> points = np.arange(0, 0.7, 0.1)
        >>> data = np.array([2, 2, 4, 7, 4, 2, 2])
        >>> Sndfield.get_FWHM(points, data)
        ((1, 5), 0.4)
        >>> Sndfield.get_FWHM(points, data, fob=True)
        ((2, 4), 0.2)
        >>> Sndfield.get_FWHM([0, 1, 2], [0, 1, 0])
        ((None, None), None)
        >>> Sndfield.get_FWHM(points, np.arange(7))
        ((None, None), None)
        >>> Sndfield.get_FWHM([0], [float('NaN')])
        ((None, None), None)
        >>> points = np.arange(0, 1, 0.1)
        >>> data = [0, 0, 7, 2, 10, 2, 7, 0, 0]
        >>> Sndfield.get_FWHM(points, data, only_highest_lobe=True)
        ((3, 5), 0.19999999999999996)
        >>> data = [0, 0, 7, 2, 10, 2, 7, 0, 0]
        >>> Sndfield.get_FWHM(points, data, only_highest_lobe=False)
        ((2, 6), 0.4000000000000001)
        >>> points = np.arange(0, 1.7, 0.1)
        >>> data = [0, 0, 0, 0, 0, 0, 7, 2, 10, 2, 7, 0, 0, 0, 0,0, 0,]
        >>> Sndfield.get_FWHM(points, data, only_highest_lobe=True)
        ((7, 9), 0.19999999999999996)
        >>> Sndfield.get_FWHM(points, data, only_highest_lobe=False)
        ((6, 10), 0.3999999999999999)
        """

        fwhm = None
        fwhm_left_index = None
        fwhm_right_index = None

        maximum_value = np.nanmax(data)

        if not np.isnan(maximum_value):
            index_at_maximum = np.flatnonzero(data == maximum_value)[0]
            half_maximum = maximum_value / 2
            if fob:
                minimum_value = np.nanmin(data)
                logger.debug('sndfield, Z215, minval: %s', minimum_value)
                difference_max_min = maximum_value - minimum_value
                half_maximum = (difference_max_min / 2) + minimum_value

            fwhm_left_indexes = np.flatnonzero(data[index_at_maximum:0:-1] < half_maximum)
            fwhm_right_indexes = np.flatnonzero(data[index_at_maximum:-1] < half_maximum)
            if not only_highest_lobe:
                fwhm_left_indexes = np.flatnonzero(data[index_at_maximum:0:-1] > half_maximum)[::-1]
                fwhm_right_indexes = np.flatnonzero(data[index_at_maximum:-1] > half_maximum)[::-1]

            if not (len(fwhm_left_indexes) == 0 or len(fwhm_right_indexes) == 0):
                fwhm_left_index = index_at_maximum - fwhm_left_indexes[0]
                fwhm_right_index = index_at_maximum + fwhm_right_indexes[0]
                fwhm = points[fwhm_right_index] - points[fwhm_left_index]
            else:
                logger.warning("FWHM is larger than the selected crossection. Returning: ((None, None), None).")

        return (fwhm_left_index, fwhm_right_index), fwhm

    def plot_axial_cross_section(self, ax, point=None, length=None, steps=400, c="black", extend_beyond_point=False):
        """
        Plots the axial cross section between center of the txarray and the defined point

        :param ax:
        :param point: Point object
        :param length: length of section in m
        :param steps: number of discretization steps
        :param c: color of line plot
        :return:
        """
        if point is None:
            point = self.focus

        tx_pos = self.task.paudv_settings.tx.array.ref_element.position
        if extend_beyond_point:
            point = (point - tx_pos) * 2 + tx_pos

        points = get_coordinates_cross_section(tx_pos, point, length, steps, type='axial')
        self.plot_cross_section(ax, points, c, zero_in_the_middle=False)
        return points

    def plot_lateral_cross_section(self, ax, point=None, length=10e-3, steps=400, c="black"):
        """
        Plots the lateral cross section perpendicular to the line between txarray center
        and the defined point at the defined point.

        :param ax:
        :param point: Point object
        :param length: length of section in m
        :param steps: number of discretization steps
        :param c: color of line plot
        :return:
        """
        if point is None:
            point = self.focus

        tx_pos = self.task.paudv_settings.tx.array.ref_element.position
        points = get_coordinates_cross_section(tx_pos, point, length, steps, type='lateral')
        self.plot_cross_section(ax, points, c)
        return points

    def prepare_axis(self, ax=None, tickdist=None, plotdir="x-y"):
        """Prepare the SOUND FIELD plot.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis
            The axis to draw into.
        tickdist : (number, number)
            Step distance of ticks for the x and y axis of the plot.
        plotdir: {"x-y", "x-z", "y-z"}
            The plane in which to plot the soundfield.
        """
        ax.set_aspect(aspect=1)

        xFormatter = FuncFormatter(tickfunc_milli)
        yFormatter = FuncFormatter(tickfunc_milli)

        ax.xaxis.set_major_formatter(xFormatter)
        ax.yaxis.set_major_formatter(yFormatter)

        if plotdir == "x-y":
            x_label = "$x$ / mm"
            y_label = "$y$ / mm"
            x_grid = self.grid[0]
            y_grid = self.grid[1]
        elif plotdir == "x-z":
            x_label = "x / mm"
            y_label = "z / mm"
            x_grid = self.grid[0]
            y_grid = self.grid[2]
        elif plotdir == "y-z":
            x_label = "y / mm"
            y_label = "z / mm"
            x_grid = self.grid[1]
            y_grid = self.grid[2]
        else:
            raise ValueError(f"unkown plot direction: {plotdir}")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Construct vector for tick positions
        x_start = np.amin(x_grid)
        x_stop = np.amax(x_grid)
        y_start = np.amin(y_grid)
        y_stop = np.amax(y_grid)
        if tickdist is None:
            tickdist = ((x_stop - x_start) / 10, (y_stop - y_start) / 10)
        x_start = np.ceil(x_start / tickdist[0]) * tickdist[0]
        y_start = np.ceil(y_start / tickdist[1]) * tickdist[1]
        x_ticks = np.arange(x_start, x_stop, tickdist[0])
        y_ticks = np.arange(y_start, y_stop, tickdist[1])
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

    def plot_prms(self, fig, ax, cmap=plt.get_cmap('gist_heat'), logarithmic=True, plot_in_dB=False, vmin=None,
                  vmax=None, plot_slice=(slice(None), slice(None), 0), plottype="contour", plotdir="x-y", outdir=None,
                  prefix_str=None, **kwargs):

        data = np.abs(self.p_rms[plot_slice])

        if vmin is None:
            vmin = np.nanmin(data)

        if vmax is None:
            vmax = np.nanmax(data)

        assert vmin < vmax

        levels = np.linspace(vmin, vmax, 20)

        if logarithmic:
            locator = ticker.LogLocator(),  # log?
        else:
            locator = None

        if plot_in_dB:
            ref = vmax
            data = 20 * np.log10(np.divide(data, ref))
            vmin, vmax = 20 * np.log10(np.divide(np.array([vmin, vmax]), ref))
            levels = np.linspace(vmin, vmax, 20)

        if plottype == "contour":
            if plotdir == "x-y":
                cp = ax.contourf(self.grid[0][plot_slice], self.grid[1][plot_slice], data,
                                 cmap=cmap,
                                 locator=locator,
                                 levels=levels, **kwargs)

                xplot = self.grid[0][plot_slice].reshape(-1, 1)
                yplot = self.grid[1][plot_slice].reshape(-1, 1)
                zplot = data.reshape(-1, 1)

            elif plotdir == "x-z":
                cp = ax.contourf(self.grid[0][plot_slice], self.grid[2][plot_slice], data,
                                 cmap=cmap,
                                 locator=locator,
                                 levels=levels, **kwargs)
                xplot = self.grid[0][plot_slice].reshape(-1, 1)
                yplot = self.grid[2][plot_slice].reshape(-1, 1)
                zplot = data.reshape(-1, 1)
            elif plotdir == "y-z":
                cp = ax.contourf(self.grid[1][plot_slice], self.grid[2][plot_slice], data,
                                 cmap=cmap,
                                 locator=locator,
                                 levels=levels, **kwargs)
                xplot = self.grid[1][plot_slice].reshape(-1, 1)
                yplot = self.grid[2][plot_slice].reshape(-1, 1)
                zplot = data.reshape(-1, 1)
        else:
            if plotdir == "x-y":
                cp = plot_image_from_grid(ax, self.grid[0][plot_slice], self.grid[1][plot_slice], data,
                                          alpha=1, vmin=vmin, vmax=vmax, cmap=cmap)
            if plotdir == "x-z":
                cp = plot_image_from_grid(ax, self.grid[0][plot_slice], self.grid[2][plot_slice], data,
                                          alpha=1, vmin=vmin, vmax=vmax, cmap=cmap)
            if plotdir == "y-z":
                cp = plot_image_from_grid(ax, self.grid[1][plot_slice], self.grid[2][plot_slice], data,
                                          alpha=1, vmin=vmin, vmax=vmax, cmap=cmap)
        if outdir != None:
            with open(os.path.join(outdir, prefix_str + "sndfield.dat"), "w") as file:
                print("x\ty\tz", file=file)
                for plotx, ploty, plotz in zip(xplot, yplot, zplot):
                    print("%3.5f\t%3.5f\t%3.5f" % (plotx, ploty, plotz), file=file)

        # cp.logscale = True

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        cbar = fig.colorbar(cp, cax=cax)
        if not plot_in_dB:
            cbar.set_ticks([0, vmax], update_ticks=True)
            cbar.set_ticklabels(['0', '1'], update_ticks=True)
            cbar.ax.set_ylabel("Intensität / a.u.") #('Intensity / a.u.') #('Normalized Pressure')   ('Intensity / a.u.')
        else:
            cbar.ax.set_ylabel('Intensity / dB')  # ('Intensity / a.u.')
        return cp, cbar

    @staticmethod
    def calc_tof(grid, environment, rx_pos, tx_pos):
        """
        calculates the time of flight from source to receiver
        :param grid:
        :param tx_pos: location of your source
        :param rx_pos: location of your receiver
        :return:
        """
        if rx_pos is None:
            rx_pos = tx_pos

        c = environment.sound_speed
        tx_tof = np.sqrt((grid[0] - tx_pos.x) ** 2 +
                         (grid[1] - tx_pos.y) ** 2 +
                         (grid[2] - tx_pos.z) ** 2) / c
        rx_tof = np.sqrt((grid[0] - rx_pos.x) ** 2 +
                         (grid[1] - rx_pos.y) ** 2 +
                         (grid[2] - rx_pos.z) ** 2) / c
        tof = tx_tof + rx_tof
        return tof

    @staticmethod
    def calc_isochrones(grid, environment, tx_pos, ref, timespan, rx_pos=None):
        """

        calculates the time of flight (TOF) from source to receiver and compares it with the TOF to a reference point

        :param grid:
        :param environment: an SimpleEnvironment
        :param tx_pos: location of your source
        :param ref: reference point
        :param timespan
        :param rx_pos: location of your receiver
        :return:
        """
        assert issubclass(type(environment), SimpleEnvironment)

        tof = Sndfield.calc_tof(grid, environment, rx_pos, tx_pos)

        tof_ref = tof[find_nearest_gridpoint(grid, ref)]
        timerange = (tof_ref - timespan / 2, tof_ref + timespan / 2)

        mask = np.logical_and(tof > timerange[0], tof < timerange[1])
        return mask

    @staticmethod
    def get_fwhm_along_beam(point1, point2, grid, data, steps=400, length_lateral_crosssection=40e-3, debug=()):
        """
        Calculates the FWHM along the beam defined by point1 and point 2

        :param point1: Point object
        :param point2: Point object
        :param grid: meshgrid
        :param data: any data coresponding to grid
        :param steps: number of discretization steps
        :param length_lateral_crosssection: length of crosssection to evaluate
        :param debug: index number of points along a beam to plot the lateral Cross section in
        :return:
        """

        beam_coordinates = get_coordinates_cross_section(point1=point1, point2=point2, steps=steps, type='axial')
        fwhm_along_beam = []
        crossections = []

        lateral_res = min([p for p in get_grid_pitch(grid) if p > 0])

        for pos_idx, pos in enumerate(beam_coordinates[0:]):  # throw away first point
            orthogonal_line = get_coordinates_cross_section(point1, pos, length=length_lateral_crosssection,
                                                            steps=int(length_lateral_crosssection / lateral_res),
                                                            type='lateral')
            vi, dists = get_cross_section_at_points(orthogonal_line, grid, data, interpolation_method="linear")
            fwhm_along_beam.append(Sndfield.get_FWHM(dists, vi)[1])
            crossections.append(vi)
            if pos_idx in debug:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.axis([min(dists), max(dists), 0, np.nanmax(vi)])
                ax.plot(dists, vi)
                title = "cross section at " + str(pos)
                fig.suptitle(title)
                plt.show()
        return fwhm_along_beam, beam_coordinates, crossections, dists

    def plot_fwhm_along_beam(self, point=None, title="FWHM along beam", fig=None, ax=None, legend_title=None, debug=()):
        """
        Plot the FWHM along a beam between the txarray and point

        :param point: Point object
        :param title: title of the plot
        :return:
        """
        if legend_title is None:
            legend_title = ""
        if point is None:
            point = self.focus
        tx_pos = self.task.paudv_settings.tx.array.ref_element.position

        fwhm_along_beam, beam_coordinates, crossections, dists = self.get_fwhm_along_beam(tx_pos, point,
                                                                                          self.grid, np.abs(self.p_rms),
                                                                                          steps=100,
                                                                                          length_lateral_crosssection=40e-3,
                                                                                          debug=debug)
        if fig is None:
            fig = plt.figure("test")
            fig.suptitle(title)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Distance to center of transducer / m')
            ax.set_ylabel('FWHM / m')
        dists = []
        for pos in beam_coordinates:
            dists.append(np.sqrt((pos.x - tx_pos.x) ** 2 + (pos.y - tx_pos.y) ** 2 + (pos.z - tx_pos.y) ** 2))
        ax.plot(dists, fwhm_along_beam, label=legend_title)
        return fig, ax

    def get_beam_width_at_point(self, point=None, length=40e-3, steps=400):
        if point is None:
            point = self.focus

        tx_pos = self.task.paudv_settings.tx.array.ref_element.position
        orthogonal_line = get_coordinates_cross_section(tx_pos, point, length=length, steps=steps, type='lateral')
        vi, dists = get_cross_section_at_points(orthogonal_line, self.grid, abs(self.p_rms))
        return self.get_FWHM(dists, vi)[1]

    def __mul__(self, other):
        if isinstance(other, Sndfield):
            fact = other.p_rms
        else:
            fact = other
        return Sndfield(self.grid, self.p_rms * fact, self.task)

    def __add__(self, other):
        if isinstance(other, Sndfield):
            fact = other.p_rms
        else:
            fact = other
        return Sndfield(self.grid, self.p_rms + fact, self.task)

    def plot_draw_setup(self, ax, plot_slice):
        if self.env is not None:
            if isinstance(self.env, GriddedEnv):
                env = self.env
            else:
                env = self.env.discretize(self.grid)
            return env.plot(ax, plot_slice, hatched=False)

    def prepare_video(self, fig=None, ax=None, plot_slice=None,
                      tickdist=None,
                      cmap=cmap_make_transp(cmap=plt.get_cmap('viridis'), transp_center=0, transp_dist=0.1),
                      mindb=-20, maxdb=0, **kwargs):
        if maxdb > 0:
            logger.warning("maxdb should usually be <= 0")
        if mindb > maxdb:
            logger.warning("mindb should usually be < maxdb")

        if plot_slice is None:
            plot_slice = (slice(None), slice(None), self.grid[0].shape[2] // 2)  # the middle x-y plane

        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        data = np.abs((self.p[plot_slice + (slice(None),)]).hilbert())
        datamean = np.mean(data[...], axis=-1)

        def todb(data):
            ref = np.nanmax(data)
            return 20 * np.log10(np.divide(data, ref))

        def saturate_data(data, mindb, maxdb):
            data[data < mindb] = mindb
            data[data > maxdb] = maxdb
            return data

        data = saturate_data(todb(data), mindb, maxdb)
        datamean = saturate_data(todb(datamean), mindb, maxdb)

        x = self.grid[0][plot_slice]
        y = self.grid[1][plot_slice]
        z = self.grid[2][plot_slice]

        dims = x.shape
        singulardim = np.array([np.min(x) == np.max(x), np.min(y) == np.max(y), np.min(z) == np.max(z)])

        if np.count_nonzero(singulardim) != 1:
            logger.error("check plot slice {}: singulardim={} dims={}".format(plot_slice, singulardim, dims))

        if singulardim[0]:
            plotdir = "y-z"
            d0 = y
            d1 = z
        elif singulardim[1]:
            plotdir = "x-z"
            d0 = x
            d1 = z
        elif singulardim[2]:
            plotdir = "x-y"
            d0 = x
            d1 = y

        self.plot_draw_setup(ax, plot_slice)
        self.prepare_axis(ax, tickdist=tickdist, plotdir=plotdir)

        cp = plot_image_from_grid(ax, d0, d1, datamean,
                                  alpha=1, vmin=mindb, vmax=maxdb, cmap=cmap, **kwargs)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        cbar = fig.colorbar(cp, cax=cax)
        cbar.ax.set_ylabel('Intensity / dB')

        text = ax.text(0.02, 0.95, "", color="w", backgroundcolor="k", family='monospace', transform=ax.transAxes)

        return fig, ax, cp, cbar, text, data, datamean

    def create_video(self, videofile, data=None, datamean=None, plot=None, text=None, framerate=25,
                     plot_slice=None,
                     first_and_last_frame_is_mean=True, bitrate=4000,
                     **kwargs):
        if plot_slice is None:
            plot_slice = (slice(None), slice(None), self.grid[0].shape[2] // 2)  # the middle x-y plane

        if plot is None:
            fig, ax, plot, cbar, text, data, datamean = self.prepare_video(plot_slice=plot_slice, **kwargs)

        nframes = data.shape[-1]
        if first_and_last_frame_is_mean:
            nframes += 2

        def animate(frame):
            s = "mean"
            idx = frame - 1 if first_and_last_frame_is_mean else frame

            # fixme
            if first_and_last_frame_is_mean and (frame == 0 or frame == (nframes - 1)):
                d = datamean
            else:
                d = data[..., idx]
                try:
                    s = 't={:5.1f} μs'.format(data.t[idx] * 1e6)
                except AttributeError:
                    pass

            update_image_from_grid(plot, d)
            if text is not None:
                text.set_text(s)

            return plot, text,

        try:
            os.makedirs(os.path.dirname(videofile), exist_ok=True)
        except:
            pass
        fig = plot.figure
        anim = animation.FuncAnimation(fig, animate, frames=nframes, interval=1, blit=False)
        anim.save(videofile, fps=framerate, dpi=200, bitrate=bitrate)
        plt.close(fig)

    def slice(self, slice_tupel=None, x=None, y=None, z=None):
        """

        :param slice_tupel:
            * floats specify the index from 0.0 (first) to 1.0 (last)
        :param x:
        :param y:
        :param z:
        :return:
        """
        slice_tupel = list(slice_tupel) if slice_tupel is not None else [slice(None), slice(None), slice(None)]
        xyz = [x, y, z]
        xyzidx = find_nearest_gridpoint(self.grid, [a if a is not None else 0 for a in xyz])
        assert len(slice_tupel) == 3
        for i, s in enumerate(slice_tupel):
            if isinstance(s, float):  # floats specify the index from 0.0 (first) to 1.0 (last)
                idx = int(s * self.grid[i].shape[i])
                slice_tupel[i] = slice(idx, idx + 1)
            if xyz[i] is not None:
                idx = xyzidx[i]
                slice_tupel[i] = slice(idx, idx + 1)
        return Sndfield([g[tuple(slice_tupel)] for g in self.grid], task=self.task,
                        p=self.p[tuple(slice_tupel + [slice(None)])])

    def save_to_resultcache(self, path):
        # create result cache of this simulations results
        r = Data()
        r.fields["sndfield"] = self
        r.save_hdf5(path)
        return r

    @classmethod
    def __from_data_and_aux__(cls, data, aux):
        if "f_samp" in aux and "t_start" in aux:
            p = TimeSignal(data, aux["f_samp"], aux["t_start"])
        else:
            p = data
        env = aux["env"] if "env" in aux else None
        return Sndfield(aux["grid"], p=p, env=env)

    def __to_data_and_aux__(self):
        aux = {"grid": self.grid}
        aux["env"] = self.env
        if isinstance(self.p, TimeSignal):
            aux["f_samp"] = self.p.f_samp
            aux["t_start"] = self.p.t_start
        return np.array(self.p, copy=False), aux


# nice tick labels, this is a global function to make it pickleable
def tickfunc_milli(value, pos):
    return "{:3.1f}".format(value * 1000)
