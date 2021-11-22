import collections
import copy
import itertools
import logging
import uuid
import joblib

logger = logging.getLogger(__name__)

from PAUDV.geometry import Parallelogram
from PAUDV.geometry.primitives import Orientation, Point
from PAUDV.hardware import abs_device


class Element(object):
    """
    single element of an array
    """

    def __init__(self, rel_position: Point, rel_surface):
        """
        :param rel_position: position of the element with respect to array
        :param rel_surface: a geometric primitive that represents the element surface
        """
        self.rel_position = rel_position
        self.rel_surface = rel_surface
        self.array = None
        self.uuid = uuid.uuid4()

    @property
    def position(self) -> Point:
        if self.array is None:
            raise ValueError("Element is not part of an array!")
        return self.array.to_global_pos(self.rel_position)

    def __hash__(self):
        return int(self.uuid)

    def __eq__(self, other: "Element"):
        return hash(self) == hash(other)

    def to_deterministic_element(self, array=None):
        clone = copy.deepcopy(self)
        clone.uuid = hash(self.position)
        if array is not None:
            clone.array = array
        return clone

    # TODO: remove compatibility handler
    def __setstate__(self, state):
        """
        Compatibility handler for pickled data without uuid
        :param state:
        """
        self.__dict__ = state
        if "uuid" not in self.__dict__:
            self.__dict__["uuid"] = uuid.uuid4()

    @property
    def surface(self):
        """
        :return:gives a surface representation in global coordinates
        """
        if self.array is None:
            raise ValueError("Element is not part of an array!")
        if self.rel_surface is None:
            return None
        return self.rel_surface.transform(self.array.global_orientation, self.position)

    def __str__(self):
        return 'Element of Array {}'.format(str(self.array))

    def __repr__(self):
        return "Element_{}".format(self.uuid)


class Array(abs_device.PAUDVDevice):
    """
    represents a phased array device
    """

    def __init__(self, elements):
        """
        :param elements: list of elements the array consists of
        :type elements: list
        """
        self.elements = elements
        self.global_position = None
        self.global_orientation = None
        self.control_hardware_channels = None
        self.fpga_channels = None

        for element in self.elements:
            element.array = self
        super().__init__(tuple(range(len(self.elements))), tuple(range(len(self.elements))))

        self.ref_element = Element(rel_position=self._get_ref_el_pos(),
                                   rel_surface=None)  # TODO determine surface of complete array
        self.ref_element.array = self

    def __iter__(self, *args, **kwargs):  # iterate over elements not ArraySlices with the length of 1
        return self.elements.__iter__()

    def __getitem__(self, item):
        """
        overwritten __getitem__ method to allow array slicing (a[start:stop:step])
        """
        if isinstance(item, int):
            self.elements[item]  # raise IndexError  if the el is not present
            return ArraySlice(self, tuple([item]))
        elif isinstance(item, slice):
            self.elements[item]  # raise IndexError  if the el is not present
            return ArraySlice(self, tuple(range(len(self.elements))[item]))
        elif isinstance(item, collections.Iterable):
            return ArraySlice(self, tuple(item))

    def __eq__(self, other: 'Array'):
        for el, other_el in zip(self.elements, other.elements):
            if el != other_el:
                return False
        return True

    def __hash__(self):
        return hash(tuple(hash(el) for el in self.elements))

    def __len__(self):
        return len(self.elements)

    def to_deterministic_array(self):
        clone = copy.copy(self)
        clone.elements = [el.to_deterministic_element(array=clone) for el in self.elements]
        clone.uuid = hash((self.global_position, self.global_orientation))
        clone.ref_element = clone.ref_element.to_deterministic_element(array=clone)
        clone.ref_element.array = clone
        return clone

    def get_control_hardware_channels(self):
        """
        get control hardware channels for all elements of the array (caching old data)
        """
        if self.control_hardware_channels is None:
            self.control_hardware_channels = self.get_channels(mode=abs_device.PAUDVDevice.control)
        return self.control_hardware_channels

    def get_center_pos(self, global_pos=True):
        """
        Returns the center between position of first and last element of the array as center
        of the array.
        :param global_pos: if True returns center position in global coordinates
        """
        elements = self.elements
        pos = (elements[0].rel_position + elements[-1].rel_position) / 2
        if global_pos is True:
            return self.to_global_pos(pos)
        else:
            return pos

    def _get_ref_el_pos(self):
        return self.get_center_pos(global_pos=False)

    def get_fpga_channels(self):
        """
        get fpga channels for all elements of the array (caching old data)
        """
        if self.fpga_channels is None:
            self.fpga_channels = self.get_channels(mode=abs_device.PAUDVDevice.fpga)
        return self.fpga_channels

    def get_element(self, con_id):
        """
        Returns element with the given connectors id.
        :param con_id: connector number for the element to get
        :type con_id: int
        :returns Element
        """
        return self.elements[con_id]

    def set_global_pos_and_ori(self, gl_pos, gl_ori):
        """
        define global position and orientation of the array
        :param gl_pos:
        :type gl_pos: Position
        :param gl_ori:
        :type gl_ori: Orientation
        """
        self.global_position = gl_pos
        self.global_orientation = gl_ori

    def to_global_pos(self, position):
        """
        Convert point given in local coordinates of the array to global coordinates.
        :param position: Position to convert
        :return: Position in global coordinates
        """
        if (self.global_position is None) or (self.global_orientation is None):
            raise AssertionError("Global position and/or global orientation of array {} not set.\n"
                                 "Call 'set_global_pos_and_ori' first!".format(str(self)))
        return self.global_orientation.transform(position) + self.global_position

    def get_elements_count(self):
        return len(self.elements)

    def __str__(self):
        return "Array(n_elements={})".format(self.get_elements_count())


class ComposedArray(Array):
    def __init__(self, *parents):
        self.parents = parents

        # parent indices list
        self.par_indices = [0]
        for parent in self.parents[:-1]:
            self.par_indices.append(self.par_indices[-1] + len(parent.elements))

        self.internal_map = None
        self.control_hardware_channels = None
        self.fpga_channels = None

        # create a new reference element which rel_position is relative to self.parent[0]
        self.ref_element = Element(rel_position=self._get_ref_el_pos(),
                                   rel_surface=self.elements[0].rel_surface)
        self.ref_element.array = self.parents[0]

    # in connection dict is created anew for every call of in_connections on a ComposedArray
    @property
    def in_connections(self):
        in_cons = {}
        idx = 0
        for parent in self.parents:
            par_in_cons = parent.in_connections
            if isinstance(parent, ArraySlice):
                indices = parent.indices
            else:
                indices = range(len(parent.elements))
            for par_index in indices:
                in_cons[idx] = par_in_cons[par_index]
                idx += 1
        return in_cons

    @property
    def global_position(self):
        return self.parents[0].global_position

    @property
    def global_orientation(self):
        return self.parents[0].global_orientation

    def get_channels(self, out_cons=None, mode="control"):
        # do mapping from internal out_con numbering of array slice to array
        parent_out_cons = [None] * len(self.parents)
        if out_cons is not None:
            parent_out_cons = [[] for idx in range(len(self.parents))]
            for out_con in out_cons:
                for par_idx in reversed(range(len(self.parents))):
                    if out_con >= self.par_indices[par_idx]:
                        parent_out_cons[par_idx].append(out_con - self.par_indices[par_idx])
                        break

        par_results = []
        for parent, par_out_cons in zip(self.parents, parent_out_cons):
            par_results.append(parent.get_channels(par_out_cons, mode))

        # merge individual parent results
        result = [[], []]
        for par_result in par_results:
            result[0].extend(par_result[0])
            result[1].extend(par_result[1])
        return result

    def get_center_pos(self, global_pos=True):
        """
        Returns the center between position of first and last element of the array as center
        of the array.
        :param global_pos: if True returns center position in global coordinates
        """
        pos = None
        for parent in self.parents:
            for element in parent.elements:
                if pos is None:
                    pos = element.position
                else:
                    pos += element.position
        pos /= len(self.elements)
        if global_pos is True:
            return pos
        else:
            return pos - self.parents[0].global_position

    def connect(self, other, output_cons_other=None, input_cons=None):
        raise NotImplementedError("Cannot add connection to a ComposedArray.")

    def set_global_pos_and_ori(self, global_position, global_orientation):
        raise RuntimeError("Can set global_pos and global_ori on a normal array only (not on ComposedArray).")

    @property
    def elements(self):
        return list(itertools.chain(*[parent.elements for parent in self.parents]))

    def to_global_pos(self, position, parent_id=0):
        """
        Convert point given in local coordinates of the array to global coordinates.
        :param position: Position to convert
        :return: Position in global coordinates
        """
        return self.parents[parent_id].to_global_pos(position)

    def __str__(self):
        return "ComposedArray(parents={})".format(str(self.parents))


class ArraySlice(Array):
    """
    Represents a part of an Array or ArraySlice (only containing a subset of elements).
    """

    def __init__(self, parent, indices):
        """
        :param parent: array this ArraySlice is a part of
        :type parent: Array
        :param indices: tuple of indices of elements of parent which are part of ArraySlice
        :type indices: tuple
        """
        self.parent = parent
        self.indices = indices
        self.input_cons = self.indices
        self.output_cons = self.indices
        self.internal_map = None
        self.control_hardware_channels = None
        self.fpga_channels = None

        self.ref_element = Element(rel_position=self._get_ref_el_pos(),
                                   rel_surface=self.elements[0].rel_surface)
        self.ref_element.array = self.parent

    def __iter__(self, *args, **kwargs):  # iterate over elements not ArraySlices with the length of 1
        return self.elements.__iter__(*args, **kwargs)

    def __getitem__(self, item):
        """
        overwritten __getitem__ method to allow array slicing (a[start:stop:step])
        """
        if isinstance(item, int):
            self.elements[item]  # raise IndexError  if the el is not present
            return ArraySlice(self.parent, tuple([self.indices[item]]))
        elif isinstance(item, slice):
            self.elements[item]  # raise IndexError  if the el is not present
            return ArraySlice(self.parent, tuple(self.indices[item]))
        elif isinstance(item, collections.Iterable):
            sel_elements = tuple([self.indices[x] for x in item])
            return ArraySlice(self.parent, sel_elements)

    def to_deterministic_array(self):
        clone = copy.copy(self)
        clone.parent = clone.parent.to_deterministic_array()
        clone.uuid = hash((self.global_position, self.global_orientation))
        clone.ref_element = clone.ref_element.to_deterministic_element(array=clone)
        return clone

    @property
    def global_position(self):
        return self.parent.global_position

    @property
    def global_orientation(self):
        return self.parent.global_orientation

    @property
    def in_connections(self):
        return self.parent.in_connections

    def get_channels(self, out_cons=None, mode="control"):
        # do mapping from internal out_con numbering of array slice to array
        if out_cons is not None:
            out_cons = [self.indices[slice_con] for slice_con in out_cons]
        return super().get_channels(out_cons, mode)

    @property
    def elements(self):
        if isinstance(self.indices, int):
            return [self.parent.elements[self.indices]]
        elif isinstance(self.indices, tuple):
            return [self.parent.elements[index] for index in self.indices]
        else:
            raise ValueError("Non proper indices object in ArraySlice: value: {}, type: {}".format(str(self.indices),
                                                                                                   type(self.indices)))

    def set_global_pos_and_ori(self, global_position, global_orientation):
        raise RuntimeError("Can set global_pos and global_ori on a normal Array only (not on ArraySlice).")

    def to_global_pos(self, position):
        """
        Convert point given in local coordinates of the array to global coordinates.
        :param position: Position to convert
        :return: Position in global coordinates
        """
        return self.parent.to_global_pos(position)

    def __str__(self):
        return "ArraySlice(parent={}, indices={})".format(str(self.parent), str(self.indices))


class CustomArray(Array):
    def __init__(self, offset_pos, WidthofElements, zero_in_the_middle=False):
        """Creates a custom array for simulating 1.5D array with different element sizes;
         no kerf or spacing between elements

        :param offset_pos: position of the first element in the array in array coords
        :param WidthofElements: tuple of with width of every element, no spacing between elements;
        :return:
        """

        if type(WidthofElements) in (float, int):
            WidthofElements = (WidthofElements,)

        if zero_in_the_middle:
            offset_pos = Point(offset_pos.x - sum(WidthofElements) / 2, offset_pos.y, offset_pos.z)

        elements = []
        for i in range(0, len(WidthofElements)):
            surface = Parallelogram(Point(-WidthofElements[i] / 2, 0, -2.5e-3),
                                    Point(-WidthofElements[i] / 2, 0, 2.5e-3),
                                    Point(WidthofElements[i] / 2, 0, -2.5e-3))
            displacement = Point(sum(WidthofElements[0:i]) + WidthofElements[i] / 2, 0, 0)
            elements.append(Element(offset_pos + displacement, surface))
        super().__init__(elements)


class LineArray(Array):
    def __init__(self, element_count, offset_pos, pitch, surface=None):
        """Creates an array whose elements are positioned on a straight line.

        :param element_count: count of elements contained in the array
        :type element_count: int
        :param offset_pos: position of the first element in the array in array coords,
                if None is given, the middle of the array is Point(0,0,0)
        :type offset_pos: Point or None
        :param pitch: distance between the elements
        :type pitch: Point

        >>> l = LineArray(3, Point(0,0,0), Point(1,0,0))
        >>> [e.rel_position for e in l.elements]
        [Point(0, 0, 0), Point(1, 0, 0), Point(2, 0, 0)]
        >>> l = LineArray(3, None, Point(1,0,0))
        >>> [e.rel_position for e in l.elements]
        [Point(-1, 0, 0), Point(0, 0, 0), Point(1, 0, 0)]

        """
        elements = []

        if offset_pos is None:
            offset_pos = Point(0, 0, 0, ) - (pitch * (element_count - 1)) / 2

        # if surface is None:
        #        surface = Parallelogram(Point(-0.25e-3, 0, -2.5e-3), Point(0.25e-3, 0, -2.5e-3), Point(0.25e-3, 0, -2.5e-3))
        # fixme: use pitch for estimation # no dimensions specified assume a thin, seamless element with zero height

        for index in range(element_count):
            elements.append(Element(offset_pos + pitch * index, surface))
        super().__init__(elements)


class pointSource(Array):
    """
    """

    def __init__(self, diameter=2e-6):
        elements = []
        surface = Parallelogram(Point(-diameter / 2, 0, -1e-3),
                                Point(-diameter / 2, 0, 1e-3),
                                Point(diameter / 2, 0, -1e-3))
        displacement = Point(diameter / 2, 0, 0)
        elements.append(Element(displacement, surface))
        super().__init__(elements)
