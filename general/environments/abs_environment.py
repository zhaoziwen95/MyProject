import abc

from PAUDV.geometry import Point


class Environment(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def t_sound_travel(self, pos1: Point, pos2: Point) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def to_complex_env(self):
        raise NotImplementedError

    def discretize(self, grid):
        return self.to_complex_env().discretize(grid)

    def plot(self, ax, **kwargs):
        # call additional plot
        pass

    def __str__(self):
        return "{}".format(str(self.__class__.__name__))

    def __repr__(self):
        return str(self)
