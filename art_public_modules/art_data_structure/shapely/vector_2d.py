import math
from typing import Sequence, TypeVar, Tuple, Union

from shapely.affinity import translate
from shapely.geometry import Point, LineString
from shapely.geometry.base import BaseGeometry


Num = TypeVar('Num', float, int)
Coord = Tuple[float, float]


class Vector2D:
    def __init__(self, x, y):
        if not isinstance(x, (float, int)) or not isinstance(y, (float, int)):
            raise TypeError("x, y should be number")
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x},{self.y})"

    @staticmethod
    def is_valid_2d_coordinate(coord: Sequence[Num]):
        return (isinstance(coord, Sequence)
                and len(coord) >= 2
                and isinstance(coord[0], (int, float))
                and isinstance(coord[1], (int, float)))

    @classmethod
    def from_coordinate(cls, coord: Sequence[Num]):
        if not cls.is_valid_2d_coordinate(coord):
            raise ValueError(f"{coord} is not valid coordinates")
        return cls(x=coord[0], y=coord[1])

    @classmethod
    def from_coordinates(cls, from_coord: Sequence[Num], to_coord: Sequence[Num]):
        if not cls.is_valid_2d_coordinate(from_coord):
            raise ValueError(f"{from_coord} is not valid from_coordinates")
        if not cls.is_valid_2d_coordinate(to_coord):
            raise ValueError(f"{to_coord} is not valid from_coordinates")
        return cls(x=to_coord[0] - from_coord[0], y=to_coord[1] - from_coord[1])

    @classmethod
    def from_points(cls, from_point: Point, to_point: Point):
        from_coord = from_point.coords[0]
        to_coord = to_point.coords[0]
        return cls.from_coordinates(from_coord=from_coord, to_coord=to_coord)

    @classmethod
    def from_angle(cls, angle_degree: float = 0, length: float = 1):
        radian = math.radians(angle_degree)
        to_x: float = math.cos(radian) * length
        to_y: float = math.sin(radian) * length
        return cls.from_coordinates((0, 0), (to_x, to_y))

    @classmethod
    def from_linestring(cls, linestring: LineString):
        if not linestring.is_valid or linestring.is_empty:
            raise ValueError("input linestring is invalid or empty")
        coords = list(linestring.coords)
        return cls.from_coordinates(coords[0], coords[-1])

    @property
    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def angle_degree(self):
        """ in range of [0, 360) """
        return (360 + math.degrees(math.atan2(self.y, self.x))) % 360

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @classmethod
    def raise_if_not_vector(cls, possible_vector):
        if not isinstance(possible_vector, cls):
            raise TypeError(f"{possible_vector} is of type {type(possible_vector)}, expect Vector2D")

    def plus(self, vector):
        self.raise_if_not_vector(vector)
        return Vector2D(self.x + vector.x, self.y + vector.y)

    def dot(self, vector) -> Num:
        self.raise_if_not_vector(vector)
        return self.x * vector.x + self.y * vector.y

    def angle_ccw_rotating_to(self, vector_2d, in_degree: bool = True):
        """
        in range of [0, 360] degree
        """
        dot_prod = self.dot(vector_2d)
        cos_val = max(-1, min(dot_prod / self.length / vector_2d.length, 1))
        angle_in_radian = math.acos(cos_val)
        cross_prod = self.x * vector_2d.y - self.y * vector_2d.x
        if cross_prod < 0:  # then vector_2d is on the cw side of self vector
            angle_in_radian = math.pi * 2 - angle_in_radian
        if in_degree:
            return math.degrees(angle_in_radian)
        return angle_in_radian

    def angle_to(self, vector_2d, in_degree: bool = True):
        """
        in range of [0, 180]
        """
        ccw_rotating_angle_degree = self.angle_ccw_rotating_to(vector_2d)
        if ccw_rotating_angle_degree > 180:
            ccw_rotating_angle_degree = 360 - ccw_rotating_angle_degree
        if not in_degree:
            return math.radians(ccw_rotating_angle_degree)
        return ccw_rotating_angle_degree

    def apply(self, geom: BaseGeometry) -> BaseGeometry:
        return translate(geom, xoff=self.x, yoff=self.y)

    def ray(self, origin: Union[Point, Coord], length: float = 1e8) -> LineString:  # default length is large enough
        if isinstance(origin, Point):
            origin = list(origin.coords)[0]

        vector = self.unit().multiply(length)
        return LineString([origin, vector.apply_coord(origin)])

    def apply_coord(self, coord: Tuple[float, float]) -> Tuple[float, float]:
        return coord[0] + self.x, coord[1] + self.y

    def multiply(self, multiple: float):
        return Vector2D(self.x * multiple, self.y * multiple)

    def __mul__(self, other):
        return self.multiply(other)

    def unit(self):
        length = math.sqrt(self.x ** 2 + self.y ** 2)
        if length == 0:
            raise ValueError('x and y cannot be both 0')
        return Vector2D(self.x / length, self.y / length)

    @property
    def ccw_perpendicular_vector(self):
        perpendicular_vector = Vector2D.from_coordinate([-self.y, self.x])
        return perpendicular_vector

    def reverse(self):
        return Vector2D(-self.x, -self.y)

    @property
    def slope(self):
        return self.y / self.x if self.x != 0 else float('inf')
