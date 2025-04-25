from typing import Union, List, Optional, Dict, Callable, Tuple
from shapely.geometry import (Polygon, Point, MultiPolygon, MultiLineString, LineString, MultiPoint, LinearRing,
                              shape, GeometryCollection)
from shapely.geometry.base import BaseGeometry, CAP_STYLE, JOIN_STYLE, BaseMultipartGeometry
from shapely import wkt
from art_public_modules.art_utils.geometry_utils.shapely.basic_utils import ShapelyBasicUtils


Coord = Tuple[float, float]


class PolygonSmoother:
    def __init__(self, n_iter: int = 2):
        self._n_iter = n_iter

    @staticmethod
    def _interpolate(pts: List[Coord], rate: float = 0.25) -> List[Coord]:
        new_pts: List[Coord] = []
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[(i - 1) % n]
            x2, y2 = pts[i]
            x3, y3 = pts[(i + 1) % n]
            new_pt1: Coord = ((1 - rate) * x2 + rate * x1, (1 - rate) * y2 + rate * y1)
            new_pt2: Coord = ((1 - rate) * x2 + rate * x3, (1 - rate) * y2 + rate * y3)
            new_pts.append(new_pt1)
            new_pts.append(new_pt2)
        return new_pts

    def bezier_smooth(self, polygon: Polygon, rate=0.25) -> List[Polygon]:
        if rate >= 0.5:
            raise ValueError("rate must be < 0.5")
        if not isinstance(polygon, Polygon):
            raise TypeError(f'expect polygon, given {polygon}')

        coords = list(polygon.exterior.coords[:-1])
        for i in range(self._n_iter):
            coords = PolygonSmoother._interpolate(coords, rate)
        try:
            return [Polygon(coords)]
        except Exception:
            return []

    def _buffer(self, polygon: Polygon,
                dist: float,
                cap_style: int,
                join_style: int,
                mitre_dist: float) -> BaseGeometry:
        return polygon.buffer(distance=dist,
                              cap_style=cap_style,
                              join_style=join_style,
                              mitre_limit=mitre_dist)

    def buffer_smooth(self,
                      polygon: Polygon,
                      cap_style=CAP_STYLE.flat,
                      join_style=JOIN_STYLE.mitre,
                      mitre_dist: float = 5,  # default as shapely does
                      buffer_distance: float = 5,
                      buffer_decay_ratio: float = 0.9,
                      ) -> List[Polygon]:
        if not isinstance(polygon, Polygon):
            raise TypeError(f'expect polygon, given {polygon}')
        for _ in range(self._n_iter):
            for factor in [-1, 1]:
                polygon = self._buffer(polygon=polygon,
                                       dist=factor * buffer_distance,
                                       cap_style=cap_style,
                                       join_style=join_style,
                                       mitre_dist=mitre_dist)

            buffer_distance *= buffer_decay_ratio

        # noinspection PyTypeChecker
        return ShapelyBasicUtils.flatten_and_filter(polygon, Polygon)


if __name__ == "__main__":
    out = PolygonSmoother().bezier_smooth(wkt.loads("POLYGON ((838548 816957, 838558 816913, 838560 816914, 838658 816933, 838659 816934, 838664 816936, 838665 816940, 838665 816941, 838661 816978, 838658 816979, 838548 816957))"))
    print(out[0])