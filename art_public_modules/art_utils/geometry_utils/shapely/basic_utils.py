import math
from itertools import chain, accumulate
from typing import Union, List, Optional, Dict, Callable, Tuple
from shapely.geometry import (Polygon, Point, MultiPolygon, MultiLineString, LineString, MultiPoint, LinearRing,
                              shape, GeometryCollection)
from shapely.geometry.base import BaseGeometry, CAP_STYLE, JOIN_STYLE, BaseMultipartGeometry
from shapely.ops import unary_union, transform


GEOMETRY_EMPTY = GeometryCollection()
MATH_EPS = 1e-6
Coord = Tuple[float, float]


class ShapelyBasicUtils:
    @staticmethod
    def geometry_flatten(geom: BaseGeometry) -> List[BaseGeometry]:
        if not isinstance(geom, BaseGeometry):
            return []
        if isinstance(geom, (Polygon, LineString, LinearRing, Point)):
            return [geom]
        return list(chain.from_iterable(ShapelyBasicUtils.geometry_flatten(sub_geom) for sub_geom in geom))

    @staticmethod
    def flatten_and_filter(geom: BaseGeometry,
                           target_clz: Union[type, Tuple[type]],
                           need_valid: bool = True) -> List[BaseGeometry]:
        if not isinstance(geom, BaseGeometry):
            raise TypeError(f'expect shapely geometry obj, given {geom}')
        return [g for g in ShapelyBasicUtils.geometry_flatten(geom) if isinstance(g, target_clz) and need_valid and ShapelyBasicUtils.is_valid(geom)]

    @staticmethod
    def is_similar(geom1: BaseGeometry, geom2: BaseGeometry, eps: float = MATH_EPS) -> bool:
        if geom1.is_empty and geom2.is_empty:
            return True
        if type(geom1) is not type(geom2):
            return False
        if isinstance(geom1, (Polygon, MultiPolygon)):
            return geom1.symmetric_difference(geom2).area < eps
        elif isinstance(geom1, (LineString, MultiLineString, Point, MultiPoint)):
            buffered_line1 = geom1.buffer(eps, cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.mitre)
            buffered_line2 = geom2.buffer(eps, cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.mitre)
            return buffered_line1.contains(geom2) or buffered_line2.contains(geom1)
        else:  # GeometryCollection
            geoms_of_geom1 = list(geom1)
            geoms_of_geom2 = list(geom2)
            if len(geoms_of_geom1) != len(geoms_of_geom2):
                return False
            points1 = unary_union([geom for geom in geoms_of_geom1 if isinstance(geom, Point)])
            lines1 = unary_union([geom for geom in geoms_of_geom1 if isinstance(geom, LineString)])
            polygon1 = unary_union([geom for geom in geoms_of_geom1 if isinstance(geom, Polygon)])

            points2 = unary_union([geom for geom in geoms_of_geom2 if isinstance(geom, Point)])
            lines2 = unary_union([geom for geom in geoms_of_geom2 if isinstance(geom, LineString)])
            polygon2 = unary_union([geom for geom in geoms_of_geom2 if isinstance(geom, Polygon)])

            return ShapelyBasicUtils.is_similar(points1, points2) and ShapelyBasicUtils.is_similar(lines1, lines2) and ShapelyBasicUtils.is_similar(polygon1, polygon2)

    @ staticmethod
    def remove_hole_in_geometry(geometry: BaseGeometry, area_limit):
        if not geometry.is_valid or geometry.is_empty:
            return geometry
        if isinstance(geometry, BaseMultipartGeometry):
            return unary_union([ShapelyBasicUtils.remove_hole_in_geometry(r, area_limit) for r in geometry.geoms])
        elif isinstance(geometry, Point) or isinstance(geometry, LineString):
            return geometry
        elif isinstance(geometry, Polygon):
            # polygon
            candidates = [Polygon(part) for part in list(geometry.interiors)]
            return unary_union([geometry] + [c for c in candidates if c.area <= area_limit])
        else:
            return geometry

    @staticmethod
    def is_valid(geom: BaseGeometry, non_empty: bool = True):
        if not isinstance(geom, BaseGeometry):
            return False
        if not geom.is_valid or (geom.is_empty and non_empty):
            return False
        return True

    @staticmethod
    def radius_of(geom: BaseGeometry) -> float:
        if isinstance(geom, Point):
            return 0
        elif isinstance(geom, Polygon):
            min_x, min_y, max_x, max_y = geom.bounds
            width = max_x - min_x
            height = max_y - min_y
            return math.sqrt(width ** 2 + height ** 2)
        elif isinstance(geom, BaseGeometry):
            return ShapelyBasicUtils.radius_of(geom.envelope)
        return 0

    @staticmethod
    def normalize(geom: BaseGeometry, ccw: bool = False):
        if not isinstance(geom, BaseGeometry):
            raise ValueError(f'expect shapely geometry, given {type(geom)}')
        if geom.is_empty:
            return geom
        if geom.is_valid:
            if ccw:
                def normalize_ring(ring: LinearRing, ccw: bool) -> LinearRing:
                    flag = ring.is_ccw == ccw
                    return LinearRing(list(ring.coords) if flag else list(ring.coords)[::-1])

                if isinstance(geom, LinearRing):
                    geom = normalize_ring(geom, ccw=True)

                elif isinstance(geom, Polygon):
                    exterior = normalize_ring(geom.exterior, ccw=True)
                    interiors = [normalize_ring(interior, ccw=False) for interior in geom.interiors]
                    geom = Polygon(shell=exterior, holes=interiors)
            return geom
        if isinstance(geom, BaseMultipartGeometry):
            sub_geoms = [ShapelyBasicUtils.normalize(g) for g in ShapelyBasicUtils.geometry_flatten(geom)]
            if all(isinstance(g, LineString) for g in sub_geoms):  # are sub geoms all lines
                geom = MultiLineString(sub_geoms)
            elif all(isinstance(g, Polygon) for g in sub_geoms):  # area sub geoms all polygons
                polys = [ShapelyBasicUtils.normalize(poly) for poly in ShapelyBasicUtils.geometry_flatten(GeometryCollection(sub_geoms).buffer(0)) if
                         isinstance(poly, Polygon)]
                geom = MultiPolygon(polys)
            elif all(isinstance(g, Point) for g in sub_geoms):  # area sub geoms all points
                geom = MultiPoint(sub_geoms)
            else:
                polys = []
                geoms = []
                for geometry in sub_geoms:
                    if isinstance(geometry, Polygon):
                        polys.append(geometry)
                    else:
                        geoms.append(geometry)
                geoms.extend([ShapelyBasicUtils.normalize(geometry) for geometry in ShapelyBasicUtils.geometry_flatten(GeometryCollection(polys).buffer(0))])
                geom = GeometryCollection(geoms)

        elif isinstance(geom, Polygon):
            parts = [geom for geom in ShapelyBasicUtils.geometry_flatten(geom.buffer(0)) if isinstance(geom, Polygon)]
            largest_part = max(parts, key=lambda g: g.area, default=GeometryCollection())
            geom = ShapelyBasicUtils.normalize(largest_part, ccw=ccw)

        elif isinstance(geom, LineString):
            # 此时的LineString必定是invalid，且所有的点都是同一个点
            coords = list(geom.coords)
            if len(coords) > 0:
                geom = Point(coords[0])
            else:
                geom = LineString()

        return geom if geom.is_valid else GEOMETRY_EMPTY

    # def concave_idxs(coords: List[Coord], simplify_eps: float = 0) -> List[int]:
    #     linearring = LinearRing(coords).simplify(simplify_eps)
    #     coords = list(linearring.coords) if linearring.is_ccw else list(linearring.coords)[::-1]
    #     coords.pop()  # first and last coord are the same, remove the last one
    #
    #     concave_idxs: List[int] = []
    #     for i, cur in enumerate(coords):
    #         pre = coords[i - 1]
    #         next_ = coords[(i + 1) % len(coords)]
    #         pre_vec = Vector2D.from_coordinates(pre, cur)
    #         next_vec = Vector2D.from_coordinates(cur, next_)
    #
    #         ccw_angle: float = pre_vec.angle_ccw_rotating_to(next_vec)
    #         if ccw_angle > 180:
    #             concave_idxs.append(i)
    #
    #     return concave_idxs

    # TODO only using @future version
    # def make_less_accurate(geometry: BaseGeometry, digits: int = 3) -> BaseGeometry:
    #     return transform(lambda x, y, z=None: (round(x, digits), round(y, digits)), geometry)
    #
    # TODO only using @future version
    # def wkt_loads(wkt_str: str) -> Optional[BaseGeometry]:
    #     if not isinstance(wkt_str, str):
    #         return None
    #     valid_wkt_prefixes = {'POINT', 'MULTIPOINT', 'LINEARRING', 'LINESTRING', 'MULTILINESTRING', 'POLYGON',
    #                           'MULTIPOLYGON',
    #                           'GEOMETRYCOLLECTION'}
    #     for valid_wkt_prefix in valid_wkt_prefixes:
    #         if wkt_str[:len(valid_wkt_prefix)].upper() == valid_wkt_prefix:
    #             try:
    #                 geom = shapely_wkt_loads(wkt_str)
    #                 return geom
    #             except Exception:
    #                 break
    #     return None
    #
    # TODO only using @future version
    # def wkb_loads(wkb_bytes: str) -> Optional[BaseGeometry]:
    #     if not isinstance(wkb_bytes, bytes):
    #         return None
    #     try:
    #         geom = shapely_wkb_loads(wkb_bytes)
    #         return geom
    #     except Exception:
    #         return None
    #
    # TODO only using @future version
    # def geojson_load(geojson: dict) -> Optional[BaseGeometry]:
    #     if not isinstance(geojson, dict):
    #         return None
    #
    #     valid_types = {'Point', 'MultiPoint', 'LineString', 'LinearRing', 'MultiLineString',
    #                    'Polygon', 'MultiPolygon', 'GeometryCollection'}
    #     has_valid_type = (geojson.get('type', None) in valid_types)
    #     if not has_valid_type:
    #         return None
    #
    #     has_valid_coordinates = isinstance(geojson.get('coordinates', None), (tuple, list))
    #     has_valid_geometries = isinstance(geojson.get('geometries', None), (list, tuple))
    #     if not has_valid_coordinates and not has_valid_geometries:
    #         return None
    #
    #     try:
    #         geom = shape(geojson)
    #         return geom
    #     except Exception:
    #         return None
    #
    # TODO only using @future version
    # def load_geom(geom_data: Union[dict, bytes, str]) -> Optional[BaseGeometry]:
    #     """ loads geojson, wkt or wkb into shapely geometry """
    #     if isinstance(geom_data, BaseGeometry):
    #         return geom_data
    #
    #     if isinstance(geom_data, dict) and geom_data.get("type", None) == "LinearRing":
    #         geom_data["type"] = "LineString"  # LinearRing cannot be recognized, change it to LineString
    #
    #     type_loader_mapping: Dict[type, Callable] = {dict: geojson_load, str: wkt_loads, bytes: wkb_loads}
    #
    #     if loader := type_loader_mapping.get(type(geom_data), None):
    #         geom = loader(geom_data)
    #         return geom
    #
    #     return None

    # TODO only using @future version
    # def geometry_multiply(geoms: List[BaseGeometry], ring_to_line: bool = True) -> BaseMultipartGeometry:
    #     if not isinstance(geoms, Iterable):
    #         raise TypeError(f'expect list of geometry, given {geoms}')
    #
    #     geoms: List[BaseGeometry] = seq(geoms).flat_map(geometry_flatten).filter(lambda geom: is_valid(geom)).to_list()
    #     geoms = [(LineString(g) if ring_to_line and isinstance(g, LinearRing) else g) for g in geoms]
    #
    #     if not geoms:
    #         return GEOMETRY_EMPTY
    #
    #     in_same_type = len(seq(geoms).group_by(type).to_dict()) == 1
    #     if not in_same_type:
    #         return GeometryCollection(geoms)
    #
    #     type_ = type(geoms[0])
    #     multi_type = {Point: MultiPoint,
    #                   LineString: MultiLineString,
    #                   LinearRing: MultiLineString,
    #                   Polygon: MultiPolygon}.get(type_)
    #
    #     if not multi_type:
    #         return GEOMETRY_EMPTY
    #
    #     return multi_type(geoms)
