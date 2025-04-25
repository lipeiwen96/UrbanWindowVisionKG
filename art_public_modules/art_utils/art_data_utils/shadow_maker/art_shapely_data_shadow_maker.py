from math import tan, radians
from functional import seq
from typing import *
from matplotlib.patches import Polygon
from shapely.geometry import Polygon, MultiPolygon, JOIN_STYLE, CAP_STYLE
from shapely import affinity
from shapely.ops import unary_union
from shapely.wkt import loads

from art_public_modules.art_data_structure.shapely.core_structure import DataElement, DataModel
from art_public_modules.art_data_structure.shapely.vector_2d import Vector2D
from art_public_modules.art_utils.geometry_utils.shapely.basic_utils import ShapelyBasicUtils
from art_public_modules.art_utils.geometry_utils.shapely.geometry_utils import ShapelyPolygonUtils


Coord = Tuple[float, float]
PROJECTION_ANGLE_DEGREE = 70
MATH_EPS = 1e-6
GEOMETRY_EMPTY = loads("GEOMETRYCOLLECTION EMPTY")


class SimpleShadowMaker:
    def __init__(self, shadow_vector: Vector2D, projection_angle_degree: float = PROJECTION_ANGLE_DEGREE):
        self._shadow_vector = shadow_vector.unit()
        self._projection_angle_degree: float = projection_angle_degree
        self._cache: Dict[str, Union[Polygon, MultiPolygon]] = {}

    def make_shadow_of(self, polygon: Polygon, height: float = 1) -> Polygon:
        polygon = polygon.buffer(0).simplify(0)
        if not ShapelyBasicUtils.is_valid(polygon):
            return GEOMETRY_EMPTY

        if height == 0 or self._projection_angle_degree == 0:
            return polygon
        shadow_vec: Vector2D = self._shadow_vector.multiply(height / tan(radians(self._projection_angle_degree)))

        coords_list: List[Coord] = []
        for polygon in ShapelyBasicUtils.flatten_and_filter(polygon, Polygon):
            coords_list.extend([polygon.exterior.coords] + [interior.coords for interior in polygon.interiors])
        shadows: List[Polygon] = []

        for coords in coords_list:
            coords = coords[:-1]
            projection_coords = seq(coords).map(lambda coord: (shadow_vec.apply_coord(coord))).to_list()
            n_coords = len(coords)

            for i in range(n_coords):
                piece = Polygon([coords[i - 1], coords[i], projection_coords[i], projection_coords[i - 1]])
                shadows.append(piece.buffer(distance=MATH_EPS,
                                            join_style=JOIN_STYLE.mitre,
                                            cap_style=CAP_STYLE.square))

        # noinspection PyTypeChecker
        shadow_pieces: List[Polygon] = sorted(ShapelyBasicUtils.flatten_and_filter(unary_union(shadows), Polygon),
                                              key=lambda piece: piece.area)

        return shadow_pieces[0].simplify(MATH_EPS) if shadow_pieces else GEOMETRY_EMPTY

    def make_edge_shadow_of(self, polygon: Polygon, depth: float = 1) -> Polygon:
        if not ShapelyBasicUtils.is_valid(polygon):
            polygon = ShapelyBasicUtils.normalize(polygon)
        return polygon.difference(self._shadow_vector.unit().multiply(depth).apply(polygon))


class ShadowMaker:
    @staticmethod
    def global_shadow_maker(data_model: DataModel, vector: Vector2D):
        if isinstance(data_model, DataModel) and isinstance(vector, Vector2D):
            shadow_maker = SimpleShadowMaker(shadow_vector=vector)
            for element in data_model.elements:
                element.shadow = shadow_maker.make_shadow_of(polygon=element.geometry, height=element.height) if isinstance(element.geometry, Polygon) else None
        else:
            raise ValueError('传入的参数类型无法识别')

    @staticmethod
    def element_shadow_maker(data_element: DataElement, vector: Vector2D):
        if isinstance(data_element.geometry, Polygon) and isinstance(vector, Vector2D):
            shadow_maker = SimpleShadowMaker(shadow_vector=vector)
            data_element.shadow = shadow_maker.make_shadow_of(polygon=data_element.geometry, height=data_element.height)
        else:
            raise ValueError('传入的参数类型无法识别或该data element中的geometry不是Shapely.Polygon对象')


class RectangleShadowMaker:
    @staticmethod
    def generate_rectangle_shadow(geometry: Polygon, horizental_shifting: float = 5, vertical_shifting: float = 5):
        """
        对输入的矩形，生成干净的阴影轮廓
        """
        if isinstance(geometry, Polygon) and len(geometry.exterior.coords) == 5:
            # 若输入图形为矩形，则进行阴影生成操作

            # 计算该矩形相应的属性
            centroid_point = geometry.centroid
            angle = ShapelyPolygonUtils.get_polygon_angle(geometry)
            # 将该矩形旋转至正交方向
            orthogonal_geometry = affinity.rotate(geometry, - angle, origin=centroid_point)
            minx, miny, maxx, maxy = orthogonal_geometry.bounds

            # 则该矩形的阴影图形为
            shadow_geometry = Polygon([[minx, maxy],
                                       [maxx, maxy],
                                       [maxx, miny],
                                       [maxx + horizental_shifting, miny + vertical_shifting],
                                       [maxx + horizental_shifting, maxy + vertical_shifting],
                                       [minx + horizental_shifting, maxy + vertical_shifting],
                                       [minx, maxy]])
            return affinity.rotate(shadow_geometry, angle, origin=centroid_point)
        else:
            print(f"[WARNING] 阴影生成模块读取到的图形:{geometry}不为矩形, 无法为其生成阴影，予以跳过")
