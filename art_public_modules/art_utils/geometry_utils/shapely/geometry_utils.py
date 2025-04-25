import math
from copy import deepcopy
from random import uniform
from typing import Union, List, Optional, Dict, Callable, Tuple
import numpy as np
from shapely.affinity import rotate, translate, scale, affine_transform
from shapely.geometry import (Polygon, Point, MultiPolygon, MultiLineString, LineString, MultiPoint, LinearRing,
                              shape, GeometryCollection)
from shapely.geometry.base import BaseGeometry, CAP_STYLE, JOIN_STYLE, BaseMultipartGeometry
from shapely.ops import unary_union, transform, substring
from shapely import affinity

from art_public_modules.art_data_structure.shapely.vector_2d import Vector2D
from art_public_modules.art_utils.geometry_utils.shapely.basic_utils import ShapelyBasicUtils


GEOMETRY_EMPTY = GeometryCollection()
MATH_EPS = 1e-6
Coord = Tuple[float, float]


class ShapelyPointUtils:
    @staticmethod
    def get_segment_pts(pt1: Point, pt2: Point, divide_seg_num: int = 2):
        segment_pts = []
        for i in range(1, divide_seg_num):
            segment_pts.append(Point((pt1.x * i + pt2.x * (divide_seg_num - i)) / divide_seg_num,
                                     (pt1.y * i + pt2.y * (divide_seg_num - i)) / divide_seg_num))
        return segment_pts


class ShapelyLineUtils:
    @staticmethod
    def get_intersection_of_line_segments(line1: LineString, line2: LineString, eps: float = MATH_EPS) -> Optional[Point]:
        if not (ShapelyBasicUtils.is_valid(line1) and ShapelyBasicUtils.is_valid(line2)):
            return None
        line1_coords = list(line1.coords)
        line2_coords = list(line2.coords)
        vec1 = Vector2D.from_coordinates(line1_coords[0], line1_coords[-1])
        vec2 = Vector2D.from_coordinates(line2_coords[0], line2_coords[-1])
        if abs(vec1.angle_degree - vec2.angle_degree) < eps:
            return None

        matrix = np.array([[vec1.y, -vec1.x],
                           [vec2.y, -vec2.x]])
        val_matrix = np.array([vec1.y * line1_coords[0][0] - vec1.x * line1_coords[0][1],
                               vec2.y * line2_coords[0][0] - vec2.x * line2_coords[0][1]])
        inverse_matrix = np.linalg.inv(matrix)
        solution = inverse_matrix @ val_matrix
        return Point(solution)

    @staticmethod
    def convert_linestring_to_polygon(line_string: LineString) -> Polygon:
        coords = line_string.coords
        coords.append(coords[0])
        return Polygon(coords)

    @staticmethod
    def simplify_line(outline: LineString or MultiLineString, simplification_tolerance, max_points) -> LineString:
        """
        简化多段线
        """
        outline_points = outline.coords
        simplification_updated = simplification_tolerance
        while len(outline_points) > max_points:
            # 简化过大的多边形，直到多边形点数小于max_points
            simplification_updated += simplification_tolerance
            outline_points = outline.simplify(simplification_updated).coords
        return LineString(outline_points)

    @staticmethod
    def complicate_line(geom: LineString, max_len: float) -> LineString:
        """
        复杂化多段线
        """
        points = []
        for previous, current in zip(geom.coords, geom.coords[1:]):
            line_segment = LineString([previous, current])
            # add points on line segment if necessary
            points.extend([
                line_segment.interpolate(line_segment.length / math.ceil(line_segment.length / max_len) * i).coords[0]
                for i in range(int(math.ceil(line_segment.length / max_len)))
            ])
            # finally, add end point
            points.append(current)
        # 删除相同坐标的点
        delete_dup_points = []
        for pre, cur in zip(points[:-1], points[1:]):
            if not pre == cur:
                delete_dup_points.append(pre)
            else:
                pass
        if points[-2] != points[-1]:
            delete_dup_points.append(points[-1])
        else:
            pass
        return LineString(delete_dup_points)

    @staticmethod
    def calculate_perpendicular_line(xj1: float, yj1: float, xi: float, yi: float, xi1: float, yi1: float) -> LineString:
        """
        求经过点p3(xj1, yj1), 与线段(xi, yi), (xi1, yi1)垂直的垂线垂足(xn, yn)
        """
        # 如果多线段(xi, yi), (xi1, yi1)垂直于X轴 --> xi1 = xi
        if xi1 - xi == 0:
            perpendicular_line = LineString([(xj1, yj1), (xi, yj1)])
        # 如果线段(xi, yi), (xi1, yi1)垂直于Y轴：--> yi1 = yi
        elif yi1 - yi == 0:
            perpendicular_line = LineString([(xj1, yj1), (xj1, yi)])
        # 如果线段(xi, yi), (xi1, yi1)是倾斜的线段：
        else:
            k = -((xi - xj1) * (xi1 - xi) + (yi - yj1) * (yi1 - yi)) / ((xi1 - xi) ** 2 + (yi1 - yi) ** 2) * 1.0
            xn = k * (xi1 - xi) + xi
            yn = k * (yi1 - yi) + yi
            perpendicular_line = LineString([(xj1, yj1), (xn, yn)])
        return perpendicular_line

    @staticmethod
    def calculate_parallel_line(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float, bounds: list) -> LineString:
        """
        计算过点(x0, y0)且平行于一条边PiPi1(x1,y1),(x2,y2)的平行线
        """
        # 该平行线起始点落在图形外接矩形上
        min_x, min_y, max_x, max_y = bounds
        # 如果多边形的边垂直于x轴：
        if x2 - x1 == 0:
            line = LineString([(x0, min_y), (x0, max_y)])
        # 如果多边形的边垂直于y轴：
        elif y2 - y1 == 0:
            line = LineString([(min_x, y0), (max_x, y0)])
        else:
            k = (y2 - y1) / (x2 - x1)
            yn1 = k * (min_x - x0) + y0
            yn2 = k * (max_x - x0) + y0
            line = LineString([(min_x, yn1), (max_x, yn2)])
        return line

    @staticmethod
    def line_equation(pt1: Point, pt2: Point) -> tuple:
        #  计算两个点的直线表达式，返回(state, k , b)
        #  若两点x坐标相等，则不存在斜率，state = 0, 表达式为 x = b
        if pt1.x == pt2.x:
            state = 0
            k = 0
            b = pt1.x
        #  若两点y坐标相等，存在斜率，state = 1, 表达式为 y = b
        elif pt1.y == pt2.y:
            state = 1
            k = 0
            b = pt1.y
        #  其余情况，均存在斜率，state = 1, 表达式为 y = kx + b
        else:
            state = 1
            k = (pt2.y - pt1.y) / (pt2.x - pt1.x)
            b = pt1.y - pt1.x * k
        return state, k, b

    @staticmethod
    def compute_line_normal_vector(pt1: Point, pt2: Point):
        state, k, b = ShapelyLineUtils.line_equation(pt1, pt2)
        if state == 0:
            return (1, 0), (-1, 0)
        else:
            if k == 0:
                return (0, 1), (0, -1)
            else:
                return (1, -1 / k), (-1, 1 / k)


class ShapelyPolygonUtils:
    @staticmethod
    def transform_triangle_to_parallelogram(triangle: Polygon) -> Polygon:
        """
        把三角形补充成平行四边形
        """
        # 先判断是不是三角形
        if len(triangle.exterior.coords) == 4:
            # 把三角形除斜边外的两条边找到（最短的两条边）
            all_side = []
            for i in range(len(triangle.exterior.coords) - 1):
                all_side.append(LineString([triangle.exterior.coords[i], triangle.exterior.coords[i + 1]]))
            all_side = sorted(all_side, key=lambda each: each.length)
            a = all_side[0]
            b = all_side[1]
            common_pt = a.intersection(b)  # 共有点
            pts_a = list(a.coords)
            pts_b = list(b.coords)
            anchor_pt_a = pts_a[1] if (Point(pts_a[0]).__eq__(common_pt)) else pts_a[0]
            anchor_pt_b = pts_b[1] if (Point(pts_b[0]).__eq__(common_pt)) else pts_b[0]
            # 求对应的点的坐标
            xd = anchor_pt_b[0] - (common_pt.x - anchor_pt_a[0])
            yd = anchor_pt_b[1] - (common_pt.y - anchor_pt_a[1])
            # 重新组合成polygon
            res_polygon = Polygon([common_pt, anchor_pt_a, (xd, yd), anchor_pt_b, common_pt]).minimum_rotated_rectangle
            return res_polygon
        else:
            return triangle

    @staticmethod
    def get_rectangle_angle(polygon: Polygon) -> float:
        """
        计算一个polygon的minimum rotated rectangle的旋转角度
        """
        rectangle = polygon.minimum_rotated_rectangle
        center = rectangle.centroid
        boundary = rectangle.exterior
        lines = []
        for i in range(len(boundary.coords) - 1):
            lines.append(LineString([boundary.coords[i], boundary.coords[i + 1]]))
        lines = sorted(lines, key=lambda each: each.length, reverse=True)[:2]
        valid_line = sorted(lines, key=lambda each: each.centroid.y)[0]
        if valid_line.centroid.x < center.x:
            a = LineString([(valid_line.bounds[0], valid_line.bounds[3]), (valid_line.bounds[0], valid_line.bounds[1])])
            angle = math.degrees(math.asin(a.length / valid_line.length))
        elif valid_line.centroid.x == center.x:
            angle = 0
        else:
            a = LineString([(valid_line.bounds[2], valid_line.bounds[3]), (valid_line.bounds[2], valid_line.bounds[1])])
            angle = -math.degrees(math.asin(a.length / valid_line.length))
        if angle == 90 or angle == -90:
            angle = 0
        return angle

    @staticmethod
    def scale_rectangle_by_distance(polygon: Polygon, x_dis: float = .0, y_dis: float = .0) -> Polygon:
        """
        按照长度来缩放polygon
        """
        # 先求该rectangle的角度
        angle = ShapelyPolygonUtils.get_rectangle_angle(polygon=polygon)
        # 把polygon旋转至平行位置
        polygon = rotate(polygon, angle=angle)
        # 每条边根据长度来缩放
        x_length = polygon.bounds[2] - polygon.bounds[0]
        y_length = polygon.bounds[3] - polygon.bounds[1]
        x_factor = (x_length + x_dis) / x_length
        y_factor = (y_length + y_dis) / y_length
        # 根据比例缩放一下
        polygon = scale(polygon, xfact=x_factor, yfact=y_factor, origin=polygon.centroid)
        # 旋转回去
        polygon = rotate(polygon, angle=-angle)
        return polygon

    @staticmethod
    def offset_polygon_by_distance(polygon: Polygon, direction: str = 'in', distance: float = 0):
        """
        对多边形进行偏移操作，direction = "in"/"out" 表示向多边形内部/外部偏移
        """
        buffer_part = polygon.exterior.buffer(distance, cap_style=2, join_style=2)
        if direction == 'in':
            res = unary_union([buffer_part, polygon])
            res = res.difference(buffer_part)
            return res
        elif direction == 'outside':
            return unary_union([buffer_part, polygon])
        else:
            raise ValueError('wrong parameters for direction.')

    @staticmethod
    def get_all_lines_from_polygon(polygon: Polygon) -> List[LineString]:
        """
        获得当前polygon的所有线段
        """
        all_pts = list(polygon.exterior.coords)
        all_lines = []
        for i in range(len(all_pts) - 1):
            all_lines.append(LineString([all_pts[i], all_pts[i + 1]]))
        return all_lines

    @staticmethod
    def jitter_outline_generator(polygon: Polygon, pts_distance: float = 0, min_move: float = 0, max_move: float = 0) -> Polygon:
        """
        对polygon的outline进行插值增加点，然后将这些点随机移动
        通常用于创造岸边、沙滩等
        """
        if isinstance(polygon, Polygon):
            outline = polygon.exterior
            pts_amount = int(outline.length / pts_distance)
            step = 1 / pts_amount
            pts = []
            for i in range(pts_amount):
                pts.append(outline.interpolate(i * step, normalized=True))
            for i in range(len(pts)):
                pts[i] = affine_transform(pts[i], [1, 0, 0, 1, uniform(min_move, max_move), uniform(min_move, max_move)])
            pts.append(pts[0])
            return Polygon(pts)
        else:
            raise ValueError('only accept Shapely Polygon Geometry')

    @staticmethod
    def get_polygon_angle(polygon: Polygon) -> float:
        """
        求任意polygon的角度
        角度的定义：最小外包矩形中所有长边中，中心点中y坐标最小的那条边与坐标轴的夹角
        """
        all_coords = list(polygon.exterior.coords)
        lines = [LineString([all_coords[i], all_coords[i + 1]]) for i in range(len(all_coords) - 1)]
        lines = list(filter(lambda each: each.intersects(
            LineString([(polygon.bounds[2], polygon.bounds[1]), (polygon.bounds[2], polygon.bounds[3])])), lines))
        valid_line = sorted(lines, key=lambda each: each.centroid.y)[0]

        pt1, pt2 = valid_line.coords
        if pt1[1] > pt2[1]:
            pt1, pt2 = pt2, pt1

        adjacent_side = LineString([pt1, (pt2[0], pt1[1], 0)])
        angle = math.degrees(math.acos(adjacent_side.length / valid_line.length))

        # 角度修正
        if angle > 45:
            angle -= 90

        return angle

    @staticmethod
    def find_rec_point(rec: Polygon, type: str = "left_below") -> Point:
        """
        找到任意矩形的任意端点坐标
        type: left_below | right_below | left_top | right_top
        """
        centroid_point = rec.centroid
        angle = ShapelyPolygonUtils.get_polygon_angle(rec)
        rotated_box = affinity.rotate(rec, - angle, origin=centroid_point)
        (minx, miny, maxx, maxy) = rotated_box.bounds
        if type == "left_below":
            return affinity.rotate(Point(minx, miny), angle, origin=centroid_point)
        elif type == "right_below":
            return affinity.rotate(Point(maxx, miny), angle, origin=centroid_point)
        elif type == "left_top":
            return affinity.rotate(Point(minx, maxy), angle, origin=centroid_point)
        else:
            return affinity.rotate(Point(maxx, maxy), angle, origin=centroid_point)

    @staticmethod
    def rotate_geometry_to_orthogonal_axis(geometry: Union[BaseGeometry, BaseMultipartGeometry], ref_point_type: str = "left_below"):
        """
        将任意形状的几何图形，根据其最小旋转矩形的偏移方向进行旋转，使其最小旋转矩形恢复为[正交方向]
        ref_point_type: left_below | right_below | left_top | right_top | center
        并依次返回：旋转后的几何图形
        """
        # 输入图形对应的最小旋转矩形
        rotate_rectangle = geometry.minimum_rotated_rectangle
        # 该最小旋转矩形的左下角参考点为
        if ref_point_type == "center":
            ref_point = geometry.centroid
        else:
            ref_point = ShapelyPolygonUtils.find_rec_point(rotate_rectangle, type=ref_point_type)
        # 该最小旋转矩形的旋转角度
        rotate_angle = ShapelyPolygonUtils.get_polygon_angle(rotate_rectangle)
        # 该图块的外轮廓图形还原至未旋转时
        original_geometry = rotate(deepcopy(geometry), - rotate_angle, origin=ref_point)
        return original_geometry


class ShapelyOpsUtils:
    @staticmethod
    def unary_union_with_eps(geom_list: List[BaseGeometry], eps: float = MATH_EPS):
        eps = abs(eps)
        buffered_geom_list = [geom.buffer(eps, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for geom in geom_list]
        unioned_geom = unary_union(buffered_geom_list)
        return ShapelyBasicUtils.normalize(unioned_geom.simplify(eps).buffer(-eps, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre))

