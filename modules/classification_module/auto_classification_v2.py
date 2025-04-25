"""
自动识别土地分类和地块有关的计算细节，计算输入地块的土地分类
算法V2版，使用HK提供的策略
"""
import copy
import math
from dataclasses import field, dataclass
from typing import List
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, Point, LineString, GeometryCollection, box, MultiPolygon, MultiPoint, MultiLineString
from map_system.coord_convertor.coords_convertor import CoordConvertor
import time
import json
import os
from map_system.tile_genenrator.map_structure import MapRoadCenterLine, MapBuilding, MapLot, MapRoadPolygon
from map_system.tile_genenrator.map_cropper import MapCropper
from modules.classification_module.valid_site import SitePoint, SiteSegment, SpecifiedRoad, DetectedLine
from shapely.ops import unary_union


SIMPLIFIED_POINT_DISTANCE = 5
SAMPLE_DISTANCE = 0.5
CHECK_DISTANCE = 4.5
CORNER_ANGLE = 140
SEG_MIN_DIS = 2.5


@dataclass
class SiteEdge:
    geometry: LineString = field(default_factory=LineString)
    start_pt: Point = field(default_factory=Point)
    end_pt: Point = field(default_factory=Point)
    normal_vector: list = field(default=list)  # 法向量
    sample_pts: List[SitePoint] = field(default_factory=list)

    def init(self, start_pt: Point, end_pt: Point, site_boundary: Polygon):
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.geometry = LineString([start_pt, end_pt])
        # 法向量
        self.normal_vector = AutoUtils.get_normal_vector(start_pt, end_pt, site_boundary)

        # 生成所有的采样点
        direction_vector = (end_pt.x - start_pt.x, end_pt.y - start_pt.y)
        direction_magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
        direction_unit_vector = (direction_vector[0] / direction_magnitude, direction_vector[1] / direction_magnitude)
        sample_pts_num = math.floor(self.geometry.length / SAMPLE_DISTANCE)
        pt_id = 0
        for i in range(sample_pts_num):
            if i == 0:
                sample_pt = SitePoint(id=0, geometry=self.start_pt, is_start=True, is_edge_point=True)
                self.sample_pts.append(sample_pt)
                pt_id += 1
            elif i * SAMPLE_DISTANCE > self.geometry.length - SAMPLE_DISTANCE/2:
                break
            else:
                x_sample = self.start_pt.x + i * SAMPLE_DISTANCE * direction_unit_vector[0]
                y_sample = self.start_pt.y + i * SAMPLE_DISTANCE * direction_unit_vector[1]
                sample_pt = SitePoint(id=i, geometry=Point(x_sample, y_sample))
                self.sample_pts.append(sample_pt)
                pt_id += 1
        sample_pt = SitePoint(id=pt_id, geometry=self.end_pt, is_end=True, is_edge_point=True)
        self.sample_pts.append(sample_pt)

        # 生成所有的检测点（偏移4.5m）
        for sample_pt in self.sample_pts:
            sample_pt.check_pt = Point(sample_pt.geometry.x + self.normal_vector[0] * CHECK_DISTANCE,
                                       sample_pt.geometry.y + self.normal_vector[1] * CHECK_DISTANCE)


@dataclass
class SiteEdgeSegment:
    name: str = field(default=str)
    geometry: LineString = field(default_factory=LineString)
    start_pt: Point = field(default_factory=Point)
    end_pt: Point = field(default_factory=Point)
    abut_larger_street: bool = field(default=False)
    start_pt_is_corner: bool = field(default=False)
    normal_vector: list = field(default=list)  # 法向量
    detected_line: LineString = field(default_factory=LineString)
    # 给旭东用的属性，更新窗间距线
    abut_small_street_status: bool = field(default=False)


@dataclass
class AutoClassificationV2:
    site_boundary: Polygon = field(default_factory=Polygon)
    simplified_site_boundary: Polygon = field(default_factory=Polygon)
    project_boundary: Polygon = field(default_factory=Polygon)
    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    crossed_road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    # 相交道路图形的Union
    crossed_pavement_polygon_union: BaseGeometry = field(default_factory=BaseGeometry)
    edges: List[SiteEdge] = field(default_factory=list)
    site_pts: List[SitePoint] = field(default_factory=list)
    segs: List[SiteEdgeSegment] = field(default_factory=list)
    has_corner: bool = field(default=False)
    valid_perimeter: float = field(default=float)

    @property
    def segs_dict(self):
        outcome = []
        for seg in self.segs:
            outcome.append({
                "geometry": seg.geometry.wkt,
                "abut_small_street_status": seg.abut_small_street_status,
            })
        return outcome

    def init(self, site_boundary: Polygon, project_boundary: Polygon,
             road_center_line_list: List[MapRoadCenterLine], road_polygon_list: List[MapRoadPolygon]):
        self.site_boundary = site_boundary
        self.project_boundary = project_boundary
        self.road_center_line_list = copy.deepcopy(road_center_line_list)
        self.road_polygon_list = copy.deepcopy(road_polygon_list)

        print("---【土地分类自动计算模块·V2】---")
        print(f"- 启动模块，原始的输入地块图形为:")
        print(self.site_boundary)

        # 1,获取周围的Union pavement polygon
        self.__get_all_pavement_polygon()

        # 2,简化场地边界，消除影响的短边
        self.__simplifiy_site_boundary()

        # 3,对每一条边，做周围道路的判断
        self.__check_road_properties()

        # 4, 有效周长
        specified_roads_num = 0  # 指明道路数量
        self.valid_perimeter = 0
        shown_seg_length = ""
        for seg in self.segs:
            if seg.abut_larger_street:
                if seg.geometry.length > SEG_MIN_DIS:
                    seg.abut_small_street_status = True
                    self.valid_perimeter += seg.geometry.length
                    specified_roads_num += 1
                    shown_seg_length += str(round(seg.geometry.length, 2)) + "m + "
        print(shown_seg_length)
        total_perimeter = min(10000, self.site_boundary.exterior.length)
        print(self.valid_perimeter)
        print(total_perimeter)
        perimeter_rate = self.valid_perimeter / total_perimeter
        print(f"-- 地块的有效周长, 为【{round(perimeter_rate * 100, 2)}%】")

        # 5, 计算地块类型
        site_classification = "A"
        if self.has_corner and specified_roads_num >= 3 and perimeter_rate > 0.6:
            site_classification = "C"
        elif self.has_corner and specified_roads_num >= 2 and perimeter_rate > 0.4:
            site_classification = "B"
        else:
            site_classification = "A"

        # site_classification_info = f"Site is automatically classified as {site_classification}: " \
        #                            f"Site contains {specified_roads_num} Specified-Street with a width greater than 4.5m, " \
        #                            f"and Site {'HAS' if self.has_corner else 'DON`T HAVE'} Corner. " \
        #                            f"Besides, {round(perimeter_rate * 100, 2)}% of site perimeter covers these streets. "

        site_classification_info = f"The site is automatically classified as [{site_classification}]:\n" \
                                   f"(1) Abutting specified street(s) with width > 4.5m = {specified_roads_num}\n" \
                                   f"(2) Has at least one corner = {'YES' if self.has_corner else 'NO'}\n" \
                                   f"(3) Specified street(s) coverage = {round(perimeter_rate * 100, 2)}%"

        print(f"-- 计算全部结束，地块类型为【{site_classification}】, 提示信息为：")
        print(site_classification_info)
        return site_classification, site_classification_info

    def __get_all_pavement_polygon(self):
        """
        获取项目地块范围内所有的pavement，polygon，并进行Union操作
        """
        # 以项目的BoundaryBox来Offset30m计算对应的道路区间
        minx, miny, maxx, maxy = self.site_boundary.bounds
        boundary_box = box(minx, miny, maxx, maxy).buffer(10, cap_style=2, join_style=2)
        print("-地块的道路检测框BoundaryBox为：")
        print(boundary_box)

        for road_polygon in self.road_polygon_list:
            if road_polygon.geometry.intersects(boundary_box):
                self.crossed_road_polygon_list.append(road_polygon)
        self.crossed_pavement_polygon_union = unary_union([road_polygon.geometry for road_polygon in self.crossed_road_polygon_list])
        print("-周边所有道路的合并图形PavementPolygonUnion为：")
        print(self.crossed_pavement_polygon_union)

    def __simplifiy_site_boundary(self):
        """
        简化场地边界
        """
        valid_points = []
        befor_pt = None
        for i, vertex in enumerate(self.site_boundary.exterior.coords[:-1]):
            if i == 0:
                valid_points.append(Point(vertex))
                befor_pt = Point(vertex)
                continue
            else:
                if Point(vertex).distance(befor_pt) >= SIMPLIFIED_POINT_DISTANCE:
                    valid_points.append(Point(vertex))
                    befor_pt = Point(vertex)
                else:
                    continue
        valid_points.append(valid_points[0])
        self.simplified_site_boundary = Polygon(valid_points)
        print("-简化后的地块图形SimplifiedSiteBoundary为：")
        print(self.simplified_site_boundary)
        print(f"将原地块图形的{len(self.site_boundary.exterior.coords)-1}条边 简化至 {len(self.simplified_site_boundary.exterior.coords)-1}条边")

    def __check_road_properties(self):
        # 生成所有的边缘及监测点
        total_start_time = time.time()

        for i, vertex in enumerate(self.simplified_site_boundary.exterior.coords[:-1]):
            edge = SiteEdge()
            edge.init(start_pt=Point(vertex), end_pt=Point(self.simplified_site_boundary.exterior.coords[i+1]), site_boundary=self.simplified_site_boundary)
            self.site_pts.extend(edge.sample_pts[:-1])

            for site_pt in edge.sample_pts:
                if self.crossed_pavement_polygon_union.contains(site_pt.check_pt):
                    site_pt.is_in_larger_street = True

            start_pt = None
            status = None
            seg_id = 0

            for j in range(len(edge.sample_pts)):
                this_pt = edge.sample_pts[j].geometry
                value = edge.sample_pts[j].is_in_larger_street

                if j == 0:
                    start_pt = copy.deepcopy(this_pt)
                    status = copy.deepcopy(value)
                elif j == len(edge.sample_pts) - 1:
                    if status is None:
                        self.segs[-1].end_pt = copy.deepcopy(this_pt)
                        self.segs[-1].geometry = LineString([self.segs[-1].start_pt, self.segs[-1].end_pt])
                    else:
                        end_pt = copy.deepcopy(this_pt)
                        self.segs.append(SiteEdgeSegment(name=f"{i+1}-{seg_id+1}",
                                                         start_pt=start_pt, end_pt=end_pt, geometry=LineString([start_pt, end_pt]),
                                                         abut_larger_street=status, normal_vector=edge.normal_vector))
                        seg_id += 1
                else:
                    if value == status:
                        continue
                    elif status is None:
                        start_pt = copy.deepcopy(this_pt)
                        status = copy.deepcopy(value)
                    else:
                        end_pt = copy.deepcopy(this_pt)
                        self.segs.append(SiteEdgeSegment(name=f"{i+1}-{seg_id+1}",
                                                         start_pt=start_pt, end_pt=end_pt, geometry=LineString([start_pt, end_pt]),
                                                         abut_larger_street=status, normal_vector=edge.normal_vector))
                        seg_id += 1
                        start_pt = None
                        status = None

            self.edges.append(edge)

        print("--经过边的计算，所有靠近>4.5m道路的边界图形为：")
        print(MultiLineString([seg.geometry for seg in self.segs if seg.abut_larger_street]))

        # 所有seg属性好了，开始计算对应属性
        for i, seg in enumerate(self.segs):
            # 顺便计算一下检测线
            seg.detected_line = LineString([Point(seg.start_pt.x + seg.normal_vector[0] * CHECK_DISTANCE,
                                                  seg.start_pt.y + seg.normal_vector[1] * CHECK_DISTANCE),
                                            Point(seg.end_pt.x + seg.normal_vector[0] * CHECK_DISTANCE,
                                                  seg.end_pt.y + seg.normal_vector[1] * CHECK_DISTANCE)])

            if i == 0:
                # 看看起始点有没有corner
                if self.segs[-1].abut_larger_street and self.segs[0].abut_larger_street:
                    angle = AutoUtils.calculate_angle(AutoUtils.pt_to_list(self.segs[-1].start_pt),
                                              AutoUtils.pt_to_list(self.segs[0].start_pt),
                                              AutoUtils.pt_to_list(self.segs[0].end_pt))
                    if angle < CORNER_ANGLE:
                        self.segs[i].start_pt_is_corner = True
                        self.has_corner = True
            else:
                if self.segs[i-1].abut_larger_street and self.segs[i].abut_larger_street:
                    angle = AutoUtils.calculate_angle(AutoUtils.pt_to_list(self.segs[i-1].start_pt),
                                                      AutoUtils.pt_to_list(self.segs[i].start_pt),
                                                      AutoUtils.pt_to_list(self.segs[i].end_pt))
                    if angle < CORNER_ANGLE:
                        self.segs[i].start_pt_is_corner = True
                        self.has_corner = True

        print("--所有靠近>4.5m道路的边界对应【检测线】为：")
        print(MultiLineString([seg.detected_line for seg in self.segs if seg.abut_larger_street]))
        # print(MultiLineString([seg.detected_line for seg in self.segs if not seg.abut_larger_street]))

        print(f"--经过端点计算，所有符合Coner(angle<140 & 相邻边靠近>4.5m马路)的端点为:")
        print(MultiPoint([seg.start_pt for seg in self.segs if seg.start_pt_is_corner]))
        print(f"总用时{time.time() - total_start_time}")

        # for seg in self.segs:
        #     print(f"Edge{seg.name}: length {round(seg.geometry.length, 2)}m "
        #           f"{'Abut streets >= 4.5m' if seg.abut_larger_street else ''} "
        #           f"{'Is Corner' if seg.start_pt_is_corner else ''}")


class AutoUtils:
    @staticmethod
    def pt_to_list(pt: Point):
        return [pt.x, pt.y]

    @staticmethod
    def calculate_angle(a, b, c):
        # print(a, b, c)
        ba = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        magnitude_ba = math.sqrt(ba[0] * ba[0] + ba[1] * ba[1])
        magnitude_bc = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1])
        angle_rad = math.acos(dot_product / (magnitude_ba * magnitude_bc))
        return math.degrees(angle_rad)

    @staticmethod
    def sample_points_on_linestring(input_line: LineString, num_points) -> List[Point]:
        # 计算Linestring的总长度
        total_length = input_line.length
        # 初始化采样点列表
        sampled_points = []
        # 从起点到终点均匀地采样num_points个点
        for i in range(num_points):
            distance_along_linestring = i * total_length / (num_points - 1)
            point = input_line.interpolate(distance_along_linestring)
            sampled_points.append(Point(point.coords[0]))
        return sampled_points

    @staticmethod
    def get_normal_vector(start_pt: Point, end_pt: Point, site_boundary: Polygon):
        # 找到法向量
        # 1. 计算线段的方向向量
        start_pt = [start_pt.x, start_pt.y]
        end_pt = [end_pt.x, end_pt.y]
        direction_vector = (end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])

        # 2. 将方向向量进行单位化，得到线段的单位方向向量
        magnitude = (direction_vector[0] ** 2 + direction_vector[1] ** 2) ** 0.5
        unit_direction_vector = (direction_vector[0] / magnitude, direction_vector[1] / magnitude)

        # 3.对每一个点，计算方向边，多数即为法向量方向
        direction_vote = 0
        reverse_direction_vote = 0
        for sample_pt in AutoUtils.sample_points_on_linestring(LineString([start_pt, end_pt]), 10):
            pt = [sample_pt.x, sample_pt.y]
            point1 = (pt[0] + unit_direction_vector[1], pt[1] - unit_direction_vector[0])
            if site_boundary.contains(Point(point1)):
                direction_vote += 1
            else:
                reverse_direction_vote += 1

        # 4.获得法向量
        if direction_vote >= reverse_direction_vote:
            return [-unit_direction_vector[1], unit_direction_vector[0]]
        else:
            return [unit_direction_vector[1], -unit_direction_vector[0]]


if __name__ == "__main__":
    cl = AutoClassificationV2()
    test_geo_wkt = "POLYGON ((838557.5 816968.4, 838617 816939, 838614 816931, 838610 816925, 838605 816923, 838598 816923, 838590 816924, 838584 816927, 838580 816928, 838575 816931, 838574 816931.6, 838571 816933, 838566.6 816935.4, 838564 816937, 838559.7 816940.6, 838559.2 816941.5, 838556.4 816945.4, 838555.9 816946.3, 838554.8 816948.2, 838554 816950, 838553.5 816951.3, 838552.2 816956.5, 838552 816965, 838557.5 816968.4))"
    new = "POLYGON ((838584 816954, 838546 816943, 838552 816936, 838570 816937, 838573.1 816932.5, 838580 816936, 838579.8 816940.5, 838593.5 816942.4, 838594.4 816948.4, 838595.6 816954.3, 838592 816959, 838584 816954))"
    cl.init(site_boundary=wkt.loads(new),
            project_boundary=Polygon(), road_center_line_list=[], road_polygon_list=[])

