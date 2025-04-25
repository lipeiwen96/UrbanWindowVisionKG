"""
自动生成建筑退线的算法
"""
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
import numpy as np


SETBACK_BUFFER_DISTANCE = 25
SAMPLE_DISTANCE = 15
HALF_LINE_LENGTH = 20


@dataclass
class SetbackRoad:
    street_name_en: str = field(default=str)
    street_name_tc: str = field(default=str)
    geometry: LineString = field(default_factory=LineString)
    width: float = field(default=float)
    need_setback_status: bool = field(default=True)
    setback_area: Polygon = field(default_factory=Polygon)  # TODO: 这里有个风险，目前默认使用sbdg规范来排


@dataclass
class AutoSetback:
    site_boundary: Polygon = field(default_factory=Polygon)
    # 读入原始地图数据
    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)

    crossed_road_center_line: List[MapRoadCenterLine] = field(default_factory=list)
    crossed_road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    # 相交道路图形的Union
    crossed_pavement_polygon_union: BaseGeometry = field(default_factory=BaseGeometry)

    setback_road_list: List[SetbackRoad] = field(default_factory=list)

    @property
    def road_info(self):
        outcome = []
        for road in self.setback_road_list:
            width_label = 0
            if road.width > 15:
                width_label = 3
            elif road.width >= 4.5:
                width_label = 2
            else:
                width_label = 1
            outcome.append({
                "street_name_en": road.street_name_en,
                "geometry": road.geometry.wkt,
                "width": road.width,
                "width_label": width_label,
                "setback_area": road.setback_area.wkt,
            })
        return outcome

    def init(self, site_boundary: Polygon, road_center_line_list: List[MapRoadCenterLine], road_polygon_list: List[MapRoadPolygon]):
        self.site_boundary = site_boundary
        print("-场地轮廓为：")
        print(site_boundary)
        self.road_center_line_list = copy.deepcopy(road_center_line_list)
        self.road_polygon_list = copy.deepcopy(road_polygon_list)

        # 先找到周围所有的道路中心线
        self.__find_around_road_centerline()

        # 计算道路中心线对应的道路宽度
        self.__compute_road_width()

        # 生成退线
        self.__generate_setback()

        # 裁剪建筑区域
        pass

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

    def __find_around_road_centerline(self):
        # 以项目的BoundaryBox来Offset30m计算对应的道路区间
        minx, miny, maxx, maxy = self.site_boundary.bounds
        boundary_box = box(minx, miny, maxx, maxy).buffer(SETBACK_BUFFER_DISTANCE, cap_style=2, join_style=2)
        print("-地块的道路检测框BoundaryBox为：")
        print(boundary_box)

        temp_crossed_road_center_line = []
        for road_center_line in self.road_center_line_list:
            if road_center_line.geometry.intersects(boundary_box):
                temp_crossed_road_center_line.append(road_center_line)
        # 反向检验，为每个道路先找到重合区间，在重合区间中生成采样投射线，若相交率过大，则去除;
        SAMPLE_NUM = 10
        specified_roads_temp = []
        site_linestring = self.site_boundary.exterior
        for i in range(len(temp_crossed_road_center_line)):
            # S1 生成重合区间
            road_center_line = temp_crossed_road_center_line[i]
            clipped_geometry = MapCropper.outcome_process(road_center_line.geometry.intersection(boundary_box))
            # print(GeometryCollection(clipped_geometry))

            # 其他所有线的图形
            other_lines = MultiLineString([temp_crossed_road_center_line[j].geometry for j in range(len(temp_crossed_road_center_line)) if i != j])
            intersects_state = False
            total_intersects_num = 0
            detected_lines = []
            for geo in clipped_geometry:
                # S2 采样，10个点
                pts = AutoSetback.sample_points_on_linestring(geo, SAMPLE_NUM)
                # print(MultiPoint(pts))

                # S3 每个点，找到最近的点，画连线
                intersects_sample_num = 0

                for sample_pt in pts:
                    # 最近点
                    nearest_point = site_linestring.interpolate(site_linestring.project(sample_pt))
                    # 交线
                    nearset_line = LineString([nearest_point, sample_pt])
                    # print(nearset_line)

                    if nearset_line.intersects(other_lines):
                        intersects_sample_num += 1
                        total_intersects_num += 1

                if intersects_sample_num > SAMPLE_NUM / 3:
                    intersects_state = True
            if intersects_state:
                print(f"{road_center_line.object_id}号路名：{road_center_line.street_name_en}，"
                      f"道路重合采样率为{round(total_intersects_num * 100 / (SAMPLE_NUM * len(clipped_geometry)), 2)}%，予以略过")
            else:
                self.crossed_road_center_line.append(road_center_line)
        print(f"经过筛选余留{len(self.crossed_road_center_line)}条道路中心线")
        self.crossed_pavement_polygon_union = unary_union([road_center_line.geometry for road_center_line in self.crossed_road_center_line])
        print("-周边所有道路的合并图形road_center_line为：")
        print(self.crossed_pavement_polygon_union)

        for road_polygon in self.road_polygon_list:
            if road_polygon.geometry.intersects(boundary_box):
                self.crossed_road_polygon_list.append(road_polygon)
        self.crossed_pavement_polygon_union = unary_union([road_polygon.geometry for road_polygon in self.crossed_road_polygon_list])
        print("-周边所有道路的合并图形PavementPolygonUnion为：")
        print(self.crossed_pavement_polygon_union)

    def __compute_road_width(self):
        for road_centerline in self.crossed_road_center_line:
            road = SetbackRoad(geometry=LineString(road_centerline.geometry),
                               street_name_en=road_centerline.street_name_en,
                               street_name_tc=road_centerline.street_name_tc,
                               )

            sample_width = []

            # 取采样点
            for i in range(len(road.geometry.coords)):
                if i == len(road.geometry.coords) - 1:
                    pass
                else:
                    start_pt = Point(road.geometry.coords[i])
                    end_pt = Point(road.geometry.coords[i+1])

                    # 计算线段的向量
                    dx = end_pt.x - start_pt.x
                    dy = end_pt.y - start_pt.y
                    # 计算线段的长度
                    line_length = np.sqrt(dx ** 2 + dy ** 2)
                    # 计算线段的单位向量
                    unit_vector = np.array([dx, dy]) / line_length

                    # 计算需要多少个采样点
                    num_samples = math.floor(line_length / SAMPLE_DISTANCE)
                    if num_samples * SAMPLE_DISTANCE >= line_length - SAMPLE_DISTANCE / 3:
                        num_samples -= 1  # 为了避免终止点的重复计算

                    # 开始生成采样点
                    for j in range(num_samples + 1):

                        # 采样点
                        x = start_pt.x + j * SAMPLE_DISTANCE * unit_vector[0]
                        y = start_pt.y + j * SAMPLE_DISTANCE * unit_vector[1]
                        sampled_point = Point(x, y)

                        # 生成垂直线
                        point1 = Point(sampled_point.x + unit_vector[1] * HALF_LINE_LENGTH, sampled_point.y - unit_vector[0] * HALF_LINE_LENGTH)
                        point2 = Point(sampled_point.x - unit_vector[1] * HALF_LINE_LENGTH, sampled_point.y + unit_vector[0] * HALF_LINE_LENGTH)
                        detected_line = LineString([point1, point2])
                        # print(detected_line)

                        # 计算道路长度
                        width = AutoSetback.get_length(detected_line.intersection(self.crossed_pavement_polygon_union), sampled_point)
                        sample_width.append(width)

            # print(sample_width)
            # 计算平均值
            road.width = round(sum(sample_width) / min(len(sample_width), 9999), 2)

            # TODO: 如果0的梳理过半，则去掉该条道路
            zero_width_num = 0
            for wdith in sample_width:
                if wdith == 0:
                    zero_width_num += 1
            if zero_width_num >= (len(sample_width) / 2):
                print(f"道路 {road.street_name_en}被过滤")
                continue

            print(f"【道路 {road.street_name_en}的宽度为{road.width},需要计算Setback】")
            self.setback_road_list.append(road)

    @staticmethod
    def get_length(outcome, original_point: Point):
        if outcome.is_empty:
            return 0
        elif isinstance(outcome, LineString):
            # 单图形结果，不做处理，直接输出
            return outcome.length
        elif isinstance(outcome, MultiLineString):
            for single_geometry in outcome:
                if original_point.distance(single_geometry) <= 0.1:
                    return single_geometry.length
        return 0

    def __generate_setback(self):
        """
        计算每一个SetbackRoad的道路宽度
        """
        print("道路的退距图形为")
        for road in self.setback_road_list:
            if road.width > 15:
                road.need_setback_status = False
            elif road.width >= 4.5:
                road.setback_area = road.geometry.buffer(7.5, cap_style=2, join_style=3)
            else:
                road.setback_area = road.geometry.buffer(7.5, cap_style=2, join_style=3)
            print(road.setback_area)
