"""
自动识别土地分类和地块有关的计算细节，计算输入地块的土地分类
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


BUFFER_SIMPLIFY_DISTANCE = 5
SIMPLIFY_TOLERANCE = 1
MIN_LENGTH = 18
# 检测近似路段的采样数
SAMPLE_NUM = 10
# 检测共线路段的剪辑
SAME_DIRECTION_DISTANCE = 2
SAME_DIRECTION_ANGLE = 140


@dataclass
class AutoClassification:
    site_boundary: Polygon = field(default_factory=Polygon)
    project_boundary: Polygon = field(default_factory=Polygon)
    simplified_boundary: Polygon = field(default_factory=Polygon)

    # 读入原始地图数据
    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)

    crossed_road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    crossed_road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)

    site_pts: List[SitePoint] = field(default_factory=list)
    site_segs: List[SiteSegment] = field(default_factory=list)
    specified_roads: List[SpecifiedRoad] = field(default_factory=list)

    # 有效道路检测框的Union
    simplified_checkbox_union: BaseGeometry = field(default_factory=BaseGeometry)
    # 相交道路图形的Union
    crossed_pavement_polygon_union: BaseGeometry = field(default_factory=BaseGeometry)
    # 有效周长占比
    perimeter_rate: float = field(default=float)

    @property
    def specified_roads_en_name(self):
        return [road.street_name_en for road in self.specified_roads]

    @property
    def specified_roads_cropped_geometry(self):
        geometry_list = []
        for specified_road in self.specified_roads:
            geometry_list.extend(specified_road.cropped_geometry)
        return GeometryCollection(geometry_list)

    @property
    def specified_larger_roads_cropped_geometry(self):
        geometry_list = []
        for specified_road in self.specified_roads:
            if specified_road.larger_than_demand_state:
                geometry_list.extend(specified_road.cropped_geometry)
        return GeometryCollection(geometry_list)

    def init(self, site_boundary: Polygon, project_boundary: Polygon,
             road_center_line_list: List[MapRoadCenterLine], road_polygon_list: List[MapRoadPolygon]):
        self.site_boundary = site_boundary
        self.project_boundary = project_boundary
        self.road_center_line_list = copy.deepcopy(road_center_line_list)
        self.road_polygon_list = copy.deepcopy(road_polygon_list)

        print("---【土地分类自动计算模块·V1】---")
        print(f"-- 启动模块，原始的输入地块图形为:")
        print(self.site_boundary)
        # 对输入任意的地块图形进行处理
        self.__init_valid_site()

        # STEP1: 计算有无邻近道路的边界
        print("-- #1 计算有无邻近道路的边界并提取相应信息：")
        self.__check_site_segment_abut_street()

        # STEP2: 有效的指明检测检测
        print("-- #2 开始检测指明街道的数量：")
        self.__check_specified_road_algorithm()

        # STEP3: 计算指明道路的宽度
        print("-- #3 开始计算指明街道的宽度：")
        self.__compute_specified_road_width_algorithm()

        # STEP4: 计算指明道路对应的地块有效周长
        target_geo = self.specified_larger_roads_cropped_geometry
        total_length = 0
        for seg in self.site_segs:
            if seg.abut_street:
                # 该条边界首先靠近道路，且与大于4.5m是地块道路相交
                if seg.centerline_checkbox.intersects(target_geo):
                    outcome = seg.centerline_checkbox.intersection(target_geo)
                    if outcome.length >= 4:
                        total_length += seg.geometry.length
                        # print(seg.geometry.length)
                        # print(total_length)
        # print(self.site_boundary.exterior.length)
        self.perimeter_rate = total_length / min(9999, self.site_boundary.exterior.length)
        print(f"-- #4 计算地块的有效周长, 为【{round(self.perimeter_rate * 100, 2)}%】")

        # STEP5: 计算地块类型
        specified_roads_num = len(self.specified_roads)
        larger_than_demand_roads_num = 0
        each_width_str = ""
        for specified_road in self.specified_roads:
            larger_than_demand_roads_num += specified_road.larger_than_demand_state
            each_width_str += str(round(specified_road.average_road_width, 2)) + "m, "

        site_classification = "A"
        if specified_roads_num >= 3 and larger_than_demand_roads_num >=3:
            if self.perimeter_rate >= 0.6:
                site_classification = "C"
            else:
                site_classification = "B"
        elif specified_roads_num >= 2 and larger_than_demand_roads_num >=2:
            if self.perimeter_rate >= 0.4:
                site_classification = "B"
            else:
                site_classification = "A"
        else:
            site_classification = "A"
        # site_classification_info = f"Site is 【Class {site_classification}】.  " \
        #                            f"Site contains {specified_roads_num} Specified-Street, " \
        #                            f"and {larger_than_demand_roads_num} of them with a width greater than 4.5m. " \
        #                            f"Besides, {round(self.perimeter_rate * 100, 2)}% of site perimeter covers these streets. "

        site_classification_info = f"The site is automatically classified as [{site_classification}]:\n" \
                                   f"(1) Abutting specified street(s) with width > 4.5m = {larger_than_demand_roads_num}\n" \
                                   f"(2) Specified street(s) coverage = {round(self.perimeter_rate * 100, 2)}%"

        print(f"-- 计算全部结束，地块类型为【{site_classification}】, 提示信息为：")
        print(site_classification_info)

        return site_classification, site_classification_info

    def __init_valid_site(self):
        # 消除异常区域
        self.simplified_boundary = self.site_boundary.buffer(BUFFER_SIMPLIFY_DISTANCE, cap_style=2, join_style=2).buffer(-BUFFER_SIMPLIFY_DISTANCE, cap_style=2, join_style=2)
        # 消除共线点
        # self.simplified_boundary = self.simplified_boundary.simplify(tolerance=SIMPLIFY_TOLERANCE, preserve_topology=True)
        print("-简化后的地块图形为:")
        print(self.simplified_boundary)

        for i, vertex in enumerate(self.simplified_boundary.exterior.coords[:-1]):
            site_pt = SitePoint(id=i, geometry=Point(vertex))
            self.site_pts.append(site_pt)

        # 找到有效顶点
        # 有效顶点判断标准：计算每个交点相邻边的夹角，若<150°，则该点视为有效顶点
        for i, site_pt in enumerate(self.site_pts):
            if i == 0:
                prev_pt = self.site_pts[len(self.site_pts)-1]
                next_pt = self.site_pts[1]
            elif i == len(self.site_pts)-1:
                prev_pt = self.site_pts[i-1]
                next_pt = self.site_pts[0]
            else:
                prev_pt = self.site_pts[i-1]
                next_pt = self.site_pts[i+1]

            angle = AutoUtils.calculate_angle(AutoUtils.pt_to_list(prev_pt.geometry),
                                              AutoUtils.pt_to_list(site_pt.geometry),
                                              AutoUtils.pt_to_list(next_pt.geometry))
            if angle < 140:
                site_pt.is_valid = True

                if self.site_boundary.contains(LineString([prev_pt.geometry, next_pt.geometry]).centroid):
                    # 该点为凸有效点
                    pass
                else:
                    site_pt.is_valid = False

        print("-检测出的有效顶点:")
        print(MultiPoint([pt.geometry for pt in self.site_pts if pt.is_valid]))

        # 重新排序
        temp_pts = copy.deepcopy(self.site_pts)
        self.site_pts.clear()
        # 找到第一个有效点作为开始
        start_id = 0
        for i, site_pt in enumerate(temp_pts):
            if site_pt.is_valid:
                site_pt.is_start = True
                start_id = i
                break
        # 找到最后一个有效点作为结束
        end_id = 0
        for i, site_pt in enumerate(temp_pts):
            if site_pt.is_valid:
                end_id = i
        temp_pts[end_id].is_end = True
        # 重新组合多边形顶点
        self.site_pts = temp_pts[start_id:] + temp_pts[:start_id]
        for i in range(len(self.site_pts)):
            self.site_pts[i].id = i
            # print(self.site_pts[i])

        # 生成有效Segment
        for start_i, start_site_pt in enumerate(self.site_pts):
            if start_site_pt.is_valid:
                seg = SiteSegment(start_site_pt=start_site_pt)
                seg.contain_site_pts.append(start_site_pt)

                # 正常区间
                if not start_site_pt.is_end:
                    for end_i, end_site_pt in enumerate(self.site_pts[start_i+1:]):
                        seg.contain_site_pts.append(end_site_pt)
                        if end_site_pt.is_valid:
                            seg.end_site_pt = end_site_pt
                            break
                    seg.init(self.simplified_boundary)
                    self.site_segs.append(seg)
                # 首位区间
                else:
                    for end_i, end_site_pt in enumerate(self.site_pts[start_i+1:]):
                        seg.contain_site_pts.append(end_site_pt)
                    seg.contain_site_pts.append(self.site_pts[0])
                    seg.end_site_pt = self.site_pts[0]
                    seg.init(self.simplified_boundary)
                    self.site_segs.append(seg)

        print("-检测出的有效边:")
        print(MultiLineString([seg.geometry for seg in self.site_segs]))

    def __check_site_segment_abut_street(self):
        """
        计算有无邻近道路的边界，有则更新在self.site_segs中
        更新周边相交的道路图形
        """
        # 检测每条seg是否与道路详相交
        for seg in self.site_segs:
            abut_street = False
            for road_polygon in self.road_polygon_list:
                if seg.road_checkbox.intersects(road_polygon.geometry):
                    # 如果相交，要计算一下交叠面积
                    outcome = seg.road_checkbox.intersection(road_polygon.geometry)
                    if outcome.area > seg.road_checkbox.area / 3:
                        abut_street = True
                        break

            # 如果相交，则计算
            if abut_street:
                seg.abut_street = True
                seg.gen_centerline_checkbox()

        # 生成所有的有效道路检测框
        self.simplified_checkbox_union = GeometryCollection(self.get_union_checkbox("road_center_line"))
        print("-道路判断的有效检测框图形:")
        print(self.simplified_checkbox_union)

        # 计算出所有相交的pavement polygon
        for road_polygon in self.road_polygon_list:
            if road_polygon.geometry.intersects(self.simplified_checkbox_union):
                self.crossed_road_polygon_list.append(road_polygon)
        print(f"-共找到地块附近【{len(self.crossed_road_polygon_list)}个】有效的道路图形")

        print("-所有与检测框相交的道路图形:")
        self.crossed_pavement_polygon_union = unary_union([road_polygon.geometry for road_polygon in self.crossed_road_polygon_list])
        print(self.crossed_pavement_polygon_union)

    def __check_specified_road_algorithm(self):
        # 1 第一步初步计算，筛选出相交的道路中心线
        crossed_road_center_line_list_temp = []  # 初步计算数值，不一定准
        for road_center_line in self.road_center_line_list:
            if road_center_line.geometry.intersects(self.simplified_checkbox_union):
                # 计算相交长度
                clipped_geometry = MapCropper.outcome_process(road_center_line.geometry.intersection(self.simplified_checkbox_union))
                total_crossed_length = 0
                for geo in clipped_geometry:
                    total_crossed_length += geo.length
                if total_crossed_length < MIN_LENGTH:
                    print(f"{road_center_line.object_id}号路名：{road_center_line.street_name_en}，检测出相交距离：{round(total_crossed_length, 2)}m 不符合要求，予以略过")
                    continue
                else:
                    print(f"-【{road_center_line.object_id}号路名：{road_center_line.street_name_en}】，检测出相交距离：{round(total_crossed_length, 2)}m, "
                          f"占总路长{round(total_crossed_length / road_center_line.geometry.length * 100, 2)}%")
                crossed_road_center_line_list_temp.append(road_center_line)

        print(f"第一次检测，初步计算共有{len(crossed_road_center_line_list_temp)}条道路相交")

        # 2 第二步，反向检验，为每个道路先找到重合区间，在重合区间中生成采样投射线，若相交率过大，则去除;
        # 3 去掉重名道路
        specified_roads_temp = []
        site_linestring = self.site_boundary.exterior
        for i in range(len(crossed_road_center_line_list_temp)):
            # S1 生成重合区间
            road_center_line = crossed_road_center_line_list_temp[i]
            clipped_geometry = MapCropper.outcome_process(road_center_line.geometry.intersection(self.simplified_checkbox_union))

            # 其他所有线的图形
            other_lines = MultiLineString([crossed_road_center_line_list_temp[j].geometry for j in range(len(crossed_road_center_line_list_temp)) if i != j])
            intersects_state = False
            total_intersects_num = 0
            detected_lines = []
            for geo in clipped_geometry:
                # S2 采样，10个点
                pts = AutoUtils.sample_points_on_linestring(geo, SAMPLE_NUM)
                # print(MultiPoint(pts))

                # S3 每个点，找到最近的点，画连线
                intersects_sample_num = 0

                for sample_pt in pts:
                    # 最近点
                    nearest_point = site_linestring.interpolate(site_linestring.project(sample_pt))
                    # 交线
                    nearset_line = LineString([nearest_point, sample_pt])

                    detected_line = DetectedLine(geometry=nearset_line,
                                                 site_pt=Point(nearest_point),
                                                 road_pt=Point(sample_pt))
                    detected_lines.append(detected_line)
                    # print(nearset_line)

                    if nearset_line.intersects(other_lines):
                        intersects_sample_num += 1
                        total_intersects_num += 1

                if intersects_sample_num > SAMPLE_NUM / 3:
                    intersects_state = True

            if intersects_state:
                print(f"{road_center_line.object_id}号路名：{road_center_line.street_name_en}，"
                      f"道路重合采样率为{round(total_intersects_num*100/(SAMPLE_NUM*len(clipped_geometry)), 2)}%，予以略过")
            else:
                # 暂时不考虑重名过滤
                new_specified_road = SpecifiedRoad(street_name_en=road_center_line.street_name_en,
                                                   street_name_tc=road_center_line.street_name_tc,)
                new_specified_road.road_center_line_list.append(road_center_line)
                new_specified_road.cropped_geometry = clipped_geometry
                new_specified_road.detected_lines = detected_lines
                specified_roads_temp.append(new_specified_road)
                # print(MultiLineString(clipped_geometry))

                # 考虑重名过滤
                # self.crossed_road_center_line_list.append(road_center_line)
                # if road_center_line.street_name_en == "" and road_center_line.street_name_tc == "":
                #     new_specified_road = SpecifiedRoad()
                #     new_specified_road.road_center_line_list.append(road_center_line)
                #     new_specified_road.cropped_geometry = clipped_geometry
                #     new_specified_road.detected_lines = detected_lines
                #     self.specified_roads.append(new_specified_road)
                # elif road_center_line.street_name_en in self.specified_roads_en_name:
                #     for specified_road in self.specified_roads:
                #         if specified_road.street_name_en == road_center_line.street_name_en:
                #             specified_road.road_center_line_list.append(road_center_line)
                #             specified_road.cropped_geometry.extend(clipped_geometry)
                #             specified_road.detected_lines.extend(detected_lines)
                #             break
                # else:
                #     new_specified_road = SpecifiedRoad(street_name_en=road_center_line.street_name_en,
                #                                        street_name_tc=road_center_line.street_name_tc,)
                #     new_specified_road.road_center_line_list.append(road_center_line)
                #     new_specified_road.cropped_geometry = clipped_geometry
                #     new_specified_road.detected_lines = detected_lines
                #     self.specified_roads.append(new_specified_road)
        print(f"第二次检测，经过筛选余留{len(specified_roads_temp)}条道路中心线")

        # 根据中心线角度，按140度的夹角来计算数量
        filter_id = []
        for i in range(len(specified_roads_temp)):
            if i in filter_id:
                continue

            specified_road = specified_roads_temp[i]
            this_filter_id = []

            for from_cropped_geometry in specified_road.cropped_geometry:
                for j in range(i+1, len(specified_roads_temp)):
                    if j in filter_id:
                        continue

                    for target_cropped_geometry in specified_roads_temp[j].cropped_geometry:
                        if from_cropped_geometry.distance(target_cropped_geometry) < SAME_DIRECTION_DISTANCE:
                            # 处于共线状态，计算夹角
                            from_start_pt = from_cropped_geometry.coords[0]
                            from_end_pt = from_cropped_geometry.coords[-1]
                            target_start_pt = target_cropped_geometry.coords[0]
                            target_end_pt = target_cropped_geometry.coords[-1]

                            # 提取两条直线的方向向量
                            direction_vector1 = (from_end_pt[0] - from_start_pt[0], from_end_pt[1] - from_start_pt[1])
                            direction_vector2 = (target_end_pt[0] - target_start_pt[0], target_end_pt[1] - target_start_pt[1])

                            # 计算向量的内积
                            dot_product = direction_vector1[0] * direction_vector2[0] + direction_vector1[1] * direction_vector2[1]

                            # 计算夹角（弧度制）
                            angle_radians = math.acos(dot_product / (math.sqrt(direction_vector1[0] ** 2 + direction_vector1[1] ** 2) *
                                                                     math.sqrt(direction_vector2[0] ** 2 + direction_vector2[1] ** 2)))

                            # 将弧度制转换为角度制
                            angle_degrees = math.degrees(angle_radians)
                            if angle_degrees < 180 - SAME_DIRECTION_ANGLE:
                                print(f"第{i}条道路中心线与第{j}道路中心线夹角为{abs(180-angle_degrees)},视为同向线")
                                filter_id.append(j)
                                this_filter_id.append(j)

            if len(this_filter_id) > 0:
                for delete_id in this_filter_id:
                    delete_specified_road = specified_roads_temp[delete_id]
                    specified_road.road_center_line_list.extend(delete_specified_road.road_center_line_list)
                    specified_road.cropped_geometry.extend(delete_specified_road.cropped_geometry)
                    specified_road.detected_lines.extend(delete_specified_road.detected_lines)

            self.specified_roads.append(specified_road)
        print("第三次140度夹角检测完成")
        print(f"-【地块紧邻{len(self.specified_roads)}条指明街道】，分别为：{self.specified_roads_en_name}")

        print("-所有指明街道的图形信息为：")
        print(self.specified_roads_cropped_geometry)

    def get_union_checkbox(self, checkbox_type: str = "road_center_line"):
        checkbox_list = []
        for seg in self.site_segs:
            if seg.abut_street:
                if checkbox_type == "road_center_line":
                    checkbox_list.append(seg.centerline_checkbox)
                else:
                    pass
        return MapCropper.outcome_process(unary_union(checkbox_list))

    def __compute_specified_road_width_algorithm(self):
        """
        计算每一条指明道路的宽度
        """
        # 对每一个detectedLine生成100个点，每个点对应0.1m精度
        inside_road_detected_pts = []
        for specified_road in self.specified_roads:
            # 对于每一个检测线
            total_road_width = 0
            for detected_line in specified_road.detected_lines:
                detected_line.init_all_points()
                # 对于每一个监测点
                for detected_pt in detected_line.detected_pts:
                    if self.crossed_pavement_polygon_union.contains(detected_pt.geometry):
                        detected_pt.in_road = True
                        inside_road_detected_pts.append(detected_pt.geometry)

                detected_line.compute_distance()
                total_road_width += detected_line.road_width
                # print(detected_line.road_width)

            # 计算道路平均长度
            specified_road.average_road_width = total_road_width / min(9999, len(specified_road.detected_lines))
            if specified_road.average_road_width >= 4.5:
                specified_road.larger_than_demand_state = True
            print(f"-指明道路：{specified_road.street_name_en}路宽为{round(specified_road.average_road_width, 2)}m, 【{'大于' if specified_road.larger_than_demand_state else '小于'} 4.5m】")

        print("-所有有效的道路宽度检测点为：")
        print(MultiPoint(inside_road_detected_pts))


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


if __name__ == "__main__":
    cl = AutoClassification()
    test_geo_wkt = "POLYGON ((838557.5 816968.4, 838617 816939, 838614 816931, 838610 816925, 838605 816923, 838598 816923, 838590 816924, 838584 816927, 838580 816928, 838575 816931, 838574 816931.6, 838571 816933, 838566.6 816935.4, 838564 816937, 838559.7 816940.6, 838559.2 816941.5, 838556.4 816945.4, 838555.9 816946.3, 838554.8 816948.2, 838554 816950, 838553.5 816951.3, 838552.2 816956.5, 838552 816965, 838557.5 816968.4))"
    new = "POLYGON ((838584 816954, 838546 816943, 838552 816936, 838570 816937, 838573.1 816932.5, 838580 816936, 838579.8 816940.5, 838593.5 816942.4, 838594.4 816948.4, 838595.6 816954.3, 838592 816959, 838584 816954))"
    cl.init(site_boundary=wkt.loads(new),
            project_boundary=Polygon(), road_center_line_list=[], road_polygon_list=[])

