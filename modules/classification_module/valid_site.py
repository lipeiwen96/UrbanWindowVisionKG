import math
from dataclasses import field, dataclass
from typing import List
from shapely.geometry import Polygon, Point, LineString, GeometryCollection, box, MultiPolygon
from shapely.geometry.base import BaseGeometry
from map_system.tile_genenrator.map_structure import MapRoadCenterLine, MapBuilding, MapLot, MapRoadPolygon


# 检测是否靠近道路的检测框，距离较小
ROAD_CHECK_DISTANCE = 10
# 检测道路中线使用的检测框宽度，适当大一些，用于涵盖尽可能所有的道路中心线
ROAD_CENTERLINE_CHECK_DISTANCE = 30

NUM_SAMPLES = 60
SAMPLE_SPACING = 0.2


@dataclass
class SitePoint:
    id: int = field(default=0)
    geometry: Point = field(default_factory=Point)
    is_valid: bool = field(default=False)
    is_start: bool = field(default=False)
    is_end: bool = field(default=False)

    # 新增
    check_pt: Point = field(default_factory=Point)
    is_in_larger_street: bool = field(default=False)  # >4.5m street
    is_edge_point: bool = field(default=False)
    is_corner: bool = field(default=False)



@dataclass
class SiteSegment:
    geometry: LineString = field(default_factory=LineString)
    start_site_pt: SitePoint = field(default_factory=SitePoint)
    end_site_pt: SitePoint = field(default_factory=SitePoint)
    contain_site_pts: List[SitePoint] = field(default_factory=list)
    normal_vector: list = field(default=list)  # 法向量

    # 检测是否靠近道路的检测框
    road_checkbox: Polygon = field(default_factory=Polygon)
    # 检测道路中线使用的检测框
    centerline_checkbox: Polygon = field(default_factory=Polygon)

    # 是否靠近道路
    abut_street: bool = field(default=False)

    # TODO: 检测图形
    # TODO: 检测属性
    # TODO: 是否临边

    def init(self, site_boundary: Polygon):
        # 生成图形
        self.geometry = LineString([site_pt.geometry for site_pt in self.contain_site_pts])

        # 找到法向量
        # 1. 计算线段的方向向量
        start_pt = [self.start_site_pt.geometry.x, self.start_site_pt.geometry.y]
        end_pt = [self.end_site_pt.geometry.x, self.end_site_pt.geometry.y]
        direction_vector = (end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
        # print(direction_vector)

        # 2. 将方向向量进行单位化，得到线段的单位方向向量
        magnitude = (direction_vector[0] ** 2 + direction_vector[1] ** 2) ** 0.5
        unit_direction_vector = (direction_vector[0] / magnitude, direction_vector[1] / magnitude)

        # 3.对每一个点，计算方向边，多数即为法向量方向
        direction_vote = 0
        reverse_direction_vote = 0
        for site_pt in self.contain_site_pts:
            pt = [site_pt.geometry.x, site_pt.geometry.y]
            point1 = (pt[0] + unit_direction_vector[1], pt[1] - unit_direction_vector[0])
            # print(Point(point1))
            if site_boundary.contains(Point(point1)):
                direction_vote += 1
            else:
                reverse_direction_vote += 1

        # 4.获得法向量
        if direction_vote >= reverse_direction_vote:
            self.normal_vector = [-unit_direction_vector[1], unit_direction_vector[0]]
        else:
            self.normal_vector = [unit_direction_vector[1], -unit_direction_vector[0]]

        _, self.road_checkbox = SiteSegment.gen_checking_box_and_seg(self.geometry, self.normal_vector, ROAD_CHECK_DISTANCE)
        # SiteSegment.gen_checking_box_and_seg(self.geometry, self.normal_vector, 5)

    def gen_centerline_checkbox(self):
        # 根据法向量生成计算框
        _, self.centerline_checkbox = SiteSegment.gen_checking_box_and_seg(self.geometry, self.normal_vector, ROAD_CENTERLINE_CHECK_DISTANCE)
        # SiteSegment.gen_checking_box_and_seg(self.geometry, self.normal_vector, 5)

    @staticmethod
    def gen_checking_box_and_seg(seg: LineString, normal_vector: list, checking_distance: float = 10):
        # 反序检测线
        checking_seg = LineString([[pt[0] + normal_vector[0] * checking_distance,
                                  pt[1] + normal_vector[1] * checking_distance] for pt in seg.coords[::-1]])
        # 检测框
        checking_box_coords = list(seg.coords) + list(checking_seg.coords)
        checking_box_coords.append(checking_box_coords[0])
        checking_box = Polygon(checking_box_coords)
        if not checking_box.is_valid:
            checking_box = checking_box.buffer(10, cap_style=2, join_style=2).buffer(-10, cap_style=2, join_style=2)

        # print(checking_box)
        return checking_seg, checking_box


@dataclass
class DetectedPt:
    geometry: Point = field(default_factory=Point)
    in_road: bool = field(default=False)
    distance: float = field(default=field())


@dataclass
class DetectedLine:
    geometry: LineString = field(default_factory=LineString)
    site_pt: Point = field(default_factory=Point)
    road_pt: Point = field(default_factory=Point)

    detected_pts: List[DetectedPt] = field(default_factory=list)
    road_width: float = field(default=float)

    def init_all_points(self):
        direction_vector = (self.road_pt.x - self.site_pt.x,
                            self.road_pt.y - self.site_pt.y)
        direction_magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
        direction_unit_vector = (direction_vector[0] / direction_magnitude, direction_vector[1] / direction_magnitude)

        # 生成采样检测点，依次从0.1m - 10.0m
        for i in range(1, NUM_SAMPLES + 1):
            x_sample = self.site_pt.x + i * SAMPLE_SPACING * direction_unit_vector[0]
            y_sample = self.site_pt.y + i * SAMPLE_SPACING * direction_unit_vector[1]
            sample_pt = Point(x_sample, y_sample)
            detected_pt = DetectedPt(geometry=sample_pt, distance=i * SAMPLE_SPACING)
            self.detected_pts.append(detected_pt)

    def compute_distance(self):
        """
        根据detected_pts计算道路的长度
        """
        # 计算起始点方式：从某个点T往后的T+5个点内,至少有三个点在道路边界内
        no_start = False
        start_id = -1
        for i in range(1, NUM_SAMPLES + 1):
            if i >= NUM_SAMPLES - 4:
                if start_id == -1:
                    no_start = True
                else:
                    break

            total_num = self.detected_pts[i].in_road + self.detected_pts[i+1].in_road + self.detected_pts[i+2].in_road + \
                        self.detected_pts[i+3].in_road + self.detected_pts[i+4].in_road
            if total_num >= 3:
                start_id = i
                break

        if no_start:
            self.road_width = 0
        else:
            # 计算结束点，：从某个点T往后的T+5个点内,至少有4个点在道路边界外
            for j in range(start_id, NUM_SAMPLES + 1):
                if j >= NUM_SAMPLES - 4:
                    self.road_width = (NUM_SAMPLES - start_id) * SAMPLE_SPACING
                    break

                total_num = self.detected_pts[j].in_road + self.detected_pts[j + 1].in_road + self.detected_pts[j + 2].in_road + \
                            self.detected_pts[j + 3].in_road + self.detected_pts[j + 4].in_road
                if total_num <= 1:
                    self.road_width = (j - start_id) * SAMPLE_SPACING
                    break


@dataclass
class SpecifiedRoad:
    """
    道路中心线、道路 pavement polygon 综合生成的道路结构
    """
    # 街道名字
    street_name_en: str = field(default="")
    street_name_tc: str = field(default="")
    # 原始道路数据(同名)
    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    # 与检测框裁切后的区域
    cropped_geometry: List[BaseGeometry] = field(default_factory=list)  # 裁切后
    detected_lines: List[DetectedLine] = field(default_factory=list)  # 裁切道路
    # 是否大于4.5m
    larger_than_demand_state: bool = field(default=False)
    average_road_width: float = field(default=float)


if __name__ == "__main__":
    pass