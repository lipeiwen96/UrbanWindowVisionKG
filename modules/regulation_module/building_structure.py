"""
Building Structure
"""
import enum
import math
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from typing import List
from typing import Union

from shapely import geometry
from shapely.affinity import affine_transform
from shapely.affinity import rotate
from shapely.geometry import LineString
from shapely.geometry import mapping
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union

from algorithm_module import generation_config
from modules.generate_module.config_schema import BuildingConfig


@dataclass(order=True)
class SiteStructure:
    # 存放，但不会用于计算
    # target住宅容积率
    input_dpr: float = field(default=float)
    # target非住宅容积率
    input_ndpr: float = field(default=float)
    # target住宅密度
    input_dsc: float = field(default=float)
    # target非住宅建筑密度
    input_ndsc: float = field(default=float)
    # 用户传入的指标
    site_area: float = field(default=100)  # 用户传入的数值
    site_plot_ratio: float = field(default=float)  # 容积率，这里暂时不分nd和d
    site_site_coverage: float = field(default=float)  # 建筑密度，这里暂时不分nd和d
    # 图形信息
    site_geometry: Polygon = field(default_factory=Polygon)
    inside_unplaced_area_list: List[Polygon] = field(default_factory=list)
    # fast_road_line_list: List[LineString] = field(default_factory=list)  # 4.5m道路线
    normal_road_line_list: List[LineString] = field(
        default_factory=list)  # 小于4.5m道路线

    @cached_property
    def bounds(self):
        return self.site_geometry.bounds

    @cached_property
    def area(self):
        return round(self.site_area, 2)

    @cached_property
    def buildable_geometry(self) -> Union[Polygon, MultiPolygon]:
        return self.site_geometry.difference(
            unary_union(self.inside_unplaced_area_list))

    @cached_property
    def buildable_boundary(self):
        return self.buildable_geometry.boundary

    @cached_property
    def reserved_geometry(self):
        return unary_union(self.inside_unplaced_area_list)

    def to_json(self):
        return {
            "input_dpr": self.input_dpr,
            "input_ndpr": self.input_ndpr,
            "input_dsc": self.input_dsc,
            "input_ndsc": self.input_ndsc,
            "site_area": self.site_area,
            "site_plot_ratio": self.site_plot_ratio,
            "site_site_coverage": self.site_site_coverage,
            "site_geometry": mapping(self.site_geometry),
            "inside_unplaced_area_list": [mapping(i) for i in self.inside_unplaced_area_list],
            "normal_road_line_list": [mapping(i) for i in self.normal_road_line_list]
            # "fast_road_line_list": [i.to_json() for i in self.fast_road_line_list],
        }

    @classmethod
    def from_json(cls, json_dict):
        return cls(
            input_dpr=json_dict["input_dpr"],
            input_ndpr=json_dict["input_ndpr"],
            input_dsc=json_dict["input_dsc"],
            input_ndsc=json_dict["input_ndsc"],
            site_area=json_dict["site_area"],
            site_plot_ratio=json_dict["site_plot_ratio"],
            site_site_coverage=json_dict["site_site_coverage"],
            site_geometry=geometry.shape(json_dict["site_geometry"]),
            inside_unplaced_area_list=[geometry.shape(i) for i in json_dict["inside_unplaced_area_list"]],
            normal_road_line_list=[geometry.shape(i) for i in json_dict["normal_road_line_list"]],
            # fast_road_line_list=[LineString.from_json(i) for i in json_dict["fast_road_line_list"]],
        )


class WindowDistanceBoxType(enum.Enum):
    inside = "inside"
    outside_normal_road = "outside_normal_road"
    outside_fast_road = "outside_fast_road"


# 窗户结构，完全不用管
@dataclass(order=True)
class WindowStructure:
    window_habitat_type: bool = field(
        default=True)  # habitat / others (含kitchen/office)
    fixed_window_line: LineString = field(default_factory=LineString)  # 这个不会变
    line_length: float = field(default=2.3)  # 窗线长度
    # 以下为会变动的内容
    window_line: LineString = field(default_factory=LineString)
    start_pt: Point = field(default_factory=Point)  # 逆时针方向
    end_pt: Point = field(default_factory=Point)
    normal_vector: tuple = field(default=tuple)
    # 有三种窗间距框
    distance_box_type: WindowDistanceBoxType = field(
        default=WindowDistanceBoxType.inside)  # inside / outside_normal_road / outside_fast_road
    # 第1种，面向园区内部:inside
    inside_distance_box: Polygon = field(default_factory=Polygon)
    # 第2种，面向<4.5m街道: outside_normal_road
    outside_normal_road_distance_box: Polygon = field(default_factory=Polygon)

    # 第3种,面向>=4.5m街道, 不需要框: outside_fast_road

    def initial_window(self, line: LineString):
        self.fixed_window_line = line

    # 刷新楼基线
    def renew(self, dx, dy, angle, height):
        # 更新图形
        self.window_line = affine_transform(
            rotate(self.fixed_window_line, angle=angle, origin=Point(0, 0)),
            [1, 0, 0, 1, dx, dy])
        self.__create_window_params(self.window_line, height)

    # 根据新的楼位置和窗基线，刷新间距框
    def __create_window_params(self, line: LineString, height: float):
        self.start_pt = Point(line.coords[0])
        self.end_pt = Point(line.coords[1])
        self.window_line = line
        self.line_length = self.window_line.length
        # 计算法向量(顺时针90)
        dx = self.end_pt.x - self.start_pt.x
        dy = self.end_pt.y - self.start_pt.y
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        self.normal_vector = (dy / magnitude, -dx / magnitude)  # 顺时针旋转90度得到法向量
        # 第1种，面向园区内部:inside -----
        inside_window_distance = height / 3 if self.window_habitat_type else height / 4
        inside_window_distance = max(21 / self.line_length,
                                     inside_window_distance)
        inside_window_distance = inside_window_distance * generation_config.DISTANCE_BOX_FACTOR
        # 生成窗间距框
        st_pt_offset = Point(
            self.start_pt.x + inside_window_distance * self.normal_vector[0],
            self.start_pt.y + inside_window_distance * self.normal_vector[1])
        end_pt_offset = Point(
            self.end_pt.x + inside_window_distance * self.normal_vector[0],
            self.end_pt.y + inside_window_distance * self.normal_vector[1])
        self.inside_distance_box = Polygon(
            [self.start_pt, st_pt_offset, end_pt_offset, self.end_pt,
             self.start_pt])
        # 第2种，面向<4.5m街道: outside_normal_road -----
        outside_normal_road_window_distance = height / 6 if self.window_habitat_type else height / 8
        outside_normal_road_window_distance = max(21 / self.line_length,
                                                  outside_normal_road_window_distance)
        # 生成窗间距框
        st_pt_offset = Point(
            self.start_pt.x + outside_normal_road_window_distance *
            self.normal_vector[0],
            self.start_pt.y + outside_normal_road_window_distance *
            self.normal_vector[1])
        end_pt_offset = Point(
            self.end_pt.x + outside_normal_road_window_distance *
            self.normal_vector[0],
            self.end_pt.y + outside_normal_road_window_distance *
            self.normal_vector[1])
        self.outside_normal_road_distance_box = Polygon(
            [self.start_pt, st_pt_offset, end_pt_offset, self.end_pt,
             self.start_pt])


# 意义不大，为了跑通
@dataclass
class BuildingTemplate:
    building_name: str
    building_geometry: Polygon
    first_floor_area: float
    habitat_window_list: List[LineString]
    others_window_list: List[LineString]
    site: SiteStructure
    building_config: BuildingConfig

    def to_json(self):
        return {
            "building_name": self.building_name,
            "building_geometry": mapping(self.building_geometry),
            "first_floor_area": self.first_floor_area,
            "habitat_window_list": [mapping(i) for i in self.habitat_window_list],
            "others_window_list": [mapping(i) for i in self.others_window_list],
            "site": self.site.to_json(),
            "building_config": self.building_config.to_json()
        }

    @classmethod
    def from_json(cls, json_dict):
        return cls(
            building_name=json_dict["building_name"],
            building_geometry=geometry.shape(json_dict["building_geometry"]),
            first_floor_area=json_dict["first_floor_area"],
            habitat_window_list=[geometry.shape(i) for i in json_dict["habitat_window_list"]],
            others_window_list=[geometry.shape(i) for i in json_dict["others_window_list"]],
            site=SiteStructure.from_json(json_dict["site"]),
            building_config=BuildingConfig.from_json(json_dict["building_config"])
        )


# 核心结构，建筑楼型
# 初始化：轮廓、窗户、所有检测框、高度
@dataclass(order=True)
class BuildingStructure:
    # 初始化信息
    building_name: str = field(default=str)
    # 低层商业
    non_domestic_geometry: Polygon = field(default_factory=Polygon)
    non_domestic_floor_height: float = field(default=5.00)  # 用户传入
    non_domestic_storey: int = field(default=2)  # 用户传入
    non_domestic_height: float = field(default=10.00)  # 用户传入
    # 上部住宅
    domestic_geometry: Polygon = field(default_factory=Polygon)  # POC阶段底商与住宅部分是一个图形
    domestic_floor_height: float = field(default=3.15)  # 用户传入
    default_window_height: float = field(default=1)  # 住宅窗高度，对于香港规范：notional sill level  默认
    # 楼栋参数
    min_tower_distance: float = field(default=5.00)  # 用户传入
    min_building_height: float = field(default=20.00)  # 用户传入
    max_building_height: float = field(default=100.00)  # 用户传入
    # 标准层面积
    first_floor_area: float = field(default=200.00)  # 用户传入

    angle: float = field(default=0)
    # 窗户信息
    window_list: List[WindowStructure] = field(default_factory=list)
    # 存储地块信息
    site: SiteStructure = field(default_factory=SiteStructure)
    # 变化信息，传入dx，dy，angle后
    domestic_storey: int = field(default=10)  # 住宅部分楼层数
    domestic_height: float = field(default=31.5)  # 住宅部分楼层高度
    building_total_height: float = field(default=31.5)  # 楼高
    true_position_geometry: Polygon = field(default_factory=Polygon)  # 偏移后的图形
    # 魏老板调用
    min_tower_distance_geometry: Polygon = field(default_factory=Polygon)  # 间距框
    window_building_distance_box: List[Polygon] = field(
        default_factory=list)  # 你可以相信的窗间距框，专用于计算楼栋相交
    window_site_distance_box: List[Polygon] = field(
        default_factory=list)  # 你可以相信的窗间距框，专用于计算基地相交，包含建筑轮廓

    @cached_property
    def total_window_building_distance_shadow(self):
        return unary_union(self.window_building_distance_box).union(self.min_tower_distance_geometry)

    @cached_property
    def center(self):
        return self.geometry.centroid

    @cached_property
    def geometry(self):
        return self.true_position_geometry

    @cached_property
    def min_tower_buffer(self):
        return self.geometry.buffer(self.min_tower_distance)

    @property
    def total_storey(self):
        return self.non_domestic_storey + self.domestic_storey

    @property
    def non_domestic_total_height(self):
        return self.non_domestic_storey * self.non_domestic_floor_height

    @property
    def domestic_total_height(self):
        return self.domestic_storey * self.domestic_floor_height

    @property
    def total_height(self):
        return self.domestic_total_height + self.non_domestic_total_height

    @property
    def min_d_floor_number(self):
        return math.ceil((self.min_building_height - (self.non_domestic_floor_height * self.non_domestic_storey)) / self.domestic_floor_height)

    @property
    def max_d_floor_number(self):
        return math.floor((self.max_building_height - (self.non_domestic_floor_height * self.non_domestic_storey)) / self.domestic_floor_height)

    @property
    def min_total_floor_number(self):
        return self.min_d_floor_number + self.non_domestic_storey

    @property
    def max_total_floor_number(self):
        return self.max_d_floor_number + self.non_domestic_storey

    @classmethod
    def create_by_template(cls,
                           building_template: BuildingTemplate,
                           dx: float = 0,
                           dy: float = 0,
                           angle: float = 0,
                           domestic_storey: int = 10) -> 'BuildingStructure':
        building = BuildingStructure(building_name=building_template.building_name)
        building.apply_config(building_template.building_config)
        building.inititialize_building(building_template.building_geometry,
                                       building_template.habitat_window_list,
                                       building_template.others_window_list,
                                       building_template.site)
        building.renew_building(dx, dy, angle, domestic_storey)
        return building

    def apply_config(self, building_config: BuildingConfig):
        self.building_name = building_config.choose_plan_name
        self.non_domestic_storey = building_config.nd_storey
        self.non_domestic_floor_height = building_config.floor_nd_height
        self.non_domestic_height = self.non_domestic_storey * self.non_domestic_floor_height
        self.first_floor_area = building_config.first_floor_area
        self.min_tower_distance = building_config.min_tower_seperation

    def inititialize_building(self,
                              building_geometry: Polygon,
                              habitat_window_list: List[LineString],
                              others_window_list: List[LineString],
                              site: SiteStructure):
        # 初始化参数
        self.non_domestic_height = self.non_domestic_floor_height * self.non_domestic_storey
        # 其他信息
        self.non_domestic_geometry = building_geometry
        self.domestic_geometry = building_geometry
        self.site = site
        # 生成窗信息
        self.__inititialize_windows(habitat_window_list, others_window_list)

    def __inititialize_windows(self, habitat_window_list: List[LineString],
                               others_window_list: List[LineString]):
        # 初始化窗户数据结构
        for habitat_window in habitat_window_list:
            window = WindowStructure(window_habitat_type=True)
            window.initial_window(habitat_window)
            self.window_list.append(window)
        for others_window in others_window_list:
            window = WindowStructure(window_habitat_type=False)
            window.initial_window(others_window)
            self.window_list.append(window)

    # 传入偏移量
    def renew_building(self, dx, dy, angle, domestic_storey):
        self.domestic_storey = domestic_storey
        self.domestic_height = domestic_storey * self.domestic_floor_height
        self.angle = angle
        # # 如果楼楼大于最低值或大于最大值，予以修正
        self.building_total_height = self.domestic_height + self.non_domestic_height
        # if self.building_total_height < self.min_building_height:
        #     self.domestic_height = self.min_building_height - self.non_domestic_height
        # if self.building_total_height > self.max_building_height:
        #     self.domestic_height = self.max_building_height - self.non_domestic_height
        # 更新图形
        self.true_position_geometry = affine_transform(
            rotate(self.domestic_geometry, angle=angle,
                   origin=self.domestic_geometry.centroid),
            [1, 0, 0, 1, dx, dy])
        # 生成间距框
        # 请注意，这里的间距要除以2
        self.min_tower_distance_geometry = self.true_position_geometry.buffer(
            self.min_tower_distance / 2, cap_style=2, join_style=2)
        # 更新窗户
        for window in self.window_list:
            window.renew(dx, dy, angle,
                         self.domestic_height - self.default_window_height)
        # 生成距离框
        self.__generate_window_distance_box()
        return self

    def __generate_window_distance_box(self):
        self.window_building_distance_box = []
        self.window_site_distance_box = [self.true_position_geometry]
        for window in self.window_list:
            # 首先判断最长的内部检测框与场地边界是否相交：
            if self.site.site_geometry.contains(window.inside_distance_box):
                # 面向地块内部
                # print("面向地块内部")
                window.distance_box_type = WindowDistanceBoxType.inside
                self.window_building_distance_box.append(
                    window.inside_distance_box)
            else:
                # 存在贴近边界的状况
                if not BuildingStructure.check_cross_status(
                        window.inside_distance_box,
                        self.site.normal_road_line_list):
                    # 面向>=4.5m道路
                    # print("面向>=4.5m道路")
                    window.distance_box_type = WindowDistanceBoxType.outside_fast_road
                    self.window_building_distance_box.append(
                        window.inside_distance_box)
                else:
                    # 面向<4.5m道路
                    # print("面向<4.5m道路")
                    window.distance_box_type = WindowDistanceBoxType.outside_normal_road
                    self.window_building_distance_box.append(
                        window.inside_distance_box)
                    self.window_site_distance_box.append(
                        window.outside_normal_road_distance_box)

    @staticmethod
    def check_cross_status(geo, line_list):
        status = False
        for target_line in line_list:
            if target_line.crosses(geo):
                status = True
                break
        return status

    def building_intersects(self, other_building: 'BuildingStructure') -> bool:
        return (any(p.intersects(other_building.geometry)
                    for p in self.window_building_distance_box)
                or other_building.total_window_building_distance_shadow.intersects(
                    self.geometry))

    def site_intersects(self) -> bool:
        # TODO 还要加上基地间距框
        return any(self.site.buildable_boundary.intersects(box)
                   for box in self.window_site_distance_box)
