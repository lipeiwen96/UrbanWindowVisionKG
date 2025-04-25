"""
建筑编号信息类
"""
from dataclasses import field, dataclass
from typing import List, Dict
from shapely.geometry import Point, Polygon


@dataclass
class BuildingIdText:
    id: int = field(default=int)  # 建筑序号
    id_text: str = field(default=str)  # 建筑序号文字：如T1
    geometry: Polygon = field(default_factory=Polygon)
    point: Point = field(default_factory=Point)  # 文字所在位置
    building_height: float = field(default=float)  # 建筑高度
    building_floor_num: float = field(default=float)  # 建筑层数
    d_floor_num: int = field(default=int)  # 建筑层数
    nd_floor_num: int = field(default=int)  # 建筑层数
    d_floor_height: float = field(default=float)  # 建筑层高
    nd_floor_height: float = field(default=float)  # 建筑层高

    def init(self, id: int, geometry: Polygon, point: Point, d_floor_num: int, d_floor_height: float, nd_floor_num: int, nd_floor_height: float):
        self.id = id
        self.point = point
        self.geometry = geometry
        self.d_floor_num = d_floor_num
        self.d_floor_height = d_floor_height
        self.nd_floor_num = nd_floor_num
        self.nd_floor_height = nd_floor_height
        # 自动更新
        self.id_text = f"T{self.id}"
        self.building_floor_num = self.d_floor_num + self.nd_floor_num
        self.building_height = self.d_floor_num * self.d_floor_height + self.nd_floor_num * self.nd_floor_height

    @property
    def to_dict(self):
        return {
            "id": self.id,
            "id_text": self.id_text,
            "geometry": self.geometry.wkt,
            "point": self.point.wkt,
            "building_height": self.building_height,
            "building_floor_num": self.building_floor_num,
            "d_floor_num": self.d_floor_num,
            "nd_floor_num": self.nd_floor_num,
            "d_floor_height": self.d_floor_height,
            "nd_floor_height": self.nd_floor_height,
        }
