"""
地图处理器 V2.0
@Peiwen
"""
import copy
from dataclasses import field, dataclass
from typing import List
import json
import os
from map_system.tile_genenrator.map_structure import MapRoadCenterLine, MapBuilding, MapLot, MapRoadPolygon
from map_system.coord_convertor.coords_convertor import CoordConvertor
from shapely.geometry import Polygon, Point, LineString, GeometryCollection, box, MultiPolygon, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from map_system.tile_genenrator.map_cropper import MapCropper
from art_public_modules.art_utils.common_utils.path_utils import PathUtils


@dataclass
class TileSystemGenerator:
    road_center_line_map_path: str = field(default=str)
    building_map_path: str = field(default=str)
    lot_map_path: str = field(default=str)
    gla_map_path: str = field(default=str)
    road_polygon_path: str = field(default=str)

    # 读入原始地图数据
    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    building_list: List[MapBuilding] = field(default_factory=list)
    lot_list: List[MapLot] = field(default_factory=list)

    # 裁切后的相交数据
    tile_boundary: Polygon = field(default_factory=Polygon)
    cropped_road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    cropped_road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    cropped_building_list: List[MapBuilding] = field(default_factory=list)
    cropped_lot_list: List[MapLot] = field(default_factory=list)
    cropped_road_center_line_id_list: List[int] = field(default_factory=list)
    cropped_road_polygon_id_list: List[int] = field(default_factory=list)
    cropped_building_id_list: List[int] = field(default_factory=list)
    cropped_lot_id_list: List[int] = field(default_factory=list)
    cropped_gla_id_list: List[int] = field(default_factory=list)

    def read_row_data(self, road_center_line_map_path: str, building_map_path: str, lot_map_path: str, gla_map_path: str, road_polygon_path: str):
        # 保存地址
        self.road_center_line_map_path = road_center_line_map_path
        self.building_map_path = building_map_path
        self.lot_map_path = lot_map_path
        self.gla_map_path = gla_map_path
        self.road_polygon_path = road_polygon_path
        for road_center_line_map in TileSystemGenerator.read_map_json(road_center_line_map_path):
            try:
                road_center_line = MapRoadCenterLine()
                road_center_line.init(road_center_line_map)
                self.road_center_line_list.append(road_center_line)
            except Exception as e:
                print(f'发生道路中心线读取错误:{e}, 跳过')
        print(f"共读入{len(self.road_center_line_list)}个道路中心线的数据.....")
        for road_polygon_map in TileSystemGenerator.read_map_json(road_polygon_path):
            try:
                road_polygon = MapRoadPolygon()
                road_polygon.init(road_polygon_map)
                self.road_polygon_list.append(road_polygon)
            except Exception as e:
                print(f'发生道路图形读取错误:{e}, 跳过')
        print(f"共读入{len(self.road_polygon_list)}个道路图形的数据.....")
        for building_map in TileSystemGenerator.read_map_json(building_map_path):
            try:
                building = MapBuilding()
                building.init(building_map)
                self.building_list.append(building)
            except Exception as e:
                print(f'发生建筑读取错误:{e}, 跳过')
        print(f"共读入{len(self.building_list)}个建筑图形的数据.....")
        # print(GeometryCollection([building.geometry for building in self.building_list[:1000]]))
        for lot_map in TileSystemGenerator.read_map_json(lot_map_path):
            try:
                lot = MapLot()
                lot.init(lot_map)
                self.lot_list.append(lot)
            except Exception as e:
                print(f'发生地块读取错误:{e}, 跳过')
        for lot_map in TileSystemGenerator.read_map_json(gla_map_path):
            try:
                lot = MapLot()
                lot.init(lot_map)
                self.lot_list.append(lot)
            except Exception as e:
                print(f'发生地块读取错误:{e}, 跳过')
        print(f"共读入{len(self.lot_list)}个地块图形的数据.....")

    @staticmethod
    def read_map_json(path: str):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data["features"]

    def simplify_map_by_boundary(self, tile_boundary: Polygon, tile_name: str = "Xkool-Area-1", base_root: str = ""):
        # 创建项目边界
        self.tile_boundary = tile_boundary
        minx, miny, maxx, maxy = self.tile_boundary.bounds
        print(f"【{tile_name} 项目范围框：{round(maxx - minx)}m * {round(maxy -miny)}m, 面积约{round(self.tile_boundary.area)}㎡】")
        print("正在开始裁切地图...")

        # 裁切地图
        self.__renew_data_from_tile_boundary()

        # 导出数据
        self.__export_processed_and_simplified_data(tile_name=tile_name, base_root=base_root)

        return len(self.cropped_road_center_line_list), len(self.cropped_road_polygon_list), len(self.cropped_building_list), len(self.cropped_lot_list)

    def __export_processed_and_simplified_data(self, tile_name: str = "Xkool-Area-1", base_root: str = ""):
        if base_root == "":
            base_root = os.path.abspath(os.path.join("library", "HK_map", "processed", tile_name))

        PathUtils.check_and_create_dir(base_root, whether_clean=False)

        # 创建区块数据源
        # 导出处理后的道路中心线数据
        data = {"features": []}
        for road_center_line in self.cropped_road_center_line_list:
            data["features"].append(road_center_line.row_data)
        road_center_line_map_json_path = os.path.join(base_root, "road_center_line_map.json")
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(road_center_line_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{road_center_line_map_json_path}的道路中心线数据生成成功")

        # 导出处理后的道路图形数据
        data = {"features": []}
        for road_polygon in self.cropped_road_polygon_list:
            data["features"].append(road_polygon.row_data)
        road_polygon_map_json_path = os.path.join(base_root, "road_polygon_map.json")
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(road_polygon_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{road_polygon_map_json_path}的道路图形数据生成成功")

        # 导出处理后的建筑数据
        data = {"features": []}
        for building in self.cropped_building_list:
            data["features"].append(building.row_data)
        building_map_json_path = os.path.join(base_root, "building_map.json")
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(building_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{building_map_json_path}的建筑数据生成成功")

        # 导出处理后的地块LOT数据
        lot_data = {"features": []}
        gla_data = {"features": []}
        for lot in self.cropped_lot_list:
            if lot.is_GLA:
                gla_data["features"].append(lot.row_data)
            else:
                lot_data["features"].append(lot.row_data)
        lot_map_json_path = os.path.join(base_root, "lot_map.json")
        json_str = json.dumps(lot_data, indent=4, ensure_ascii=False)
        with open(lot_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{lot_map_json_path}的地块LOT数据生成成功")
        gla_map_json_path = os.path.join(base_root, "gla_map.json")
        json_str = json.dumps(gla_data, indent=4, ensure_ascii=False)
        with open(gla_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{gla_map_json_path}的地块GLA数据生成成功")
        print(f"【{tile_name} 地图导出已经完成】")

    def __renew_data_from_tile_boundary(self):
        """
        根据项目的Tile边界，保留与Tile边界相交的地图数据，不相交的予以过滤
        """
        # 更新道路中心线
        self.cropped_road_center_line_list = []
        for road_center_line in self.road_center_line_list:
            if road_center_line.geometry.intersects(self.tile_boundary):
                self.cropped_road_center_line_list.append(road_center_line)
        self.cropped_road_center_line_id_list = [road_center_line.object_id for road_center_line in self.cropped_road_center_line_list]
        print(f"保留的道路中心线数量: {len(self.cropped_road_center_line_list)}, id数量: {len(self.cropped_road_center_line_id_list)}")

        # 更新道路图形
        self.cropped_road_polygon_list = []
        for road_polygon in self.road_polygon_list:
            if road_polygon.geometry.intersects(self.tile_boundary):
                self.cropped_road_polygon_list.append(road_polygon)
        self.cropped_road_polygon_id_list = [road_pl.object_id for road_pl in self.cropped_road_polygon_list]
        print(f"保留的道路图形数量: {len(self.cropped_road_polygon_list)}, id数量: {len(self.cropped_road_polygon_id_list)}")

        # 更新建筑图形
        self.cropped_building_list = []
        for building in self.building_list:
            if building.geometry.intersects(self.tile_boundary):
                self.cropped_building_list.append(building)
        self.cropped_building_id_list = [building.object_id for building in self.cropped_building_list]
        print(f"保留的建筑图形数量: {len(self.cropped_building_list)}, id数量: {len(self.cropped_building_id_list)}")

        # 更新地块图形
        self.cropped_lot_list = []
        for lot in self.lot_list:
            if lot.geometry.intersects(self.tile_boundary):
                self.cropped_lot_list.append(lot)
        self.cropped_lot_id_list = [lot.lot_id for lot in self.cropped_lot_list if not lot.is_GLA]
        self.cropped_gla_id_list = [lot.gla_id for lot in self.cropped_lot_list if lot.is_GLA]
        print(f"保留的地块数量: {len(self.cropped_lot_list)}", "LOT id 数量:", len(self.cropped_lot_id_list), "+ GLA id 数量:", len(self.cropped_gla_id_list))

    @property
    def cropped_building_geo(self):
        return GeometryCollection([building.geometry for building in self.cropped_building_list])

    @property
    def cropped_road_center_line_geo(self):
        return GeometryCollection([road_center_line.geometry for road_center_line in self.cropped_road_center_line_list])

    @property
    def cropped_road_polygon_geo(self):
        return GeometryCollection([road_pl.geometry for road_pl in self.cropped_road_polygon_list])

    @property
    def cropped_lot_geo(self):
        return GeometryCollection([lot.geometry for lot in self.cropped_lot_list])


