"""
地图瓦片生成器 V1 -- 弃用
 - 输入项目范围框
 - 读入地图分层数据（这里有两个方式，读入完整的CSDI数据，及xkool已经过滤的数据）
输出项目范围框内的瓦片数据
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
class Tile:
    site_boundary: Polygon = field(default_factory=Polygon)  # 项目场地边界
    tile_boundary: Polygon = field(default_factory=Polygon)  # 项目范围框边界
    left_bottom_nor: float = field(default=float)
    left_bottom_eas: float = field(default=float)
    right_top_nor: float = field(default=float)
    right_top_eas: float = field(default=float)
    nor_dis: float = field(default=float)
    eas_dis: float = field(default=float)
    area: float = field(default=float)

    def init_by_map_box(self, left_bottom_eas: float, left_bottom_nor: float, right_top_eas: float, right_top_nor: float):
        self.left_bottom_eas = left_bottom_eas
        self.left_bottom_nor = left_bottom_nor
        self.right_top_eas = right_top_eas
        self.right_top_nor = right_top_nor
        self.__init_properties()

    def init_by_site_boundary(self, site_boundary: Polygon):
        self.site_boundary = site_boundary
        # 项目boundary_box
        minx, miny, maxx, maxy = self.site_boundary.bounds
        x_dis = 220
        y_dis = 120
        self.left_bottom_eas = minx - x_dis
        self.left_bottom_nor = miny - y_dis
        self.right_top_eas = maxx + x_dis
        self.right_top_nor = maxy + y_dis
        self.__init_properties()

    def __init_properties(self):
        self.tile_boundary = box(self.left_bottom_eas, self.left_bottom_nor, self.right_top_eas, self.right_top_nor)
        self.nor_dis = self.right_top_nor - self.left_bottom_nor
        self.eas_dis = self.right_top_eas - self.left_bottom_eas
        self.area = self.nor_dis * self.eas_dis


@dataclass
class TileMapGenerator:
    tile: Tile = field(default_factory=Tile)

    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    building_list: List[MapBuilding] = field(default_factory=list)
    lot_list: List[MapLot] = field(default_factory=list)

    new_road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    new_road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    new_building_list: List[MapBuilding] = field(default_factory=list)
    new_lot_list: List[MapLot] = field(default_factory=list)

    geo_list: List[BaseGeometry] = field(default_factory=list)
    geo_collection: GeometryCollection = field(default_factory=GeometryCollection)

    road_center_line_map_path: str = field(default=str)
    building_map_path: str = field(default=str)
    lot_map_path: str = field(default=str)
    gla_map_path: str = field(default=str)
    road_polygon_path: str = field(default=str)

    def read_row_road_data(self, road_center_line_map_path: str, road_polygon_path: str):
        for road_center_line_map in TileMapGenerator.read_map_json(road_center_line_map_path):
            try:
                road_center_line = MapRoadCenterLine()
                road_center_line.init(road_center_line_map)
                self.road_center_line_list.append(road_center_line)
            except Exception as e:
                print(f'发生道路中心线读取错误:{e}, 跳过')
        print(f"共读入{len(self.road_center_line_list)}个道路中心线的数据.....")
        for road_polygon_map in TileMapGenerator.read_map_json(road_polygon_path):
            try:
                road_polygon = MapRoadPolygon()
                road_polygon.init(road_polygon_map)
                self.road_polygon_list.append(road_polygon)
            except Exception as e:
                print(f'发生地块读取错误:{e}, 跳过')
        print(f"共读入{len(self.road_polygon_list)}个道路图形的数据.....")

    def read_row_data(self, road_center_line_map_path: str, building_map_path: str, lot_map_path: str, gla_map_path: str, road_polygon_path: str):
        # 保存地址
        self.road_center_line_map_path = road_center_line_map_path
        self.building_map_path = building_map_path
        self.lot_map_path = lot_map_path
        self.gla_map_path = gla_map_path
        self.road_polygon_path = road_polygon_path
        for road_center_line_map in TileMapGenerator.read_map_json(road_center_line_map_path):
            try:
                road_center_line = MapRoadCenterLine()
                road_center_line.init(road_center_line_map)
                self.road_center_line_list.append(road_center_line)
            except Exception as e:
                print(f'发生道路中心线读取错误:{e}, 跳过')
        print(f"共读入{len(self.road_center_line_list)}个道路中心线的数据.....")
        for road_polygon_map in TileMapGenerator.read_map_json(road_polygon_path):
            try:
                road_polygon = MapRoadPolygon()
                road_polygon.init(road_polygon_map)
                self.road_polygon_list.append(road_polygon)
            except Exception as e:
                print(f'发生地块读取错误:{e}, 跳过')
        print(f"共读入{len(self.road_polygon_list)}个道路图形的数据.....")
        for building_map in TileMapGenerator.read_map_json(building_map_path):
            try:
                building = MapBuilding()
                building.init(building_map)
                self.building_list.append(building)
            except Exception as e:
                print(f'发生建筑读取错误:{e}, 跳过')
        print(f"共读入{len(self.building_list)}个建筑图形的数据.....")
        for lot_map in TileMapGenerator.read_map_json(lot_map_path):
            try:
                lot = MapLot()
                lot.init(lot_map)
                self.lot_list.append(lot)
            except Exception as e:
                print(f'发生地块读取错误:{e}, 跳过')
        for lot_map in TileMapGenerator.read_map_json(gla_map_path):
            try:
                lot = MapLot()
                lot.init(lot_map)
                self.lot_list.append(lot)
            except Exception as e:
                print(f'发生地块读取错误:{e}, 跳过')
        print(f"共读入{len(self.lot_list)}个地块图形的数据.....")

    @property
    def road_polygon_type_set(self):
        return set([road_pl.feat_type for road_pl in self.road_polygon_list])
    @property
    def road_center_line_id_list(self):
        return [road_center_line.object_id for road_center_line in self.road_center_line_list]
    @property
    def road_polygon_id_list(self):
        return [road_pl.object_id for road_pl in self.road_polygon_list]
    @property
    def building_id_list(self):
        return [building.object_id for building in self.building_list]
    @property
    def lot_id_list(self):
        return [lot.lot_id for lot in self.lot_list if not lot.is_GLA]
    @property
    def gla_id_list(self):
        return [lot.gla_id for lot in self.lot_list if lot.is_GLA]
    @property
    def map_geometry_collection(self):
        geo_list = []
        geo_list.extend([road_center_line.geometry for road_center_line in self.road_center_line_list])
        geo_list.extend([road_pl.geometry for road_pl in self.road_polygon_list])
        geo_list.extend([building.geometry for building in self.building_list])
        geo_list.extend([lot.geometry for lot in self.lot_list])
        return GeometryCollection(geo_list)

    def __export_processed_and_simplified_data(self, tile_name: str = "Xkool-Area-1", base_root: str = ""):
        if base_root == "":
            base_root = os.path.abspath(os.path.join("library", "HK_map", "processed", tile_name))

        PathUtils.check_and_create_dir(base_root, whether_clean=False)

        # 创建区块数据源
        # 导出处理后的道路中心线数据
        data = {"features": []}
        for road_center_line_map in TileMapGenerator.read_map_json(self.road_center_line_map_path):
            if road_center_line_map["properties"]["OBJECTID"] in self.road_center_line_id_list:
                data["features"].append(road_center_line_map)
        road_center_line_map_json_path = os.path.join(base_root, "road_center_line_map.json")
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(road_center_line_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{road_center_line_map_json_path}的道路中心线数据生成成功")

        # 导出处理后的道路图形数据
        data = {"features": []}
        for road_polygon_map in TileMapGenerator.read_map_json(self.road_polygon_path):
            if road_polygon_map["properties"]["OBJECTID"] in self.road_polygon_id_list:
                data["features"].append(road_polygon_map)
        road_polygon_map_json_path = os.path.join(base_root, "road_polygon_map.json")
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(road_polygon_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{road_polygon_map_json_path}的道路图形数据生成成功")

        # 导出处理后的建筑数据
        data = {"features": []}
        for building_map in TileMapGenerator.read_map_json(self.building_map_path):
            if building_map["properties"]["OBJECTID"] in self.building_id_list:
                data["features"].append(building_map)
        building_map_json_path = os.path.join(base_root, "building_map.json")
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(building_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{building_map_json_path}的建筑数据生成成功")

        # 导出处理后的地块LOT数据
        data = {"features": []}
        for lot_map in TileMapGenerator.read_map_json(self.lot_map_path):
            if lot_map["properties"]["LOTID"] in self.lot_id_list:
                data["features"].append(lot_map)
        lot_map_json_path = os.path.join(base_root, "lot_map.json")
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(lot_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{lot_map_json_path}的地块LOT数据生成成功")

        # 导出处理后的地块GLA数据
        data = {"features": []}
        for gla_map in TileMapGenerator.read_map_json(self.gla_map_path):
            if gla_map["properties"]["GLAID"] in self.gla_id_list:
                data["features"].append(gla_map)
        gla_map_json_path = os.path.join(base_root, "gla_map.json")
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(gla_map_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"{gla_map_json_path}的地块GLA数据生成成功")
        print(f"【{tile_name} 地图导出已经完成】")

    def __renew_data_from_tile_boundary(self):
        """
        根据项目的Tile边界，保留与Tile边界相交的地图数据，不相交的予以过滤
        """
        # 更新道路中心线
        self.new_road_center_line_list = []
        for road_center_line in self.road_center_line_list:
            if road_center_line.geometry.intersects(self.tile.tile_boundary):
                self.new_road_center_line_list.append(road_center_line)
        # self.road_center_line_list.clear()
        # self.road_center_line_list = self.new_road_center_line_list
        # 更新道路图形
        self.new_road_polygon_list = []
        for road_polygon in self.road_polygon_list:
            if road_polygon.geometry.intersects(self.tile.tile_boundary):
                self.new_road_polygon_list.append(road_polygon)
        # self.road_polygon_list.clear()
        # self.road_polygon_list = self.new_road_polygon_list
        # 更新建筑图形
        self.new_building_list = []
        for building in self.building_list:
            if building.geometry.intersects(self.tile.tile_boundary):
                self.new_building_list.append(building)
        # self.building_list.clear()
        # self.building_list = self.new_building_list
        # 更新地块图形
        self.new_lot_list = []
        for lot in self.lot_list:
            if lot.geometry.intersects(self.tile.tile_boundary):
                self.new_lot_list.append(lot)
        # self.lot_list.clear()
        # self.lot_list = self.new_lot_list

    def simplify_map_by_boundary(self, left_bottom_pt: tuple, right_top_pt: tuple, is_WGS: bool = True, tile_name: str = "Xkool-Area-1", base_root: str = ""):
        # 创建项目边界
        if is_WGS:
            left_bottom_nor, left_bottom_eas = CoordConvertor.wgs_to_hk(left_bottom_pt[0], left_bottom_pt[1])
            right_top_nor, right_top_eas = CoordConvertor.wgs_to_hk(right_top_pt[0], right_top_pt[1])
            self.tile.init_by_map_box(left_bottom_eas, left_bottom_nor, right_top_eas, right_top_nor)
            print("读入WGS坐标，已转化为HK80坐标")
        else:
            self.tile.init_by_map_box(left_bottom_pt[1], left_bottom_pt[0], right_top_pt[1], right_top_pt[0])

        print(f"项目范围框：{round(self.tile.eas_dis)}m * {round(self.tile.nor_dis)}m, 面积约{round(self.tile.area)}㎡")
        print(self.tile.tile_boundary)

        self.__renew_data_from_tile_boundary()
        self.__export_processed_and_simplified_data(tile_name=tile_name, base_root=base_root)

    def crop_map_by_box(self, left_bottom_pt: tuple, right_top_pt: tuple, is_WGS: bool = True):
        # 创建项目边界
        if is_WGS:
            left_bottom_nor, left_bottom_eas = CoordConvertor.wgs_to_hk(left_bottom_pt[0], left_bottom_pt[1])
            right_top_nor, right_top_eas = CoordConvertor.wgs_to_hk(right_top_pt[0], right_top_pt[1])
            self.tile.init_by_map_box(left_bottom_eas, left_bottom_nor, right_top_eas, right_top_nor)
            print("读入WGS坐标，已转化为HK80坐标")
        else:
            self.tile.init_by_map_box(left_bottom_pt[1], left_bottom_pt[0], right_top_pt[1], right_top_pt[0])
        print(f"项目范围框：{round(self.tile.eas_dis)}m * {round(self.tile.nor_dis)}m, 面积约{round(self.tile.area)}㎡")
        print(self.tile.tile_boundary)

        self.__renew_data_from_tile_boundary()
        self.__generate_tile_data_model()
        self.__generate_tile_geo()

    def crop_map_by_site_boundary(self, site_boundary: Polygon, is_WGS: bool = True):
        # 创建项目边界
        if is_WGS:
            site_boundary = Polygon([Point(CoordConvertor.wgs_to_hk_reverse(coord[0], coord[1]))
                                     for coord in site_boundary.exterior.coords])
            print("读入WGS坐标，已转化为HK80坐标")

        self.tile.init_by_site_boundary(site_boundary)
        print(f"项目范围框：{round(self.tile.eas_dis)}m * {round(self.tile.nor_dis)}m, 面积约{round(self.tile.area)}㎡")
        print(self.tile.tile_boundary)

        self.__initial_site_model()
        # self.__generate_tile_data_model()

    def __initial_site_model(self):
        """
        初始化场地模型，包括：
         - 仅保留项目范围框内的数据
         - 删除项目场地内的地图数据
         输出为：
         DataModel
        """
        self.model = DataModel(name="tile_model")
        element = DataElement(geometry=self.tile.tile_boundary,
                              layer="Xkool_ProjectBoundary",
                              start_height=-20,
                              height=20)
        self.model.insert_element(element)
        element = DataElement(geometry=self.tile.site_boundary,
                              layer="Xkool_TargetSiteBoundary", start_height=0, height=1)
        self.model.insert_element(element)

        outcome = MapCropper.crop(self.building_list, site_boundary=self.tile.site_boundary,
                                  tile_box=self.tile.tile_boundary)
        for out in outcome:
            element = DataElement(geometry=out.geometry,
                                  layer="Xkool_SurroundBuilding",
                                  start_height=out.start_height,
                                  height=out.height)
            self.model.insert_element(element)
        outcome = MapCropper.crop(self.road_polygon_list, site_boundary=self.tile.site_boundary,
                                  tile_box=self.tile.tile_boundary)
        for out in outcome:
            element = DataElement(geometry=out.geometry,
                                  layer="Road")
            self.model.insert_element(element)
        outcome = MapCropper.crop(self.lot_list, site_boundary=self.tile.site_boundary,
                                  tile_box=self.tile.tile_boundary)
        for out in outcome:
            element = DataElement(geometry=out.geometry,
                                  layer="Xkool_PlotBoundary")
            self.model.insert_element(element)
        self.model.renew()

    def __generate_tile_data_model(self):
        self.model = DataModel(name="tile_model")
        element = DataElement(geometry=self.tile.tile_boundary,
                              layer="Box",
                              start_height=-20,
                              height=20)
        self.model.insert_element(element)
        for building in self.building_list:
            if building.geometry.intersects(self.tile.tile_boundary):
                clipped_geometry = building.geometry.intersection(self.tile.tile_boundary)
                if isinstance(clipped_geometry, LineString) or isinstance(clipped_geometry, Polygon):
                    element = DataElement(geometry=clipped_geometry,
                                          layer="Building",
                                          start_height=building.start_height,
                                          height=building.height)
                    self.model.insert_element(element)
                if isinstance(clipped_geometry, GeometryCollection) or isinstance(clipped_geometry, MultiPolygon) or isinstance(clipped_geometry, MultiLineString):
                    for clip_geo in clipped_geometry:
                        element = DataElement(geometry=clip_geo,
                                              layer="Building",
                                              start_height=building.start_height,
                                              height=building.height)
                        self.model.insert_element(element)
        for road_polygon in self.road_polygon_list:
            if road_polygon.geometry.intersects(self.tile.tile_boundary):
                clipped_geometry = road_polygon.geometry.intersection(self.tile.tile_boundary)
                if isinstance(clipped_geometry, LineString) or isinstance(clipped_geometry, Polygon):
                    element = DataElement(geometry=clipped_geometry,
                                          layer="Road",
                                          start_height=0, height=0.5)
                    self.model.insert_element(element)
                if isinstance(clipped_geometry, GeometryCollection) or isinstance(clipped_geometry, MultiPolygon) or isinstance(clipped_geometry, MultiLineString):
                    for clip_geo in clipped_geometry:
                        element = DataElement(geometry=clip_geo,
                                              layer="Road",
                                              start_height=0, height=0.5)
                        self.model.insert_element(element)
        self.model.renew()

    def __generate_tile_geo(self):
        for road_center_line in self.road_center_line_list:
            if road_center_line.geometry.intersects(self.tile.tile_boundary):
                clipped_geometry = road_center_line.geometry.intersection(self.tile.tile_boundary)
                if isinstance(clipped_geometry, LineString) or isinstance(clipped_geometry, Polygon):
                    self.geo_list.append(clipped_geometry)
                if isinstance(clipped_geometry, GeometryCollection) or isinstance(clipped_geometry, MultiPolygon) or isinstance(clipped_geometry, MultiLineString):
                    for clip_geo in clipped_geometry:
                        self.geo_list.append(clip_geo)
        for lot in self.lot_list:
            if lot.geometry.intersects(self.tile.tile_boundary):
                clipped_geometry = lot.geometry.intersection(self.tile.tile_boundary)
                if isinstance(clipped_geometry, LineString) or isinstance(clipped_geometry, Polygon):
                    self.geo_list.append(clipped_geometry)
                if isinstance(clipped_geometry, GeometryCollection) or isinstance(clipped_geometry, MultiPolygon) or isinstance(clipped_geometry, MultiLineString):
                    for clip_geo in clipped_geometry:
                        self.geo_list.append(clip_geo)
        for building in self.building_list:
            if building.geometry.intersects(self.tile.tile_boundary):
                clipped_geometry = building.geometry.intersection(self.tile.tile_boundary)
                if isinstance(clipped_geometry, LineString) or isinstance(clipped_geometry, Polygon):
                    self.geo_list.append(clipped_geometry)
                if isinstance(clipped_geometry, GeometryCollection) or isinstance(clipped_geometry, MultiPolygon) or isinstance(clipped_geometry, MultiLineString):
                    for clip_geo in clipped_geometry:
                        self.geo_list.append(clip_geo)
        for road_polygon in self.road_polygon_list:
            if road_polygon.geometry.intersects(self.tile.tile_boundary):
                clipped_geometry = road_polygon.geometry.intersection(self.tile.tile_boundary)
                if isinstance(clipped_geometry, LineString) or isinstance(clipped_geometry, Polygon):
                    self.geo_list.append(clipped_geometry)
                if isinstance(clipped_geometry, GeometryCollection) or isinstance(clipped_geometry, MultiPolygon) or isinstance(clipped_geometry, MultiLineString):
                    for clip_geo in clipped_geometry:
                        self.geo_list.append(clip_geo)
        print(GeometryCollection(self.geo_list))

    @staticmethod
    def read_map_json(path: str):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data["features"]
