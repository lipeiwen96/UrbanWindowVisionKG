"""香港 大学 aigc
MapSite类
输入：Xiaojuan Ma
- site_boundary: Polygon 一个外轮廓图形
- unbuildable_region_list: list 内部不可排布的区域
"""
from dataclasses import field, dataclass
from typing import List
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, Point, LineString, GeometryCollection, box, MultiPolygon
from map_system.coord_convertor.coords_convertor import CoordConvertor
import time
import json
import os
from map_system.tile_genenrator.map_structure import MapRoadCenterLine, MapBuilding, MapLot, MapRoadPolygon
from map_system.tile_genenrator.map_cropper import MapCropper
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial
from modules.input_module.site_builder import SiteIndex
from modules.classification_module.auto_classification import AutoClassification
from modules.classification_module.auto_classification_v2 import AutoClassificationV2
from modules.classification_module.auto_setback import AutoSetback
import platform


@dataclass
class HKTile:
    tile_id: int = field(default=int)
    tile_name: str = field(default=str)
    tile_boundary: Polygon = field(default_factory=Polygon)
    road_center_line_num: int = field(default=0)
    road_polygon_num: int = field(default=0)
    building_num: int = field(default=0)
    lot_num: int = field(default=0)

    @property
    def to_dict(self):
        return {
            "tile_id": self.tile_id,
            "tile_name": self.tile_name,
            "tile_boundary": self.tile_boundary.wkt,
            "road_center_line_num": self.road_center_line_num,
            "road_polygon_num": self.road_polygon_num,
            "building_num": self.building_num,
            "lot_num": self.lot_num,
        }

    def create_by_dict(self, input_dict: dict) -> 'HKTile':
        self.tile_id = input_dict["tile_id"]
        self.tile_name = input_dict["tile_name"]
        self.tile_boundary = Polygon(wkt.loads(input_dict["tile_boundary"]))
        self.road_center_line_num = input_dict["road_center_line_num"]
        self.road_polygon_num = input_dict["road_polygon_num"]
        self.building_num = input_dict["building_num"]
        self.lot_num = input_dict["lot_num"]
        return self


@dataclass
class MapSite:
    site_boundary: Polygon = field(default_factory=Polygon)
    detecting_site_boundary: Polygon = field(default_factory=Polygon)  # buffer, 用于裁切距离很近的建筑
    has_unbuildable_region: bool = field(default=False)
    unbuildable_region_list: List[Polygon] = field(default_factory=list)
    project_boundary: Polygon = field(default_factory=Polygon)

    # 读取的Tile信息
    intersects_tiles: List[HKTile] = field(default_factory=list)

    # 读入原始地图数据
    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    building_list: List[MapBuilding] = field(default_factory=list)
    lot_list: List[MapLot] = field(default_factory=list)
    road_center_line_id_list: List[int] = field(default_factory=list)
    road_polygon_id_list: List[int] = field(default_factory=list)
    building_id_list: List[int] = field(default_factory=list)
    lot_id_list: List[int] = field(default_factory=list)
    gla_id_list: List[int] = field(default_factory=list)

    # 裁切后的数据
    cropped_road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    cropped_road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    cropped_building_list: List[MapBuilding] = field(default_factory=list)
    cropped_lot_list: List[MapLot] = field(default_factory=list)

    # 场地是否有周边信息
    site_empty: bool = field(default=False)

    __base_height: float = field(default=20)
    __plot_height: float = field(default=0.65)
    __site_height: float = field(default=0.35)
    __inside_area_height: float = field(default=0.15)

    # 所有指标
    site_index: SiteIndex = field(default_factory=SiteIndex)
    # 前端显示用的模型
    model: DataModel = field(default_factory=DataModel)

    def init(self, site_boundary: Polygon, unbuildable_region_list: list = None, is_WGS: bool = True, site_name: str = "", site_id: str = "",
             version: str = "V1"):
        start_time = time.time()
        self.model = DataModel(name=site_name, id=site_id)
        self.model.user_data = {}
        self.site_index = SiteIndex()

        if len(unbuildable_region_list) != 0:
            self.has_unbuildable_region = True

        # STEP1-关联地图TILE数据读取
        # 坐标格式转化并存储
        self.__load_site_and_convert_coord_to_HK80(site_boundary, unbuildable_region_list, is_WGS=is_WGS)
        # 计算对应TILE区域，读取数据
        self.__read_map_library()
        step1_time = time.time()
        print(f"---【STEP1-关联地图TILE数据读取已完成!】--- 用时：{round(step1_time - start_time, 2)}s")

        # STEP2-裁切地图，生成项目周边信息
        self.__init_site_data_model()
        step2_time = time.time()
        print(f"---【STEP2-项目三维场地创建已完成!】--- 用时：{round(step2_time - step1_time, 2)}s")

        # STEP3-自动类别判断
        self.__auto_compute_site_classification(version)

        # STEP4-计算道路属性及退线
        self.__auto_compute_road_properties()

        self.site_index.site_mode = 1

        # TODO: 导出
        self.model.user_data["site_index"] = self.site_index.export_to_dict
        # print(self.model)

    def __load_site_and_convert_coord_to_HK80(self, site_boundary: Polygon, unbuildable_region_list: list = None, is_WGS: bool = True):
        if is_WGS:
            self.site_boundary = Polygon([Point(CoordConvertor.wgs_to_hk_reverse(coord[0], coord[1], wgs_type="3857"))
                                          for coord in site_boundary.exterior.coords])
        else:
            self.site_boundary = site_boundary

        # 面积检测
        if self.site_boundary.area > 30000000:
            raise Exception(f"The Area of Input Site Boundary is {round(self.site_boundary.area/1000000, 2)}k㎡！It's too Large to Generate 3D Environment.")

        self.detecting_site_boundary = self.site_boundary.buffer(5, cap_style=2, join_style=2)

        if unbuildable_region_list:
            if len(unbuildable_region_list) > 0:
                # 获取内部不可排布的区域
                self.has_unbuildable_region = True
                for unbuildable_region in unbuildable_region_list:
                    self.unbuildable_region_list.append(Polygon([Point(CoordConvertor.wgs_to_hk_reverse(coord[0], coord[1], wgs_type="3857"))
                                                        for coord in unbuildable_region.exterior.coords]))
        print(f"读入场地地块图形，并转化为HK80坐标, 地块面积 {round(self.site_boundary.area, 2)} ㎡")

        # 生成项目的边界范围
        minx, miny, maxx, maxy = self.site_boundary.bounds
        x_dis = 100
        y_dis = 100
        left_bottom_eas = minx - x_dis
        left_bottom_nor = miny - y_dis
        right_top_eas = maxx + x_dis
        right_top_nor = maxy + y_dis
        self.project_boundary = box(left_bottom_eas, left_bottom_nor, right_top_eas, right_top_nor)
        print(f"自动生成项目的边界范围，项目边界为{round(right_top_eas-left_bottom_eas, 2)}m * {round(right_top_nor-left_bottom_nor, 2)}m,"
              f"项目总面积 {round(self.project_boundary.area, 2)} ㎡")
        print(f"BOUNDS X: {left_bottom_eas} ~ {right_top_eas}, Y: {left_bottom_nor} ~ {right_top_nor}]")

    @staticmethod
    def get_all_tiles_info():
        print(f"当前系统为{platform.system()}")
        if platform.system() == "Windows":
            library_tiles_path = os.path.abspath(os.path.join("library", "HK_map", "TileBoundary", "HK_Tiles.json"))
        else:
            library_tiles_path = "/map_data/HK_map/TileBoundary/HK_Tiles.json"
            # library_tiles_path = os.path.join("map_data", "HK_map", "TileBoundary", "HK_Tiles.json")
        with open(library_tiles_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        tile_list = []
        for tile_dict in data["simplified_tiles"]:
            tile = HKTile().create_by_dict(tile_dict)

            # 这里加一个简单的筛选
            # if tile.road_center_line_num == 0 or tile.road_polygon_num == 0:
            #     print(f"{tile.tile_name}缺少地图数据，跳过载入操作")

            tile_list.append(tile)

        print(f"成功读入{len(tile_list)}个地图TILE数据")
        # print(GeometryCollection([tile.tile_boundary for tile in tile_list]))
        return tile_list

    @staticmethod
    def read_map_json(path: str):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data["features"]

    def __read_map_library(self):
        if platform.system() == "Windows":
            library_path = os.path.abspath(os.path.join("library", "HK_map", "TileBoundary"))
        else:
            library_path = "/map_data/HK_map/TileBoundary"
            # library_path = os.path.join("map_data", "HK_map", "TileBoundary")

        # 读取数据库中的TILE信息
        tile_list = MapSite.get_all_tiles_info()

        # 读入与项目边界相交的数据
        for tile in tile_list:
            if self.project_boundary.intersects(tile.tile_boundary):
                print(f"检测到: 目标地块与 {tile.tile_name}-TILE 相交")
                self.intersects_tiles.append(tile)

        # 以不重叠的方式，读入所有TILE数据
        if len(self.intersects_tiles) == 0:
            self.site_empty = True
            print("未找到相交TILE，以【默认状态：空白场地】来处理")
        else:
            for tile in self.intersects_tiles:
                tile_root = os.path.join(library_path, tile.tile_name)

                # 读取道路中心线
                road_center_line_map_path = os.path.join(tile_root, "road_center_line_map.json")
                for road_center_line_map in MapSite.read_map_json(road_center_line_map_path):
                    try:
                        road_center_line = MapRoadCenterLine()
                        road_center_line.init(road_center_line_map)
                        if road_center_line.object_id in self.road_center_line_id_list:
                            pass
                        else:
                            self.road_center_line_list.append(road_center_line)
                            self.road_center_line_id_list.append(road_center_line.object_id)
                    except Exception as e:
                        print(f'发生道路中心线读取错误:{e}, 跳过')

                # 读取道路图形
                road_polygon_path = os.path.join(tile_root, "road_polygon_map.json")
                for road_polygon_map in MapSite.read_map_json(road_polygon_path):
                    try:
                        road_polygon = MapRoadPolygon()
                        road_polygon.init(road_polygon_map)
                        if road_polygon.object_id in self.road_polygon_id_list:
                            pass
                        else:
                            self.road_polygon_list.append(road_polygon)
                            self.road_polygon_id_list.append(road_polygon.object_id)
                    except Exception as e:
                        print(f'发生道路图形读取错误:{e}, 跳过')

                # 读取建筑图形
                building_map_path = os.path.join(tile_root, "building_map.json")
                for building_map in MapSite.read_map_json(building_map_path):
                    try:
                        building = MapBuilding()
                        building.init(building_map)
                        if building.object_id in self.building_id_list:
                            pass
                        else:
                            self.building_list.append(building)
                            self.building_id_list.append(building.object_id)
                    except Exception as e:
                        print(f'发生建筑读取错误:{e}, 跳过')

                # 读取地块数据
                lot_map_path = os.path.join(tile_root, "lot_map.json")
                for lot_map in MapSite.read_map_json(lot_map_path):
                    try:
                        lot = MapLot()
                        lot.init(lot_map)
                        if lot.lot_id in self.lot_id_list:
                            pass
                        else:
                            self.lot_list.append(lot)
                            self.lot_id_list.append(lot.lot_id)
                    except Exception as e:
                        print(f'发生地块读取错误:{e}, 跳过')
                gla_map_path = os.path.join(tile_root, "gla_map.json")
                for gla_map in MapSite.read_map_json(gla_map_path):
                    try:
                        lot = MapLot()
                        lot.init(gla_map)
                        if lot.gla_id in self.gla_id_list:
                            pass
                        else:
                            self.lot_list.append(lot)
                            self.gla_id_list.append(lot.gla_id)
                    except Exception as e:
                        print(f'发生地块读取错误:{e}, 跳过')

                print(f"读取成功！{tile.tile_name} - TILE 数据已读入")

        print(f"【共读入{len(self.intersects_tiles)}个相交的TILE】")
        print(f"共读入{len(self.road_center_line_list)}个道路中心线的数据.....")
        print(f"共读入{len(self.road_polygon_list)}个道路图形的数据.....")
        print(f"共读入{len(self.building_list)}个建筑图形的数据.....")
        print(f"共读入{len(self.lot_list)}个地块图形的数据.....")

    def __init_site_data_model(self):
        # 生成项目范围
        element = DataElement(geometry=self.project_boundary,
                              layer="Xkool_ProjectBoundary",
                              material=DataElementMaterial(color="0xe5ebf1"),
                              start_height=-(self.__base_height + self.__plot_height + self.__site_height),
                              height=self.__base_height)
        self.model.insert_element(element)

        # 目标地块范围
        self.site_index.site_area = round(self.site_boundary.area, 2)
        element = DataElement(geometry=self.site_boundary,
                              layer="Xkool_TargetSiteBoundary",
                              material=DataElementMaterial(color="0xc7e29f"),
                              start_height=-self.__site_height,
                              height=self.__site_height)
        self.model.insert_element(element)

        # Xkool_SiteUnbuildableRegion
        for outline in self.unbuildable_region_list:
            element = DataElement(geometry=outline,
                                  layer="Xkool_SiteUnbuildableRegion",
                                  material=DataElementMaterial(color="0x000000"),
                                  start_height=-self.__site_height,
                                  height=self.__site_height)
            self.model.insert_element(element)

        # 周围建筑
        outcome = MapCropper.filter(self.building_list, site_boundary=self.site_boundary, tile_box=self.project_boundary)
        for out in outcome:
            element = DataElement(geometry=out.geometry,
                                  layer="Xkool_SurroundBuilding",
                                  material=DataElementMaterial(color="0xbbbec2"),
                                  start_height=out.start_height,
                                  height=out.height)
            self.model.insert_element(element)

        # 周围道路图形
        outcome = MapCropper.crop(self.road_polygon_list, site_boundary=self.site_boundary, tile_box=self.project_boundary)
        print(GeometryCollection([out.geometry for out in outcome]))
        for out in outcome:
            element = DataElement(geometry=out.geometry,
                                  layer="Road",
                                  # material=DataElementMaterial(color="0xe5ebf1"),
                                  material=DataElementMaterial(color="0xe5ebf1", opacity=1, outline_type="drawing_road"),
                                  start_height=-(self.__plot_height),
                                  )
            self.model.insert_element(element)
            # print(element)

        # 周围地块
        outcome = MapCropper.crop(self.lot_list, site_boundary=self.site_boundary, tile_box=self.project_boundary)
        for out in outcome:
            element = DataElement(geometry=out.geometry,
                                  layer="Xkool_PlotBoundary",
                                  material=DataElementMaterial(color="0xced4d6"),
                                  start_height=-(self.__plot_height + self.__site_height),
                                  height=self.__plot_height)
            self.model.insert_element(element)
        self.model.renew()

    def __auto_compute_site_classification(self, version: str):
        """
        自动计算场地类型
        """
        if version == "V1":
            auto_cl = AutoClassification()
        else:
            auto_cl = AutoClassificationV2()

        # 获得计算结果
        try:
            site_classification, site_classification_info = auto_cl.init(site_boundary=self.site_boundary,
                                                                         project_boundary=self.project_boundary,
                                                                         road_center_line_list=self.road_center_line_list,
                                                                         road_polygon_list=self.road_polygon_list)
            self.model.user_data["site_info"] = auto_cl.segs_dict

        except Exception as e:
            print(f"Error: [{e}] occurs when doing automatically classified, please calculate manually")
            site_classification = "A"
            site_classification_info = f"Error occurs when doing automatically classified, please calculate manually "

        self.site_index.site_classification = site_classification
        self.site_index.site_classification_info = site_classification_info

    def __auto_compute_road_properties(self):
        # 先找到周围所有的道路中心线
        auto_setback = AutoSetback()
        auto_setback.init(site_boundary=self.site_boundary,
                          road_center_line_list=self.road_center_line_list,
                          road_polygon_list=self.road_polygon_list)

        # 画一些需要开关的道路中心线和间距框
        self.model.user_data["road_info"] = auto_setback.road_info
        for road in auto_setback.setback_road_list:
            if road.width > 15:
                color = "0x000000"
            elif road.width >= 4.5:
                color = "0xff0000"
            else:
                color = "0x1800ff"
            element = DataElement(geometry=road.geometry,
                                  layer="Switch_Road_Center_Line",
                                  material=DataElementMaterial(color=color, opacity=1),
                                  start_height=-(self.__plot_height) + 2)
            self.model.insert_element(element)

            element = DataElement(geometry=road.setback_area,
                                  layer="Switch_Road_Setback_Area",
                                  material=DataElementMaterial(color="0xff9c00", opacity=0.5),
                                  start_height=-(self.__plot_height) + 2)
            self.model.insert_element(element)

        self.model.renew()












