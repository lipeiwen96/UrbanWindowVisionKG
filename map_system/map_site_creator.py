# -*- coding: utf-8 -*-
# map_site_creator.py
"""
MapSite类
作者：Peiwen Li
功能：
  - 定义 HKTile 类，用于表示香港地图瓦片信息。
  - 定义 MapSite 类，用于处理和表示一个特定的项目场地。
  - 输入：地块边界 (Polygon)，内部不可排布区域列表 (可选)。
  - 处理：
    - 坐标转换 (WGS84 -> HK80)。
    - 读取与项目边界相交的地图瓦片数据 (道路、建筑、地块等)。
    - 裁切地图元素至项目范围。
    - 创建三维可视化模型 (DataModel)。
    - 自动计算场地分类。
    - 自动计算道路属性和退线。
  - 输出：包含场地信息、周边环境、分析结果的 DataModel 对象和 SiteIndex 对象。
"""
from dataclasses import field, dataclass
from typing import List
from shapely import wkt  # 用于处理WKT(Well-Known Text)格式的地理空间数据
from shapely.geometry import Polygon, Point, box, MultiPolygon # Shapely 几何对象
from map_system.utils.coords_convertor import CoordConvertor # 自定义坐标转换模块
import time # 时间模块，用于计时
import json # JSON 数据处理
import os # 操作系统交互，用于路径处理
from map_system.map_structure import MapRoadCenterLine, MapBuilding, MapLot, MapRoadPolygon # 自定义地图数据结构
from map_system.tile_genenrator.map_cropper import MapCropper # 自定义地图裁切模块
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement # 自定义数据模型和元素结构 (可能用于可视化)
from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial # 自定义数据元素的材质信息
from modules.input_module.site_builder import SiteIndex # 自定义场地指标计算模块
from modules.classification_module.auto_classification import AutoClassification # 自动场地分类模块 V1
from modules.classification_module.auto_classification_v2 import AutoClassificationV2 # 自动场地分类模块 V2
from modules.classification_module.auto_setback import AutoSetback # 自动计算退线模块
import platform # 用于检测操作系统平台 (Windows/Linux)


@dataclass # 使用 dataclass 装饰器简化数据类的创建
class HKTile:
    """表示香港地图瓦片信息的类"""
    tile_id: int = field(default=int) # 瓦片ID
    tile_name: str = field(default=str) # 瓦片名称 (例如 "HP02SE")
    tile_boundary: Polygon = field(default_factory=Polygon) # 瓦片边界 (Shapely Polygon对象)
    road_center_line_num: int = field(default=0) # 瓦片内道路中心线数量
    road_polygon_num: int = field(default=0) # 瓦片内道路多边形数量
    building_num: int = field(default=0) # 瓦片内建筑数量
    lot_num: int = field(default=0) # 瓦片内地块数量

    @property # 将方法转换为只读属性
    def to_dict(self):
        """将 HKTile 对象转换为字典，方便序列化 (例如存入JSON)"""
        return {
            "tile_id": self.tile_id,
            "tile_name": self.tile_name,
            "tile_boundary": self.tile_boundary.wkt, # 将 Polygon 对象转换为 WKT 字符串
            "road_center_line_num": self.road_center_line_num,
            "road_polygon_num": self.road_polygon_num,
            "building_num": self.building_num,
            "lot_num": self.lot_num,
        }

    def create_by_dict(self, input_dict: dict) -> 'HKTile':
        """从字典创建或更新 HKTile 对象"""
        self.tile_id = input_dict["tile_id"]
        self.tile_name = input_dict["tile_name"]
        # 从 WKT 字符串加载 Polygon 对象
        self.tile_boundary = Polygon(wkt.loads(input_dict["tile_boundary"]))
        self.road_center_line_num = input_dict["road_center_line_num"]
        self.road_polygon_num = input_dict["road_polygon_num"]
        self.building_num = input_dict["building_num"]
        self.lot_num = input_dict["lot_num"]
        return self


@dataclass
class MapSite:
    """
    表示一个项目场地及其周边环境的类。
    负责加载、处理、分析地图数据，并生成可视化模型。
    """
    # --- 输入参数 ---
    site_boundary: Polygon = field(default_factory=Polygon) # 项目地块边界 (HK80坐标系)
    detecting_site_boundary: Polygon = field(default_factory=Polygon) # 缓冲后的地块边界，用于检测邻近元素
    has_unbuildable_region: bool = field(default=False) # 是否有内部不可排布区域
    unbuildable_region_list: List[Polygon] = field(default_factory=list) # 内部不可排布区域列表 (HK80坐标系)
    project_boundary: Polygon = field(default_factory=Polygon) # 项目影响范围边界 (比site_boundary稍大，用于裁切周边环境)

    # --- 读取的Tile信息 ---
    intersects_tiles: List[HKTile] = field(default_factory=list) # 与项目范围相交的瓦片列表

    # --- 读入的原始地图数据 (来自相交的Tiles) ---
    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list) # 道路中心线列表
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list) # 道路多边形列表
    building_list: List[MapBuilding] = field(default_factory=list) # 建筑列表
    lot_list: List[MapLot] = field(default_factory=list) # 地块列表 (包含LOT和GLA)
    # 用于去重的ID列表
    road_center_line_id_list: List[int] = field(default_factory=list)
    road_polygon_id_list: List[int] = field(default_factory=list)
    building_id_list: List[int] = field(default_factory=list)
    lot_id_list: List[int] = field(default_factory=list) # LOT 地块ID
    gla_id_list: List[int] = field(default_factory=list) # GLA 地块ID

    # --- 裁切后的数据 (在 project_boundary 范围内) ---
    # 注意：这些列表在当前代码中并未被填充和使用，主要通过 MapCropper 直接处理并生成 DataElement
    cropped_road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    cropped_road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    cropped_building_list: List[MapBuilding] = field(default_factory=list)
    cropped_lot_list: List[MapLot] = field(default_factory=list)

    # --- 状态与配置 ---
    site_empty: bool = field(default=False) # 场地是否没有找到任何相交的地图瓦片数据
    # 默认高度值 (用于 3D 可视化)
    __base_height: float = field(default=20) # 基础平台高度
    __plot_height: float = field(default=0.65) # 地块抬升高度
    __site_height: float = field(default=0.35) # 目标地块抬升高度
    __inside_area_height: float = field(default=0.15) # 内部区域抬升高度 (未使用)

    # --- 输出结果 ---
    site_index: SiteIndex = field(default_factory=SiteIndex) # 存储场地的各种计算指标 (面积、分类等)
    model: DataModel = field(default_factory=DataModel) # 用于前端显示的3D数据模型，包含多个DataElement

    def init(self, site_boundary: Polygon, unbuildable_region_list: list = None, is_WGS: bool = True, site_name: str = "", site_id: str = "",
             version: str = "V1"):
        """
        初始化 MapSite 对象的主方法。

        Args:
            site_boundary (Polygon): 输入的地块边界 (默认为 WGS 坐标)。
            unbuildable_region_list (list, optional): 不可排布区域列表 (默认为 WGS 坐标)。默认为 None。
            is_WGS (bool, optional): 输入坐标是否为 WGS84 (通常是 EPSG:3857)。默认为 True。如果为 False，则假定为 HK80。
            site_name (str, optional): 场地名称。默认为 ""。
            site_id (str, optional): 场地唯一标识符。默认为 ""。
            version (str, optional): 使用的自动分类算法版本 ("V1" 或 "V2")。默认为 "V1"。
        """
        start_time = time.time() # 记录开始时间
        self.model = DataModel(name=site_name, id=site_id) # 初始化数据模型
        self.model.user_data = {} # 初始化用户数据字典，用于存储额外信息
        self.site_index = SiteIndex() # 初始化场地指标对象

        if unbuildable_region_list is None:
            unbuildable_region_list = [] # 确保列表存在

        if len(unbuildable_region_list) != 0:
            self.has_unbuildable_region = True

        # --- STEP1: 关联地图 TILE 数据读取 ---
        # 1.1 加载场地边界和不可排布区域，并将坐标转换为 HK80
        self.__load_site_and_convert_coord_to_HK80(site_boundary, unbuildable_region_list, is_WGS=is_WGS)
        # 1.2 计算与项目边界相交的 TILE 区域，并读取这些 TILE 的地图数据
        self.__read_map_library()
        step1_time = time.time()
        print(f"---【STEP1-关联地图TILE数据读取已完成!】--- 用时：{round(step1_time - start_time, 2)}s")

        # --- STEP2: 裁切地图，生成项目周边三维模型 ---
        self.__init_site_data_model()
        step2_time = time.time()
        print(f"---【STEP2-项目三维场地创建已完成!】--- 用时：{round(step2_time - step1_time, 2)}s")

        # --- STEP3: 自动计算场地类别 ---
        self.__auto_compute_site_classification(version)
        step3_time = time.time()
        print(f"---【STEP3-自动场地类别判断已完成!】--- 用时：{round(step3_time - step2_time, 2)}s")

        # --- STEP4: 计算道路属性及退线 ---
        self.__auto_compute_road_properties()
        step4_time = time.time()
        print(f"---【STEP4-自动计算道路属性及退线已完成!】--- 用时：{round(step4_time - step3_time, 2)}s")

        self.site_index.site_mode = 1 # 标记场地模式 (可能表示已成功加载地图数据)

        # --- 导出结果 ---
        # 将计算出的场地指标存入 DataModel 的 user_data 中
        self.model.user_data["site_index"] = self.site_index.export_to_dict
        # print(self.model) # (调试用) 打印最终模型
        end_time = time.time()
        print(f"---【MapSite 初始化总用时：{round(end_time - start_time, 2)}s】---")

    def __load_site_and_convert_coord_to_HK80(self, site_boundary: Polygon, unbuildable_region_list: list = None, is_WGS: bool = True):
        """加载输入的场地边界和不可排布区域，并将其坐标从 WGS84 (EPSG:3857) 转换为 HK80。"""
        if is_WGS:
            # 转换地块边界坐标
            self.site_boundary = Polygon([Point(CoordConvertor.wgs_to_hk_reverse(coord[0], coord[1], wgs_type="3857"))
                                          for coord in site_boundary.exterior.coords])
        else:
            # 如果输入已经是 HK80，则直接使用
            self.site_boundary = site_boundary

        # 面积检测，防止处理过大的地块
        if self.site_boundary.area > 1000000000: # 30 平方公里
            raise Exception(f"输入的场地边界面积为 {round(self.site_boundary.area/1000000, 2)}k㎡！太大，无法生成三维环境。")

        # 创建一个缓冲区域，用于检测靠近边界的元素
        self.detecting_site_boundary = self.site_boundary.buffer(5, cap_style=2, join_style=2) # 缓冲5米

        # 处理不可排布区域
        if unbuildable_region_list:
            self.unbuildable_region_list = [] # 清空可能存在的默认值
            if len(unbuildable_region_list) > 0:
                self.has_unbuildable_region = True
                for unbuildable_region in unbuildable_region_list:
                    if is_WGS:
                        # 转换不可排布区域坐标
                         converted_region = Polygon([Point(CoordConvertor.wgs_to_hk_reverse(coord[0], coord[1], wgs_type="3857"))
                                                    for coord in unbuildable_region.exterior.coords])
                         self.unbuildable_region_list.append(converted_region)
                    else:
                         self.unbuildable_region_list.append(unbuildable_region) # 直接添加

        print(f"读入场地地块图形，并转化为HK80坐标, 地块面积 {round(self.site_boundary.area, 2)} ㎡")

        # 生成项目的整体边界范围 (project_boundary)，比地块边界大 100 米
        minx, miny, maxx, maxy = self.site_boundary.bounds
        x_dis = 100 # X 方向扩展距离
        y_dis = 100 # Y 方向扩展距离
        left_bottom_eas = minx - x_dis
        left_bottom_nor = miny - y_dis
        right_top_eas = maxx + x_dis
        right_top_nor = maxy + y_dis
        # 创建一个矩形作为项目边界
        self.project_boundary = box(left_bottom_eas, left_bottom_nor, right_top_eas, right_top_nor)
        print(f"自动生成项目的边界范围，项目边界为{round(right_top_eas-left_bottom_eas, 2)}m * {round(right_top_nor-left_bottom_nor, 2)}m,"
              f"项目总面积 {round(self.project_boundary.area, 2)} ㎡")
        print(f"项目边界坐标范围 BOUNDS X: {left_bottom_eas} ~ {right_top_eas}, Y: {left_bottom_nor} ~ {right_top_nor}]")

    @staticmethod
    def get_all_tiles_info():
        """静态方法：读取所有预处理好的 HK Tile 的信息 (边界、元素数量等)。"""
        print(f"当前系统为{platform.system()}")
        # 根据操作系统确定 Tile 索引文件的路径
        if platform.system() == "Windows":
            # 本地开发环境路径
            library_tiles_path = os.path.abspath(os.path.join("library", "HK_map", "TileBoundary", "HK_Tiles.json"))
        else:
            # 服务器/容器环境路径
            library_tiles_path = "/map_data/HK_map/TileBoundary/HK_Tiles.json"
            # library_tiles_path = os.path.join("map_data", "HK_map", "TileBoundary", "HK_Tiles.json") # 备用相对路径

        # 读取 JSON 文件
        with open(library_tiles_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        tile_list = []
        # 遍历 JSON 中的每个瓦片信息
        for tile_dict in data["simplified_tiles"]:
            # 使用 HKTile.create_by_dict 方法创建 HKTile 对象
            tile = HKTile().create_by_dict(tile_dict)

            # 可以加一些简单的筛选逻辑，例如跳过数据不完整的瓦片
            # if tile.road_center_line_num == 0 or tile.road_polygon_num == 0:
            #     print(f"{tile.tile_name}缺少地图数据，跳过载入操作")
            #     continue # 跳过这个瓦片

            tile_list.append(tile)

        print(f"成功读入{len(tile_list)}个地图TILE数据")
        # (调试用) 打印所有瓦片的边界集合
        # print(GeometryCollection([tile.tile_boundary for tile in tile_list]))
        return tile_list

    @staticmethod
    def read_map_json(path: str):
        """静态方法：读取指定路径的 JSON 地图数据文件，并返回 'features' 列表。"""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            # 假设 JSON 结构符合 GeoJSON FeatureCollection 格式
            return data.get("features", []) # 使用 .get 防止 "features" 不存在时报错
        except FileNotFoundError:
            print(f"警告: 文件未找到 {path}")
            return []
        except json.JSONDecodeError:
            print(f"警告: JSON 解析错误 {path}")
            return []
        except Exception as e:
            print(f"警告: 读取 JSON 时发生未知错误 {path}: {e}")
            return []

    def __read_map_library(self):
        """读取与项目边界 (project_boundary) 相交的 Tile 中的详细地图数据。"""
        # 根据操作系统确定地图库的根路径
        if platform.system() == "Windows":
            library_path = os.path.abspath(os.path.join("library", "HK_map", "TileBoundary"))
        else:
            library_path = "/map_data/HK_map/TileBoundary"
            # library_path = os.path.join("map_data", "HK_map", "TileBoundary")

        # 获取所有 Tile 的基本信息
        tile_list = MapSite.get_all_tiles_info()

        # 查找与项目边界 (project_boundary) 相交的 Tile
        self.intersects_tiles = []
        for tile in tile_list:
            # 使用 intersects 方法检查几何对象是否相交
            if self.project_boundary.intersects(tile.tile_boundary):
                print(f"检测到: 目标地块与 {tile.tile_name}-TILE 相交")
                self.intersects_tiles.append(tile)

        # 如果没有找到相交的 Tile
        if len(self.intersects_tiles) == 0:
            self.site_empty = True # 标记场地为空
            print("未找到相交TILE，以【默认状态：空白场地】来处理")
            # 清空可能存在的旧数据
            self.road_center_line_list = []
            self.road_polygon_list = []
            self.building_list = []
            self.lot_list = []
            self.road_center_line_id_list = []
            self.road_polygon_id_list = []
            self.building_id_list = []
            self.lot_id_list = []
            self.gla_id_list = []
        else:
            # 清空列表，准备加载新数据
            self.road_center_line_list = []
            self.road_polygon_list = []
            self.building_list = []
            self.lot_list = []
            self.road_center_line_id_list = []
            self.road_polygon_id_list = []
            self.building_id_list = []
            self.lot_id_list = []
            self.gla_id_list = []
            # 遍历所有相交的 Tile
            for tile in self.intersects_tiles:
                # 构建该 Tile 数据所在的目录路径
                tile_root = os.path.join(library_path, tile.tile_name)
                print(f"开始读取 TILE: {tile.tile_name} 的数据...")

                # --- 读取道路中心线 ---
                road_center_line_map_path = os.path.join(tile_root, "road_center_line_map.json")
                for road_center_line_map in MapSite.read_map_json(road_center_line_map_path):
                    try:
                        road_center_line = MapRoadCenterLine()
                        road_center_line.init(road_center_line_map) # 从字典初始化对象
                        # 检查 ID 是否已存在，避免重复加载
                        if road_center_line.object_id not in self.road_center_line_id_list:
                            self.road_center_line_list.append(road_center_line)
                            self.road_center_line_id_list.append(road_center_line.object_id)
                    except Exception as e:
                        # 捕获并打印错误，然后跳过该条数据
                        print(f'发生道路中心线读取错误:{e}, 跳过')

                # --- 读取道路图形 ---
                road_polygon_path = os.path.join(tile_root, "road_polygon_map.json")
                for road_polygon_map in MapSite.read_map_json(road_polygon_path):
                    try:
                        road_polygon = MapRoadPolygon()
                        road_polygon.init(road_polygon_map)
                        if road_polygon.object_id not in self.road_polygon_id_list:
                            self.road_polygon_list.append(road_polygon)
                            self.road_polygon_id_list.append(road_polygon.object_id)
                    except Exception as e:
                        print(f'发生道路图形读取错误:{e}, 跳过')

                # --- 读取建筑图形 ---
                building_map_path = os.path.join(tile_root, "building_map.json")
                for building_map in MapSite.read_map_json(building_map_path):
                    try:
                        building = MapBuilding()
                        building.init(building_map)
                        if building.object_id not in self.building_id_list:
                            self.building_list.append(building)
                            self.building_id_list.append(building.object_id)
                    except Exception as e:
                        print(f'发生建筑读取错误:{e}, 跳过')

                # --- 读取地块数据 (LOT) ---
                lot_map_path = os.path.join(tile_root, "lot_map.json")
                for lot_map in MapSite.read_map_json(lot_map_path):
                    try:
                        lot = MapLot()
                        lot.init(lot_map)
                        if lot.lot_id not in self.lot_id_list: # 使用 lot_id 判断重复
                            self.lot_list.append(lot)
                            self.lot_id_list.append(lot.lot_id)
                    except Exception as e:
                        print(f'发生地块(LOT)读取错误:{e}, 跳过')

                # --- 读取地块数据 (GLA) ---
                gla_map_path = os.path.join(tile_root, "gla_map.json")
                for gla_map in MapSite.read_map_json(gla_map_path):
                    try:
                        lot = MapLot()
                        lot.init(gla_map) # 使用同一个 MapLot 类处理 GLA
                        if lot.gla_id not in self.gla_id_list: # 使用 gla_id 判断重复
                             # 检查是否在 lot_id_list 中也存在 (防止数据源问题导致重复)
                            is_duplicate_lot = hasattr(lot, 'lot_id') and lot.lot_id in self.lot_id_list
                            if not is_duplicate_lot:
                                self.lot_list.append(lot)
                                self.gla_id_list.append(lot.gla_id)
                            # else:
                            #     print(f"警告: GLA 数据 {lot.gla_id} 可能与 LOT 数据重复，已跳过。")
                    except Exception as e:
                        print(f'发生地块(GLA)读取错误:{e}, 跳过')

                print(f"读取成功！{tile.tile_name} - TILE 数据已读入")

        print(f"【共读入{len(self.intersects_tiles)}个相交的TILE】")
        print(f"共读入{len(self.road_center_line_list)}个道路中心线的数据.....")
        print(f"共读入{len(self.road_polygon_list)}个道路图形的数据.....")
        print(f"共读入{len(self.building_list)}个建筑图形的数据.....")
        print(f"共读入{len(self.lot_list)}个地块图形的数据 (含LOT和GLA).....")

    def __init_site_data_model(self):
        """根据加载和裁切的地图数据，创建用于前端显示的 DataModel 对象。"""

        # 1. 创建项目范围的基底 (灰色平台)
        element = DataElement(geometry=self.project_boundary,
                              layer="Xkool_ProjectBoundary", # 图层名称
                              material=DataElementMaterial(color="0xe5ebf1"), # 材质颜色
                              start_height=-(self.__base_height + self.__plot_height + self.__site_height), # 起始高度 (在最下方)
                              height=self.__base_height) # 物体高度
        self.model.insert_element(element) # 添加到模型

        # 2. 创建目标地块范围 (绿色抬升区域)
        self.site_index.site_area = round(self.site_boundary.area, 2) # 记录地块面积到指标对象
        element = DataElement(geometry=self.site_boundary,
                              layer="Xkool_TargetSiteBoundary",
                              material=DataElementMaterial(color="0xc7e29f"), # 绿色
                              start_height=-self.__site_height, # 起始高度 (比基底高)
                              height=self.__site_height)
        self.model.insert_element(element)

        # 3. 创建内部不可排布区域 (黑色标记)
        for outline in self.unbuildable_region_list:
            element = DataElement(geometry=outline,
                                  layer="Xkool_SiteUnbuildableRegion",
                                  material=DataElementMaterial(color="0x000000"), # 黑色
                                  start_height=-self.__site_height, # 与目标地块同高
                                  height=self.__site_height)
            self.model.insert_element(element)

        # 4. 添加周围建筑
        # 使用 MapCropper.filter 过滤和裁切建筑数据
        # filter 通常保留与 site_boundary 相交或在 project_boundary 内的完整建筑
        outcome_buildings = MapCropper.filter(self.building_list, site_boundary=self.detecting_site_boundary, tile_box=self.project_boundary) # 使用缓冲边界检测
        print(f"过滤/裁切后保留 {len(outcome_buildings)} 个周边建筑")
        for out in outcome_buildings:
            element = DataElement(geometry=out.geometry, # 建筑的几何形状
                                  layer="Xkool_SurroundBuilding", # 周边建筑图层
                                  material=DataElementMaterial(color="0xbbbec2"), # 灰色
                                  start_height=out.start_height, # 建筑的起始高度 (可能来自原始数据)
                                  height=out.height) # 建筑的高度 (可能来自原始数据)
            self.model.insert_element(element)

        # 5. 添加周围道路图形 (路面)
        # 使用 MapCropper.crop 裁切道路数据
        # crop 会将与 project_boundary 相交的道路图形进行精确裁切
        outcome_roads = MapCropper.crop(self.road_polygon_list, site_boundary=self.site_boundary, tile_box=self.project_boundary)
        print(f"过滤/裁切后保留 {len(outcome_roads)} 个周边道路图形")
        # (调试用) 打印裁切后的道路几何集合
        # print(GeometryCollection([out.geometry for out in outcome_roads]))
        for out in outcome_roads:
            element = DataElement(geometry=out.geometry, # 裁切后的道路几何形状
                                  layer="Road", # 道路图层
                                  # material=DataElementMaterial(color="0xe5ebf1"), # 浅灰色路面
                                  material=DataElementMaterial(color="0xe5ebf1", opacity=1, outline_type="drawing_road"), # 使用带轮廓线的材质
                                  start_height=-(self.__plot_height), # 道路高度 (比基底高，比地块低)
                                  # height 默认为 0 或一个很小的值，表示平面
                                  )
            self.model.insert_element(element)
            # print(element) # (调试用)

        # 6. 添加周围地块
        outcome_lots = MapCropper.crop(self.lot_list, site_boundary=self.site_boundary, tile_box=self.project_boundary)
        print(f"过滤/裁切后保留 {len(outcome_lots)} 个周边地块")
        for out in outcome_lots:
            # 检查几何类型，确保是 Polygon 或 MultiPolygon
            if isinstance(out.geometry, (Polygon, MultiPolygon)):
                element = DataElement(geometry=out.geometry, # 裁切后的地块几何形状
                                      layer="Xkool_PlotBoundary", # 周边地块图层
                                      material=DataElementMaterial(color="0xced4d6"), # 另一种灰色
                                      start_height=-(self.__plot_height + self.__site_height), # 地块高度 (位于基底和目标地块之间)
                                      height=self.__plot_height) # 地块自身厚度
                self.model.insert_element(element)
            else:
                print(f"警告: 裁切后的地块几何类型不是 Polygon/MultiPolygon: {type(out.geometry)}")


        self.model.renew() # 更新模型的内部状态 (例如边界框)

    def __auto_compute_site_classification(self, version: str):
        """自动计算场地的类型 (例如 A, B, C 类)。"""
        print(f"开始自动计算场地分类 (版本: {version})...")
        # 根据传入的版本号选择分类器
        if version == "V1":
            auto_cl = AutoClassification()
        else: # 默认为 V2 或更新版本
            auto_cl = AutoClassificationV2()

        # 准备输入数据
        # 注意：这里传入的是原始的 road_center_line_list 和 road_polygon_list
        # 分类算法内部可能会根据 site_boundary 和 project_boundary 进行过滤
        try:
            site_classification, site_classification_info, segs_dict = auto_cl.init(
                site_boundary=self.site_boundary,
                project_boundary=self.project_boundary,
                road_center_line_list=self.road_center_line_list,
                road_polygon_list=self.road_polygon_list
            )
            # 将分类的详细信息 (例如各边的分类) 存储到 user_data
            self.model.user_data["site_info"] = segs_dict
            print(f"场地分类计算完成: {site_classification} - {site_classification_info}")

        except Exception as e:
            # 如果自动分类出错，则设置默认值并记录错误信息
            print(f"错误: 自动分类时发生错误 [{e}]，请手动计算")
            site_classification = "A" # 默认分类
            site_classification_info = f"自动分类时发生错误，请手动计算"
            self.model.user_data["site_info"] = {"error": str(e)} # 记录错误

        # 将最终的分类结果存储到 site_index 对象中
        self.site_index.site_classification = site_classification
        self.site_index.site_classification_info = site_classification_info

    def __auto_compute_road_properties(self):
        """自动计算场地周边道路的属性（如宽度）和所需的退线距离。"""
        print("开始自动计算道路属性和退线...")
        try:
            auto_setback = AutoSetback()
            # 初始化退线计算器，传入地块边界和原始道路数据
            auto_setback.init(site_boundary=self.site_boundary,
                              road_center_line_list=self.road_center_line_list,
                              road_polygon_list=self.road_polygon_list)

            # 将计算得到的道路详细信息 (例如每条相关道路的宽度、类型、退线距离) 存储到 user_data
            self.model.user_data["road_info"] = auto_setback.road_info
            print(f"共计算 {len(auto_setback.setback_road_list)} 条相关道路的退线信息")

            # --- 可视化退线结果 (可选，用于前端开关显示) ---
            added_elements = 0
            for road in auto_setback.setback_road_list:
                # 1. 可视化相关的道路中心线，根据宽度用不同颜色表示
                if road.width > 15: # 主干道?
                    color = "0x000000" # 黑色
                elif road.width >= 4.5: # 次干道/支路?
                    color = "0xff0000" # 红色
                else: # 小路?
                    color = "0x1800ff" # 蓝色

                # 创建道路中心线 DataElement
                element_line = DataElement(geometry=road.geometry, # 道路中心线的几何 (LineString)
                                          layer="Switch_Road_Center_Line", # 可切换显示的图层
                                          material=DataElementMaterial(color=color, opacity=1),
                                          start_height=-(self.__plot_height) + 2) # 略高于路面，防止 Z-fighting
                self.model.insert_element(element_line)
                added_elements += 1

                # 2. 可视化计算出的退线区域 (通常是道路中心线两侧的缓冲区域)
                if road.setback_area and not road.setback_area.is_empty:
                    element_area = DataElement(geometry=road.setback_area, # 退线区域的几何 (Polygon/MultiPolygon)
                                            layer="Switch_Road_Setback_Area", # 可切换显示的图层
                                            material=DataElementMaterial(color="0xff9c00", opacity=0.5), # 半透明橙色
                                            start_height=-(self.__plot_height) + 2) # 与中心线同高
                    self.model.insert_element(element_area)
                    added_elements += 1
            print(f"已添加 {added_elements} 个用于显示道路中心线和退线区域的可视化元素")
            self.model.renew() # 更新模型状态

        except Exception as e:
            print(f"错误: 计算道路属性和退线时发生错误 [{e}]")
            self.model.user_data["road_info"] = {"error": str(e)} # 记录错误