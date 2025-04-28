# -*- coding: utf-8 -*-
# tile_system_generator.py
"""
地图处理器 V2.0
作者：Peiwen Li
功能：
 - 读取原始、未分割的大范围地图数据 (JSON 格式)。
 - 根据输入的瓦片边界 (Tile Boundary)，裁切原始地图数据。
 - 将每个瓦片对应的裁切后数据分别导出为 JSON 文件。
 - 目的是将大的原始地图数据预处理成分块的小文件，方便 MapSite 类按需加载。
"""
from dataclasses import field, dataclass
from typing import List
import json
import os
from map_system.map_structure import MapRoadCenterLine, MapBuilding, MapLot, MapRoadPolygon # 自定义地图数据结构
from shapely.geometry import Polygon, GeometryCollection  # Shapely 几何对象
from art_public_modules.art_utils.common_utils.path_utils import PathUtils # 自定义路径处理工具类


@dataclass
class TileSystemGenerator:
    """
    地图瓦片系统生成器。
    负责读取原始地图数据，并根据指定的瓦片边界进行裁切和导出。
    """
    # --- 输入文件路径 ---
    road_center_line_map_path: str = field(default=str)  # 道路中心线原始数据路径
    building_map_path: str = field(default=str)  # 建筑原始数据路径
    lot_map_path: str = field(default=str)  # LOT 地块原始数据路径
    gla_map_path: str = field(default=str)  # GLA 地块原始数据路径
    road_polygon_path: str = field(default=str)  # 道路多边形原始数据路径

    # --- 读入的全部原始地图数据 (内存中持有整个地图的数据) ---
    road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    building_list: List[MapBuilding] = field(default_factory=list)
    lot_list: List[MapLot] = field(default_factory=list) # 包含 LOT 和 GLA

    # --- 当前处理的瓦片信息和裁切后的数据 ---
    tile_boundary: Polygon = field(default_factory=Polygon) # 当前正在处理的瓦片边界
    # 裁切后，仅包含与当前 tile_boundary 相交的数据
    cropped_road_center_line_list: List[MapRoadCenterLine] = field(default_factory=list)
    cropped_road_polygon_list: List[MapRoadPolygon] = field(default_factory=list)
    cropped_building_list: List[MapBuilding] = field(default_factory=list)
    cropped_lot_list: List[MapLot] = field(default_factory=list) # 包含 LOT 和 GLA
    # 裁切后数据的 ID 列表 (主要用于统计和可能的去重)
    cropped_road_center_line_id_list: List[int] = field(default_factory=list)
    cropped_road_polygon_id_list: List[int] = field(default_factory=list)
    cropped_building_id_list: List[int] = field(default_factory=list)
    cropped_lot_id_list: List[int] = field(default_factory=list) # LOT ID
    cropped_gla_id_list: List[int] = field(default_factory=list) # GLA ID

    def read_row_data(self, road_center_line_map_path: str, building_map_path: str, lot_map_path: str, gla_map_path: str, road_polygon_path: str):
        """
        读取所有原始地图数据文件到内存中。
        这个方法会加载所有指定路径下的 JSON 文件，并将所有 feature 转换为对应的 Map* 对象。
        注意：这可能会消耗大量内存，取决于原始数据的大小。
        """
        print("开始读取原始地图数据...")
        # 保存文件路径
        self.road_center_line_map_path = road_center_line_map_path
        self.building_map_path = building_map_path
        self.lot_map_path = lot_map_path
        self.gla_map_path = gla_map_path
        self.road_polygon_path = road_polygon_path

        # 清空可能存在的旧数据
        self.road_center_line_list = []
        self.road_polygon_list = []
        self.building_list = []
        self.lot_list = []

        # --- 读取道路中心线 ---
        features = TileSystemGenerator.read_map_json(road_center_line_map_path)
        print(f"读取道路中心线文件: {len(features)} 个 features")
        for road_center_line_map in features:
            try:
                road_center_line = MapRoadCenterLine()
                road_center_line.init(road_center_line_map)
                self.road_center_line_list.append(road_center_line)
            except Exception as e:
                print(f'发生道路中心线读取错误:{e}, 跳过')
        print(f"共读入{len(self.road_center_line_list)}个道路中心线的数据.....")

        # --- 读取道路图形 ---
        features = TileSystemGenerator.read_map_json(road_polygon_path)
        print(f"读取道路图形文件: {len(features)} 个 features")
        for road_polygon_map in features:
            try:
                road_polygon = MapRoadPolygon()
                road_polygon.init(road_polygon_map)
                self.road_polygon_list.append(road_polygon)
            except Exception as e:
                print(f'发生道路图形读取错误:{e}, 跳过')
        print(f"共读入{len(self.road_polygon_list)}个道路图形的数据.....")

        # --- 读取建筑图形 ---
        features = TileSystemGenerator.read_map_json(building_map_path)
        print(f"读取建筑文件: {len(features)} 个 features")
        for building_map in features:
            try:
                building = MapBuilding()
                building.init(building_map)
                self.building_list.append(building)
            except Exception as e:
                print(f'发生建筑读取错误:{e}, 跳过')
        print(f"共读入{len(self.building_list)}个建筑图形的数据.....")
        # (调试用) 打印前1000个建筑的几何集合
        # print(GeometryCollection([building.geometry for building in self.building_list[:1000]]))

        # --- 读取地块数据 (LOT) ---
        features = TileSystemGenerator.read_map_json(lot_map_path)
        print(f"读取 LOT 地块文件: {len(features)} 个 features")
        for lot_map in features:
            try:
                lot = MapLot()
                lot.init(lot_map)
                self.lot_list.append(lot)
            except Exception as e:
                print(f'发生地块(LOT)读取错误:{e}, 跳过')

        # --- 读取地块数据 (GLA) ---
        features = TileSystemGenerator.read_map_json(gla_map_path)
        print(f"读取 GLA 地块文件: {len(features)} 个 features")
        for lot_map in features:
             try:
                lot = MapLot() # 使用同一个 MapLot 类处理
                lot.init(lot_map)
                # 简单检查是否与已加载的 LOT 重复 (基于坐标可能更可靠，但这里简化处理)
                # 这里假设 LOT 和 GLA 的 ID 系统是独立的
                self.lot_list.append(lot)
             except Exception as e:
                print(f'发生地块(GLA)读取错误:{e}, 跳过')

        print(f"共读入{len(self.lot_list)}个地块图形的数据 (包含 LOT 和 GLA).....")
        print("原始地图数据读取完成。")


    @staticmethod
    def read_map_json(path: str):
        """静态方法：读取指定路径的 JSON 地图数据文件，并返回 'features' 列表。 (与 MapSite 中的版本相同)"""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
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

    def simplify_map_by_boundary(self, tile_boundary: Polygon, tile_name: str = "Xkool-Area-1", base_root: str = ""):
        """
        根据指定的瓦片边界 (tile_boundary) 裁切已加载的原始地图数据，并将结果导出到指定目录。

        Args:
            tile_boundary (Polygon): 用于裁切的瓦片边界。
            tile_name (str, optional): 瓦片的名称，用于创建输出目录和文件名。默认为 "Xkool-Area-1"。
            base_root (str, optional): 导出数据的根目录。如果为空，则使用默认路径。默认为 ""。

        Returns:
            tuple: 包含裁切后各种元素数量的元组 (road_center_line, road_polygon, building, lot)。
        """
        # 设置当前处理的瓦片边界
        self.tile_boundary = tile_boundary
        minx, miny, maxx, maxy = self.tile_boundary.bounds
        print(f"\n--- 开始处理瓦片: {tile_name} ---")
        print(f"【{tile_name} 瓦片范围框：{round(maxx - minx)}m * {round(maxy -miny)}m, 面积约{round(self.tile_boundary.area)}㎡】")
        print("正在根据瓦片边界筛选地图数据...")

        # 1. 筛选数据：根据瓦片边界过滤内存中的原始数据，填充 cropped_* 列表
        self.__renew_data_from_tile_boundary()

        # 2. 导出数据：将筛选后的 cropped_* 列表中的数据写入 JSON 文件
        self.__export_processed_and_simplified_data(tile_name=tile_name, base_root=base_root)

        print(f"--- 瓦片: {tile_name} 处理完成 ---")
        # 返回该瓦片包含的各种元素的数量
        return len(self.cropped_road_center_line_list), len(self.cropped_road_polygon_list), len(self.cropped_building_list), len(self.cropped_lot_list)

    def __export_processed_and_simplified_data(self, tile_name: str = "Xkool-Area-1", base_root: str = ""):
        """将当前瓦片裁切后的数据 (cropped_* 列表) 导出为 JSON 文件。"""
        # 确定导出目录
        if not base_root:
            # 如果未指定 base_root，则使用默认路径 (通常在 'library/HK_map/processed' 下)
             # 注意：这个默认路径可能与主脚本中的 library_path 不同，主脚本通常指向 TileBoundary
            base_root = os.path.abspath(os.path.join("library", "HK_map", "processed", tile_name))

        print(f"准备导出数据到: {base_root}")
        # 使用 PathUtils 确保目录存在 (如果不存在则创建)
        PathUtils.check_and_create_dir(base_root, whether_clean=False) # 不清空，因为是按瓦片写入

        # --- 导出处理后的道路中心线数据 ---
        data = {"type": "FeatureCollection", "features": []} # 标准 GeoJSON 格式
        for road_center_line in self.cropped_road_center_line_list:
             # 假设 Map* 对象有一个 row_data 属性存储原始 feature 字典
            if hasattr(road_center_line, 'row_data'):
                data["features"].append(road_center_line.row_data)
            else:
                 print(f"警告: RoadCenterLine 对象 {getattr(road_center_line, 'object_id', '未知ID')} 缺少 row_data 属性")

        road_center_line_map_json_path = os.path.join(base_root, "road_center_line_map.json")
        try:
            json_str = json.dumps(data, indent=4, ensure_ascii=False) # 格式化输出 JSON
            with open(road_center_line_map_json_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"  - {os.path.basename(road_center_line_map_json_path)} ({len(data['features'])} features) 生成成功")
        except Exception as e:
            print(f"错误: 导出道路中心线数据失败: {e}")

        # --- 导出处理后的道路图形数据 ---
        data = {"type": "FeatureCollection", "features": []}
        for road_polygon in self.cropped_road_polygon_list:
            if hasattr(road_polygon, 'row_data'):
                data["features"].append(road_polygon.row_data)
            else:
                print(f"警告: RoadPolygon 对象 {getattr(road_polygon, 'object_id', '未知ID')} 缺少 row_data 属性")

        road_polygon_map_json_path = os.path.join(base_root, "road_polygon_map.json")
        try:
            json_str = json.dumps(data, indent=4, ensure_ascii=False)
            with open(road_polygon_map_json_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"  - {os.path.basename(road_polygon_map_json_path)} ({len(data['features'])} features) 生成成功")
        except Exception as e:
             print(f"错误: 导出道路图形数据失败: {e}")

        # --- 导出处理后的建筑数据 ---
        data = {"type": "FeatureCollection", "features": []}
        for building in self.cropped_building_list:
             if hasattr(building, 'row_data'):
                data["features"].append(building.row_data)
             else:
                 print(f"警告: Building 对象 {getattr(building, 'object_id', '未知ID')} 缺少 row_data 属性")

        building_map_json_path = os.path.join(base_root, "building_map.json")
        try:
            json_str = json.dumps(data, indent=4, ensure_ascii=False)
            with open(building_map_json_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"  - {os.path.basename(building_map_json_path)} ({len(data['features'])} features) 生成成功")
        except Exception as e:
            print(f"错误: 导出建筑数据失败: {e}")

        # --- 导出处理后的地块 LOT 和 GLA 数据 (分开存储) ---
        lot_data = {"type": "FeatureCollection", "features": []}
        gla_data = {"type": "FeatureCollection", "features": []}
        for lot in self.cropped_lot_list:
            if hasattr(lot, 'row_data'):
                # 根据 is_GLA 属性判断是 LOT 还是 GLA
                if hasattr(lot, 'is_GLA') and lot.is_GLA:
                    gla_data["features"].append(lot.row_data)
                else:
                    lot_data["features"].append(lot.row_data)
            else:
                 print(f"警告: Lot 对象 {getattr(lot, 'lot_id', getattr(lot, 'gla_id', '未知ID'))} 缺少 row_data 属性")

        # 导出 LOT 数据
        lot_map_json_path = os.path.join(base_root, "lot_map.json")
        try:
            json_str = json.dumps(lot_data, indent=4, ensure_ascii=False)
            with open(lot_map_json_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"  - {os.path.basename(lot_map_json_path)} ({len(lot_data['features'])} features) 生成成功")
        except Exception as e:
             print(f"错误: 导出 LOT 数据失败: {e}")

        # 导出 GLA 数据
        gla_map_json_path = os.path.join(base_root, "gla_map.json")
        try:
            json_str = json.dumps(gla_data, indent=4, ensure_ascii=False)
            with open(gla_map_json_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"  - {os.path.basename(gla_map_json_path)} ({len(gla_data['features'])} features) 生成成功")
        except Exception as e:
             print(f"错误: 导出 GLA 数据失败: {e}")

        print(f"【瓦片 {tile_name} 地图数据导出已经完成】")

    def __renew_data_from_tile_boundary(self):
        """
        核心筛选逻辑：
        根据当前设置的 self.tile_boundary，遍历内存中的全部原始数据列表 (self.road_center_line_list 等)，
        将与该瓦片边界相交 (intersects) 的元素添加到对应的 cropped_* 列表和 ID 列表中。
        """
        # --- 更新道路中心线 ---
        self.cropped_road_center_line_list = []
        self.cropped_road_center_line_id_list = []
        # 遍历所有原始道路中心线
        for road_center_line in self.road_center_line_list:
            # 检查几何对象是否有效且与瓦片边界相交
            if road_center_line.geometry and road_center_line.geometry.is_valid and road_center_line.geometry.intersects(self.tile_boundary):
                self.cropped_road_center_line_list.append(road_center_line)
                if hasattr(road_center_line, 'object_id'):
                    self.cropped_road_center_line_id_list.append(road_center_line.object_id)
        print(f"筛选后保留的道路中心线数量: {len(self.cropped_road_center_line_list)}, id数量: {len(self.cropped_road_center_line_id_list)}")

        # --- 更新道路图形 ---
        self.cropped_road_polygon_list = []
        self.cropped_road_polygon_id_list = []
        for road_polygon in self.road_polygon_list:
            if road_polygon.geometry and road_polygon.geometry.is_valid and road_polygon.geometry.intersects(self.tile_boundary):
                self.cropped_road_polygon_list.append(road_polygon)
                if hasattr(road_polygon, 'object_id'):
                     self.cropped_road_polygon_id_list.append(road_polygon.object_id)
        print(f"筛选后保留的道路图形数量: {len(self.cropped_road_polygon_list)}, id数量: {len(self.cropped_road_polygon_id_list)}")

        # --- 更新建筑图形 ---
        self.cropped_building_list = []
        self.cropped_building_id_list = []
        for building in self.building_list:
            if building.geometry and building.geometry.is_valid and building.geometry.intersects(self.tile_boundary):
                self.cropped_building_list.append(building)
                if hasattr(building, 'object_id'):
                     self.cropped_building_id_list.append(building.object_id)
        print(f"筛选后保留的建筑图形数量: {len(self.cropped_building_list)}, id数量: {len(self.cropped_building_id_list)}")

        # --- 更新地块图形 (LOT 和 GLA) ---
        self.cropped_lot_list = []
        self.cropped_lot_id_list = []
        self.cropped_gla_id_list = []
        for lot in self.lot_list:
            if lot.geometry and lot.geometry.is_valid and lot.geometry.intersects(self.tile_boundary):
                self.cropped_lot_list.append(lot)
                # 根据 is_GLA 属性分别记录 ID
                if hasattr(lot, 'is_GLA') and lot.is_GLA:
                    if hasattr(lot, 'gla_id'):
                        self.cropped_gla_id_list.append(lot.gla_id)
                else:
                     if hasattr(lot, 'lot_id'):
                        self.cropped_lot_id_list.append(lot.lot_id)
        print(f"筛选后保留的地块数量: {len(self.cropped_lot_list)}", "LOT id 数量:", len(self.cropped_lot_id_list), "+ GLA id 数量:", len(self.cropped_gla_id_list))


    # --- 以下为属性方法，主要用于调试或快速获取几何集合 ---
    @property
    def cropped_building_geo(self):
        """返回裁切后建筑的 GeometryCollection 对象"""
        return GeometryCollection([building.geometry for building in self.cropped_building_list if building.geometry])

    @property
    def cropped_road_center_line_geo(self):
        """返回裁切后道路中心线的 GeometryCollection 对象"""
        return GeometryCollection([road_center_line.geometry for road_center_line in self.cropped_road_center_line_list if road_center_line.geometry])

    @property
    def cropped_road_polygon_geo(self):
        """返回裁切后道路图形的 GeometryCollection 对象"""
        return GeometryCollection([road_pl.geometry for road_pl in self.cropped_road_polygon_list if road_pl.geometry])

    @property
    def cropped_lot_geo(self):
        """返回裁切后地块的 GeometryCollection 对象"""
        return GeometryCollection([lot.geometry for lot in self.cropped_lot_list if lot.geometry])