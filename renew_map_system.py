"""
地图数据库自动刷新脚本
- 读取最新的分层地图数据、Tile边界数据
- 按Tile边界切分地图后存储进本地数据库

数据库刷新可能用时10min，请耐心等待，有问题请联系 @Peiwen
"""
import json
import os
from shapely.geometry import GeometryCollection
from map_system.tile_system_generator import TileSystemGenerator
from map_system.tile_genenrator.row_map_data_reader import MapGeoReader
from map_system.map_site_creator import HKTile
from art_public_modules.art_utils.common_utils.path_utils import PathUtils
import platform


class XKOOLMapSystem:

    def init(self):
        # STEP1：读取HK-TILE信息
        self.read_row_tile_index()

        # STEP2：自动裁切地图
        self.auto_crop_map()

    def read_row_tile_index(self):
        print(f"当前系统为{platform.system()}")
        if platform.system() == "Windows":
            tile_index_path = os.path.abspath(os.path.join("library", "HK_map", "row_map", "TileIndex.json"))
        else:
            tile_index_path = "/map_data/HK_map/row_map/TileIndex.json"
            # tile_index_path = os.path.join("map_data", "HK_map", "row_map", "TileIndex.json")

        tile_data = TileSystemGenerator.read_map_json(tile_index_path)
        self.tile_list = []
        for tile_row_data in tile_data:
            tile_boundary = MapGeoReader.geo_reader(tile_row_data["geometry"]["type"], tile_row_data["geometry"]["coordinates"])
            tile_name = tile_row_data["properties"]["SHEETNO"]
            tile_id = tile_row_data["properties"]["OBJECTID"]
            self.tile_list.append(HKTile(tile_id=tile_id, tile_name=tile_name, tile_boundary=tile_boundary))
        print("所有TILE图形数据")
        print(GeometryCollection([tile.tile_boundary for tile in self.tile_list]))

    def auto_crop_map(self):
        # 清空地图文件夹
        if platform.system() == "Windows":
            library_path = os.path.abspath(os.path.join("library", "HK_map", "TileBoundary"))
        else:
            library_path = "/map_data/HK_map/TileBoundary"
            # library_path = os.path.join("map_data", "HK_map", "TileBoundary")
        PathUtils.check_and_create_dir(library_path, whether_clean=True)

        if platform.system() == "Windows":
            base_path = os.path.abspath(os.path.join("library", "HK_map", "row_map"))
        else:
            base_path = "/map_data/HK_map/row_map"
            # base_path = os.path.join("map_data", "HK_map", "row_map")

        file_path1 = os.path.join(base_path, "road", "GEO_STREET_CENTRELINE.json")
        file_path2 = os.path.join(base_path, "building", "BUILDING_STRUCTURE.json")
        file_path3 = os.path.join(base_path, "lot", "LOT.json")
        file_path4 = os.path.join(base_path, "lot", "GovernmentLandAllocation.json")
        file_path5 = os.path.join(base_path, "road", "INV_PG.json")
        t = TileSystemGenerator()
        t.read_row_data(file_path1, file_path2, file_path3, file_path4, file_path5)

        for tile in self.tile_list:
            tile_dir_path = os.path.join(library_path, tile.tile_name)
            tile.road_center_line_num, tile.road_polygon_num, tile.building_num, tile.lot_num\
                = t.simplify_map_by_boundary(tile_boundary=tile.tile_boundary, tile_name=tile.tile_name, base_root=tile_dir_path)  # 裁切地图+自动导出数据
            # print(tile.tile_boundary)
            # print(t.cropped_building_geo)
            # print(t.cropped_road_center_line_geo)
            # print(t.cropped_road_polygon_geo)
            # print(t.cropped_lot_geo)

        library_tiles_path = os.path.join(library_path, "HK_Tiles.json")
        data = {"simplified_tiles": []}
        for tile in self.tile_list:
            data["simplified_tiles"].append(tile.to_dict)
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(library_tiles_path, "w", encoding="utf-8") as f:
            f.write(json_str)


if __name__ == "__main__":
    """
    最新TileIndex下载链接 搜索：Digital Topographic Map iB1000
    https://portal.csdi.gov.hk/geoportal/#metadataInfoPanel
    """
    hk_map = XKOOLMapSystem()
    hk_map.init()