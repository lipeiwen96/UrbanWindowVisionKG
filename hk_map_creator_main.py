from map_system.hk_3d_downloader import SpatialTileDownloader
from map_system.hk_map_processor import MapProcessor
import os
from shapely.geometry import box
from map_system.utils.map_utils import km


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 研究范围 & 路径配置
    # ------------------------------------------------------------------
    # research_box = (830000, 815000, 843500, 828800)
    research_box = (830000, 814400, 840500, 824600)
    SPATIAL_GEOJSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "3d_spatial_data", "b1000_Tile_Feb25.gdb_converted.json"))
    SPATIAL_TILES_OUT_DIR = os.path.abspath(os.path.join("library", "HK_map", "processed", "3d_spatial_tiles"))
    COPY_DIR = os.path.abspath(os.path.join("library", "HK_map", "processed", "project", "spatial"))

    # ------------------------------------------------------------------
    # 基本信息打印
    # ------------------------------------------------------------------
    research_geom = box(*research_box)
    width_m = research_box[2] - research_box[0]
    height_m = research_box[3] - research_box[1]
    area_m2 = research_geom.area
    print("================ Study Region =================")
    print(f"WKT        : {research_geom.wkt}")
    print(f"Width      : {km(width_m)} km  ({width_m:,.0f} m)")
    print(f"Height     : {km(height_m)} km  ({height_m:,.0f} m)")
    print(f"Area       : {round(area_m2 / 1e6, 3)} km²  ({area_m2:,.0f} m²)")
    print("===============================================\n")

    # 下载、载入并创建地形
    files = SpatialTileDownloader.download_b1000_tiles(
        min_x=research_box[0],
        min_y=research_box[1],
        max_x=research_box[2],
        max_y=research_box[3],
        fmt="3DS",
        geojson_path=SPATIAL_GEOJSON_PATH,
        out_dir=SPATIAL_TILES_OUT_DIR,
        copy_dir=COPY_DIR,
        workers=12,
        overwrite=False,  # 如需重新下载改为 True
        load_terrain=False
    )
    # print("\nDownloaded / existing tile folders:")
    # for f in files:
    #     print(" ·", f)

    # --- 调用 BuildingProcessor 处理建筑数据 ---
    # --- 新增选项：是否执行详细分析 ---
    PERFORM_BUILDING_ANALYSIS = True  # 设置为 True 执行分析, False 则跳过
    BUILDING_GEOJSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "building", "BUILDING_STRUCTURE.json"))
    LOT_GEOJSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "lot", "LOT.json"))
    GLA_GEOJSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "lot", "GovernmentLandAllocation.json"))
    INV_GEOJSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "road", "INV_PG.json"))  # 道路
    HKBD_GEOJSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "DCD_hk80.json"))  # 道路
    GREENING_GEOJSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "greening", "greening_format.json"))  # 道路
    MODEL_OUT_DIR = os.path.abspath(os.path.join("library", "HK_map", "processed", "model"))
    print("--- 调用 MAP-Processor ---")
    hk_map = MapProcessor(
        building_geojson=BUILDING_GEOJSON_PATH,
        lot_geojson=LOT_GEOJSON_PATH,
        gla_geojson=GLA_GEOJSON_PATH,
        inv_geojson=INV_GEOJSON_PATH,
        hkbd_geojson=HKBD_GEOJSON_PATH,
        gn_geojson=GREENING_GEOJSON_PATH,
        research_box=research_box, min_building_area=90, min_lot_area=100
    )
    hk_map.run(out_dir=MODEL_OUT_DIR, analyse=PERFORM_BUILDING_ANALYSIS)

    print(f"--- 主程序: MAP-Processor 处理完成 ---")


    # 读取建筑

    # 读取地块

    # 读取DEM

    # 创建树

    # 创建窗户

    # 创建水