# -*- coding: utf-8 -*-
"""hk_map_processor.py – 统一处理 *建筑 + 地块* 并导出 Rhino 3DM

* 面向 CSDI Hong Kong：Building CSU、Lot / GLA、道路等
* 取消 @staticmethod，全部转实例方法，属性通过 **self.*** 持久；便于外部多次调用或继承
* 新增 **Lot**（或任何地块 GeoJSON）加载 → `self.lots`
* 建模阶段同时写入地块（低平台）与建筑（高体量）
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box, GeometryCollection

matplotlib.use("Agg")

# -- project imports --------------------------------------------------------
from map_system.utils.map_structure import MapBaseGeometry, MapBuilding, MapLot, MapRoadPolygon, MapAdminBoundary, MapGreeningBoundary  # noqa: E402
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial
from art_public_modules.art_data_exchange.data_exchange import ARTShapelyDataExchanger
from map_system.map_cropper import RobustMapCropper

# ---------------------------------------------------------------------------
# Helper utilities (unchanged impl.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 辅助工具类 (实现不变，备注改为中文)
# ---------------------------------------------------------------------------

class _PlotUtils:
    """绘图相关的辅助方法"""
    @staticmethod
    def save(fig, out_dir: Path, name: str) -> None:
        """保存 matplotlib 图表到文件"""
        print(f"    正在保存图表: {name}")
        try:
            fig.savefig(out_dir / name, bbox_inches="tight", dpi=150)
        except Exception as e:
            print(f"    错误：保存图表 {name} 时出错: {e}")
        finally:
            plt.close(fig) # 关闭图表释放内存

    @staticmethod
    def trimmed_hist(s: pd.Series, *, bins: int, q: Tuple[float, float]):
        """计算并返回修剪（按分位数）后的直方图数据"""
        if s.empty:
            # print("    警告: 用于直方图的数据序列为空。") # 减少打印
            return np.array([]), np.array([])
        try:
            s_numeric = pd.to_numeric(s, errors='coerce').dropna()
            if s_numeric.empty:
                # print("    警告: 清理后，用于直方图的数据序列为空。") # 减少打印
                return np.array([]), np.array([])

            lo, hi = s_numeric.quantile(q).tolist();
            if hi - lo < 1e-9:
                 # print(f"    警告: 数据序列在分位数 {q} 内的值范围过小 ({lo:.2f} - {hi:.2f})，无法生成有效直方图。") # 减少打印
                 return np.array([]), np.array([])

            clipped = s_numeric.clip(lo, hi)
            actual_bins = max(1, bins)
            counts, edges = np.histogram(clipped, bins=actual_bins)
            return counts, edges
        except Exception as e:
            print(f"    警告: 计算直方图时出错: {e}")
            return np.array([]), np.array([])


class _Json:
    """JSON 文件操作辅助方法"""
    @staticmethod
    def dump(obj: Dict[str, Any], out_dir: Path, name: str) -> None:
        """将字典对象保存为 JSON 文件"""
        filepath = out_dir / name
        print(f"    正在保存 JSON 数据到: {filepath}")
        try:
            with filepath.open("w", encoding="utf-8") as fp:
                json.dump(obj, fp, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"    错误：保存 JSON 文件 {name} 时出错: {e}")

# ---------------------------------------------------------------------------
# 主处理类
# ---------------------------------------------------------------------------

class MapProcessor:
    """加载多种地理空间要素 GeoJSON，进行分析并导出 3DM 文件。"""

    # ---------- 初始化 ----------
    def __init__(
        self,
        building_geojson: Union[str, Path],
        lot_geojson: Union[str, Path] | None = None,
        gla_geojson: Union[str, Path] | None = None,
        inv_geojson: Union[str, Path] | None = None,
        hkbd_geojson: Union[str, Path] | None = None, # 新增行政区文件参数
        gn_geojson: Union[str, Path] | None = None,   # 新增绿地文件参数
        *,
        research_box: Optional[Tuple[float, float, float, float]] = None,
        min_building_area: float = 90,
        min_lot_area: float = 100,
        min_road_area: float = 10,
        min_admin_area: float = 100, # 新增最小行政区面积阈值 (示例)
        min_green_space_area: float = 50, # 新增最小绿地面积阈值 (示例)
        # 新增：进度提示频率
        progress_interval: int = 10000,
        # 新增：用于绘图的字体名称
        plot_font_name: str = 'SimHei' # 默认为 SimHei，可根据环境修改
    ) -> None:
        """
        初始化地图处理器。

        Args:
            building_geojson: 建筑 GeoJSON 文件路径。
            lot_geojson: LOT 地块 GeoJSON 文件路径 (可选)。
            gla_geojson: GLA 地块 GeoJSON 文件路径 (可选)。
            inv_geojson: INV 道路 GeoJSON 文件路径 (可选)。
            hkbd_geojson: HKBD 行政区 GeoJSON 文件路径 (可选)。
            gn_geojson: GN 绿地 GeoJSON 文件路径 (可选)。
            research_box: 研究范围边界框 (minx, miny, maxx, maxy) (可选)。
            min_building_area: 最小保留建筑面积阈值。
            min_lot_area: 最小保留地块面积阈值 (裁剪后)。
            min_road_area: 最小保留道路面积阈值 (裁剪后)。
            min_admin_area: 最小保留行政区面积阈值 (裁剪后)。
            min_green_space_area: 最小保留绿地面积阈值 (裁剪后)。
            progress_interval: 处理多少个要素后打印一次进度。
            plot_font_name: 用于 Matplotlib 绘图的字体名称 (需确保已安装)。
        """
        print("--- 初始化地图处理器 ---")
        # --- 设置 Matplotlib 中文字体 ---
        try:
            plt.rcParams['font.sans-serif'] = [plot_font_name]
            plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
            print(f"  已设置 Matplotlib 字体为: {plot_font_name}")
        except Exception as e:
            print(f"  警告: 设置 Matplotlib 字体 '{plot_font_name}' 时出错: {e}。图表中文可能显示为乱码。")
            print(f"  请确保字体已安装或尝试其他字体名称 (如 'Microsoft YaHei', 'Source Han Sans CN')。")
        # ---------------------------------

        # 文件路径
        self.src_bldg = Path(building_geojson).resolve()
        self.src_lot = Path(lot_geojson).resolve() if lot_geojson else None
        self.gla_geojson = Path(gla_geojson).resolve() if gla_geojson else None
        self.inv_geojson = Path(inv_geojson).resolve() if inv_geojson else None
        self.hkbd_geojson = Path(hkbd_geojson).resolve() if hkbd_geojson else None
        self.gn_geojson = Path(gn_geojson).resolve() if gn_geojson else None

        # 处理参数
        self.research_box = research_box
        self.progress_interval = progress_interval

        # 面积阈值
        self.min_building_area = min_building_area
        self.min_lot_area = min_lot_area
        self.min_road_area = min_road_area
        self.min_admin_area = min_admin_area
        self.min_green_space_area = min_green_space_area

        # 数据存储列表
        self.buildings: List[MapBuilding] = []
        self.lots: List[MapLot] = []
        self.roads: List[MapRoadPolygon] = []
        self.admin_boundaries: List[MapAdminBoundary] = []  # 新增
        self.green_spaces: List[MapGreeningBoundary] = []     # 新增

        # ID 追踪列表 (注意：LOT 和 GLA 共享 lot_id_list)
        self.lot_id_list: List[Any] = []
        self.road_id_list: List[Any] = []
        self.admin_boundary_id_list: List[Any] = [] # 新增
        self.green_space_id_list: List[Any] = []    # 新增

        # 打印初始化参数
        print(f"  建筑 GeoJSON: {self.src_bldg}")
        print(f"  LOT 地块 GeoJSON: {self.src_lot if self.src_lot else '未提供'}")
        print(f"  GLA 地块 GeoJSON: {self.gla_geojson if self.gla_geojson else '未提供'}")
        print(f"  INV 道路 GeoJSON: {self.inv_geojson if self.inv_geojson else '未提供'}")
        print(f"  HKBD 行政区 GeoJSON: {self.hkbd_geojson if self.hkbd_geojson else '未提供'}")
        print(f"  GN 绿地 GeoJSON: {self.gn_geojson if self.gn_geojson else '未提供'}")

        if self.research_box:
            print(f"  研究范围 (minx, miny, maxx, maxy): {self.research_box}")
            try:
                self.research_box_polygon = box(*self.research_box)
                print(f"  已创建研究范围边界多边形用于裁剪。")
            except Exception as e:
                print(f"  警告: 无法创建研究范围多边形: {e}。裁剪功能将禁用。")
                self.research_box_polygon = None
        else:
            print("  未定义研究范围 (裁剪功能禁用)。")
            self.research_box_polygon = None

        print(f"  最小建筑面积阈值: {self.min_building_area} 平方米")
        print(f"  最小地块面积阈值: {self.min_lot_area} 平方米")
        print(f"  最小道路面积阈值: {self.min_road_area} 平方米")
        print(f"  最小行政区面积阈值: {self.min_admin_area} 平方米")
        print(f"  最小绿地面积阈值: {self.min_green_space_area} 平方米")
        print(f"  进度提示间隔: 每处理 {self.progress_interval} 个要素")
        print("------------------------------------")

    # ---------- 公共执行入口 ----------
    def run(self, out_dir: Union[str, Path], analyse: bool = True, *, bins=50, q=(0.01, 0.99)) -> None:
        """
        执行地图处理流程：加载、(可选)分析、导出。

        Args:
            out_dir: 输出目录路径。
            analyse: 是否执行数据分析步骤。
            bins: 分析时直方图的箱数。
            q: 分析时直方图的分位数裁剪范围。
        """
        print(f"\n=== 开始执行地图处理流程 ===")
        self.out = Path(out_dir).resolve();
        print(f"  输出目录: {self.out}")
        self.out.mkdir(parents=True, exist_ok=True) # 确保输出目录存在

        print("\n--- 步骤 1: 加载建筑数据 ---")
        self._load_buildings()
        print("--- 建筑数据加载完成 ---")

        print("\n--- 步骤 2: 加载并裁剪地块数据 (LOT & GLA) ---")
        self._load_lots()
        print("--- 地块数据加载完成 ---")

        print("\n--- 步骤 3: 加载并裁剪道路数据 (INV) ---")
        self._load_roads()
        print("--- 道路数据加载完成 ---")

        print("\n--- 步骤 4: 加载并裁剪行政区数据 (HKBD) ---") # 新增
        self._load_admin_boundaries()
        print("--- 行政区数据加载完成 ---") # 新增

        print("\n--- 步骤 5: 加载并裁剪绿地数据 (GN) ---") # 新增
        self._load_green_spaces()
        print("--- 绿地数据加载完成 ---") # 新增

        if analyse:
            print("\n--- 步骤 6: 分析数据并生成分布图 ---") # 步骤编号更新
            self._analyse(bins=bins, q=q)
            print("--- 数据分析与绘图完成 ---")
        else:
            print("\n--- 步骤 6: 跳过数据分析与绘图 ---") # 步骤编号更新

        print("\n--- 步骤 7: 导出为 3DM 文件 ---") # 步骤编号更新
        self._export_3dm()
        print("--- 3DM 文件导出完成 ---")
        print("\n=== 地图处理流程执行完毕 ===")

    # ---------- 数据加载方法 ----------
    def _load_buildings(self):
        """加载建筑 GeoJSON 文件，根据条件过滤并存入 self.buildings。"""
        # (此方法与上一版本相同，保持不变)
        if not self.src_bldg or not self.src_bldg.is_file():
             print(f"  错误: 建筑文件未找到或未指定 ({self.src_bldg})")
             return

        print(f"  正在读取建筑 GeoJSON: {self.src_bldg}")
        try:
            gj = json.loads(self.src_bldg.read_text(encoding="utf-8"))
            feats = gj.get("features", [])
            total_features = len(feats)
            print(f"  在建筑文件中找到 {total_features} 个要素。")
            if total_features == 0: return
        except Exception as e:
            print(f"  错误: 读取或解析建筑 GeoJSON 时出错: {e}")
            return

        prep_bound = None
        if self.research_box_polygon:
             try:
                 from shapely.prepared import prep
                 prep_bound = prep(self.research_box_polygon)
                 print(f"  已准备研究范围用于检查建筑 *包含* 情况。")
             except Exception as e:
                 print(f"  警告: 无法准备几何图形用于建筑包含性检查: {e}。边界检查将跳过。")
                 prep_bound = None

        loaded_building_count = 0
        skipped_filter_count = 0
        skipped_boundary_count = 0
        skipped_error_count = 0

        print(f"  开始处理 {total_features} 个建筑要素...")
        for i, f in enumerate(feats):
            current_count = i + 1
            if current_count % self.progress_interval == 0 or current_count == total_features:
                print(f"    已处理 {current_count}/{total_features} 个建筑要素...")

            try:
                b = MapBuilding()
                b.init(row_data=f)

                if (b.geometry is None or b.geometry.is_empty or
                    b.height <= 0.1 or b.area <= self.min_building_area or
                    getattr(b, 'status', None) != "A"):
                    skipped_filter_count += 1
                    continue

                if prep_bound and not prep_bound.contains(b.geometry):
                    skipped_boundary_count += 1
                    continue

                self.buildings.append(b)
                loaded_building_count += 1

            except Exception as e:
                skipped_error_count += 1
                continue

        print(f"  建筑要素处理完成。")
        print(f"  总要素数: {total_features}")
        print(f"  成功加载的建筑: {loaded_building_count}")
        print(f"  因过滤条件跳过: {skipped_filter_count}")
        print(f"  因超出边界跳过: {skipped_boundary_count}")
        print(f"  因处理错误跳过: {skipped_error_count}")

    # ---------- 新增：通用多边形要素加载函数 ----------
    def _load_polygon_features(
        self,
        geojson_path: Optional[Path],
        map_object_class: Type[MapBaseGeometry], # 使用 Type 进行类提示
        target_list: List[MapBaseGeometry],
        id_attribute: str,
        id_tracking_list: List[Any],
        min_area_threshold: float,
        feature_type_name: str,
        extra_attributes: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int, int]: # 返回 (处理数, 添加数, 总要素数)
        """
        通用的加载、裁剪和过滤多边形要素的函数。

        Args:
            geojson_path: GeoJSON 文件路径。
            map_object_class: 用于实例化要素的地图结构类 (如 MapLot)。
            target_list: 存储最终结果的实例列表 (如 self.lots)。
            id_attribute: 要素 JSON 中唯一标识符的属性名 (如 'lot_id', 'object_id')。
                         对于 GLA，可能需要特殊处理或传入 'gla_id'。
            id_tracking_list: 用于追踪已添加要素 ID 的实例列表 (如 self.lot_id_list)。
            min_area_threshold: 最小面积阈值。
            feature_type_name: 要素类型的中文名称 (用于打印信息)。
            extra_attributes: 一个字典，包含需要在实例化后设置到对象上的额外属性 (如 {'is_GLA': True})。

        Returns:
            一个元组，包含 (处理的要素数, 最终添加的片段数, 文件中的总要素数)。
        """
        if not geojson_path or not geojson_path.is_file():
            print(f"  信息: 未提供或未找到 {feature_type_name} 文件: {geojson_path}，跳过加载。")
            return 0, 0, 0

        print(f"\n  正在读取 {feature_type_name} GeoJSON: {geojson_path}")
        try:
            gj = json.loads(geojson_path.read_text(encoding="utf-8"))
            features = gj.get("features", [])
            total_features = len(features)
            print(f"  在 {feature_type_name} 文件中找到 {total_features} 个要素。")
        except Exception as e:
            print(f"  错误: 读取或解析 {feature_type_name} GeoJSON 时出错: {e}")
            return 0, 0, 0 # 返回 0 表示无法处理

        if total_features == 0:
            return 0, 0, 0

        processed_count = 0
        added_pieces = 0
        print(f"  开始处理 {total_features} 个 {feature_type_name} 要素...")

        for i, f in enumerate(features):
            current_progress_count = i + 1
            feature_id_str = f"{feature_type_name} 要素 {current_progress_count}/{total_features}"
            # 打印进度
            if current_progress_count % self.progress_interval == 0 or current_progress_count == total_features:
                print(f"    已处理 {current_progress_count}/{total_features} 个 {feature_type_name} 要素...")

            # try:
            # 1. 初始化地图对象
            map_object = map_object_class()
            # 假设 init 方法处理属性映射，或者需要在这里手动映射
            # 对于 MapLot/GLA，init(f) 可能已足够
            # 对于其他类型，可能需要调整 init 或在这里处理
            map_object.init(row_data=f)
            # print(map_object.geometry.wkt)
            processed_count += 1

            # 2. 基本有效性检查
            if map_object.geometry is None or map_object.geometry.is_empty: continue

            # 3. 裁剪或获取原始对象列表
            objects_to_process: List[MapBaseGeometry] = []
            if self.research_box_polygon:
                clipped_objects = RobustMapCropper.clip_geometry(map_object.geometry, self.research_box_polygon)
                # print(GeometryCollection(clipped_objects).wkt)

                # 对输出的图形继承MapBaseGeometry的数据
                for obj in clipped_objects:
                    map_object_copy = copy.deepcopy(map_object)
                    map_object_copy.geometry = obj
                    objects_to_process.append(map_object_copy)
            else:
                objects_to_process.append(map_object)

            # 4. 处理每个片段
            for processed_obj in objects_to_process:
                # 检查面积阈值
                if processed_obj.geometry.area < min_area_threshold: continue

                target_list.append(processed_obj)
                added_pieces += 1

            # except Exception as e:
            #     print(f"  处理 {feature_id_str} 时出错: {e}。已跳过。")
            #     continue

        print(f"  {feature_type_name} 要素处理完成。")
        return processed_count, added_pieces, total_features

    # ---------- 修改后的加载方法 ----------
    def _load_lots(self):
        """加载 LOT 和 GLA 地块数据，使用通用加载函数。"""
        # LOT 和 GLA 共享 self.lots 列表和 self.lot_id_list 追踪列表
        processed_lot, added_lot, total_lot = self._load_polygon_features(
            geojson_path=self.src_lot,
            map_object_class=MapLot,
            target_list=self.lots,
            id_attribute='lot_id', # LOT 使用 lot_id
            id_tracking_list=self.lot_id_list,
            min_area_threshold=self.min_lot_area,
            feature_type_name="LOT 地块",
            extra_attributes={'is_GLA': False} # 明确设置非 GLA
        )

        processed_gla, added_gla, total_gla = self._load_polygon_features(
            geojson_path=self.gla_geojson,
            map_object_class=MapLot, # 仍然使用 MapLot 类
            target_list=self.lots, # 添加到同一个 self.lots 列表
            # 尝试获取 gla_id，如果 MapLot.init 能处理，否则可能需要调整 MapLot 或这里的逻辑
            # 假设 MapLot.init 后，可以通过 gla_id 或 lot_id 获取标识符
            id_attribute='gla_id', # 优先使用 gla_id
            id_tracking_list=self.lot_id_list, # 使用同一个追踪列表
            min_area_threshold=self.min_lot_area,
            feature_type_name="GLA 地块",
            extra_attributes={'is_GLA': True} # 设置为 GLA
        )

        print("\n  地块加载与裁剪总结:")
        print(f"  处理的 LOT 要素总数: {processed_lot}/{total_lot}")
        print(f"  处理的 GLA 要素总数: {processed_gla}/{total_gla}")
        print(f"  最终添加的地块片段总数 (裁剪和过滤后): {added_lot + added_gla}") # 合计添加数量
        print(f"  添加的唯一地块 ID 总数: {len(self.lot_id_list)}")

    def _load_roads(self):
        """加载 INV 道路数据，使用通用加载函数。"""
        processed_road, added_road, total_road = self._load_polygon_features(
            geojson_path=self.inv_geojson,
            map_object_class=MapRoadPolygon, # 使用道路类
            target_list=self.roads,          # 添加到 self.roads 列表
            id_attribute='object_id',        # 假设道路使用 object_id
            id_tracking_list=self.road_id_list, # 使用独立的道路 ID 列表
            min_area_threshold=self.min_road_area,
            feature_type_name="道路",
            extra_attributes=None # 没有额外属性需要设置
        )
        print("\n  道路加载与裁剪总结:")
        print(f"  处理的 INV 要素总数: {processed_road}/{total_road}")
        print(f"  最终添加的道路片段总数 (裁剪和过滤后): {added_road}")
        print(f"  添加的唯一道路 ID 总数: {len(self.road_id_list)}")

    # ---------- 新增加载方法 ----------
    def _load_admin_boundaries(self):
        """加载 HKBD 行政区数据，使用通用加载函数。"""
        # 假设行政区使用 object_id
        processed, added, total = self._load_polygon_features(
            geojson_path=self.hkbd_geojson,
            map_object_class=MapAdminBoundary, # 使用行政区类
            target_list=self.admin_boundaries, # 添加到新列表
            id_attribute='object_id',         # 假设使用 object_id
            id_tracking_list=self.admin_boundary_id_list, # 独立追踪列表
            min_area_threshold=self.min_admin_area,
            feature_type_name="行政区",
            extra_attributes=None
        )
        print("\n  行政区加载与裁剪总结:")
        print(f"  处理的 HKBD 要素总数: {processed}/{total}")
        print(f"  最终添加的行政区片段总数: {added}")
        print(f"  添加的唯一行政区 ID 总数: {len(self.admin_boundary_id_list)}")

    def _load_green_spaces(self):
        """加载 GN 绿地数据，使用通用加载函数。"""
        # 假设绿地使用 object_id
        processed, added, total = self._load_polygon_features(
            geojson_path=self.gn_geojson,
            map_object_class=MapGreeningBoundary, # 使用绿地类
            target_list=self.green_spaces, # 添加到新列表
            id_attribute='object_id',      # 假设使用 object_id
            id_tracking_list=self.green_space_id_list, # 独立追踪列表
            min_area_threshold=self.min_green_space_area,
            feature_type_name="绿地",
            extra_attributes=None
        )
        print("\n  绿地加载与裁剪总结:")
        print(f"  处理的 GN 要素总数: {processed}/{total}")
        print(f"  最终添加的绿地片段总数: {added}")
        print(f"  添加的唯一绿地 ID 总数: {len(self.green_space_id_list)}")

    # ---------- 数据分析与绘图方法 (修改) ----------
    def _analyse(self, *, bins: int, q: Tuple[float, float]):
        """对加载的建筑、地块、道路等数据进行统计分析，并生成分布直方图。"""
        print("  开始进行数据分析与分布图生成...")
        # 初始化最终的统计结果字典
        combined_stats = {
            "分析参数": {"bins": bins, "quantile_trim": q},
            "建筑统计": {"数量": 0, "数值总结": {}, "直方图": {}},
            "地块统计": {"数量": 0, "直方图": {}},
            "道路统计": {"数量": 0, "直方图": {}},
            "行政区统计": {"数量": 0}, # 新增
            "绿地统计": {"数量": 0}   # 新增
        }

        # --- 分析建筑 ---
        # (逻辑同上一版本)
        if self.buildings:
            print(f"  正在分析 {len(self.buildings)} 个建筑...")
            combined_stats["建筑统计"]["数量"] = len(self.buildings)
            try:
                bldg_data = {
                    "height": [max(b.height, 0) for b in self.buildings if b.height is not None],
                    "area": [max(b.area, 0) for b in self.buildings if b.area is not None],
                    "start": [max(b.start_height, 0) for b in self.buildings if b.start_height is not None]
                }
                df_bldg = pd.DataFrame(bldg_data)
                num_sum, hists = {}, {}
                for col, title_cn in [("height", "建筑高度"), ("area", "建筑面积"), ("start", "建筑起始高度")]:
                     if col not in df_bldg.columns: continue
                     s = df_bldg[col]
                     if s.empty: continue
                     desc = s.describe(percentiles=[.25, .5, .75])
                     num_sum[col] = {k: round(float(desc[k]), 2) for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]}
                     try:
                         cnt, edge = _PlotUtils.trimmed_hist(s, bins=bins, q=q)
                         if edge.size > 0 and cnt.size > 0 :
                             hists[col] = {"bins": edge.round(2).tolist(), "counts": cnt.tolist()}
                             fig, ax = plt.subplots();
                             ax.hist(s, bins=edge, edgecolor="black");
                             ax.set_title(f"{title_cn} 分布 (分位数裁剪 {q})")
                             ax.set_xlabel(f"{title_cn}")
                             ax.set_ylabel("数量")
                             _PlotUtils.save(fig, self.out, f"hist_{title_cn}.png")
                     except Exception as e:
                         print(f"    错误: 为 {title_cn} 生成直方图时出错: {e}")
                combined_stats["建筑统计"]["数值总结"] = num_sum
                combined_stats["建筑统计"]["直方图"] = hists
            except Exception as e:
                 print(f"  错误: 分析建筑数据时出错: {e}")
        else:
            print("  没有加载建筑数据，跳过建筑分析。")

        # --- 分析地块面积 (区分 LOT 和 GLA) ---
        # (逻辑同上一版本)
        if self.lots:
            print(f"  正在分析 {len(self.lots)} 个地块片段...")
            combined_stats["地块统计"]["数量"] = len(self.lots)
            lot_areas_lot = []
            lot_areas_gla = []
            try:
                for lot in self.lots:
                    if lot.geometry:
                        if getattr(lot, 'is_GLA', False): lot_areas_gla.append(lot.geometry.area)
                        else: lot_areas_lot.append(lot.geometry.area)
                all_lot_areas = lot_areas_lot + lot_areas_gla
                if all_lot_areas:
                    s_all_lots = pd.Series(all_lot_areas)
                    cnt_all, edge_all = _PlotUtils.trimmed_hist(s_all_lots, bins=bins, q=q)
                    if edge_all.size > 0:
                        combined_stats["地块统计"]["直方图"]["area_combined"] = {
                            "bins": edge_all.round(2).tolist(), "counts": cnt_all.tolist()
                        }
                        fig, ax = plt.subplots(); colors = []; labels = []; data_to_plot = []
                        if lot_areas_lot:
                            colors.append('gray'); labels.append(f'LOT 地块 ({len(lot_areas_lot)})')
                            data_to_plot.append(pd.Series(lot_areas_lot).clip(edge_all[0], edge_all[-1]))
                        if lot_areas_gla:
                            colors.append('green'); labels.append(f'GLA 地块 ({len(lot_areas_gla)})')
                            data_to_plot.append(pd.Series(lot_areas_gla).clip(edge_all[0], edge_all[-1]))
                        if data_to_plot:
                            ax.hist(data_to_plot, bins=edge_all, color=colors, label=labels, alpha=0.7, edgecolor="black", stacked=False)
                            ax.set_title(f"地块面积 分布 (分位数裁剪 {q})"); ax.set_xlabel("地块面积 (平方米)"); ax.set_ylabel("数量"); ax.legend()
                            _PlotUtils.save(fig, self.out, "hist_地块面积_区分LOT_GLA.png")
                        else: print("    警告: 没有有效的 LOT 或 GLA 数据用于绘制区分颜色的地块面积直方图。")
                    else: print("    警告: 未能为地块面积生成有效的直方图边界。")
                else: print("    警告: 地块面积数据为空。")
            except Exception as e: print(f"  错误: 分析地块面积时出错: {e}")
        else: print("  没有加载地块数据，跳过地块面积分析。")

        # --- 分析道路面积 ---
        # (逻辑同上一版本)
        if self.roads:
            print(f"  正在分析 {len(self.roads)} 个道路片段...")
            combined_stats["道路统计"]["数量"] = len(self.roads)
            try:
                road_areas = [road.geometry.area for road in self.roads if road.geometry]
                if road_areas:
                    s_roads = pd.Series(road_areas)
                    cnt, edge = _PlotUtils.trimmed_hist(s_roads, bins=bins, q=q)
                    if edge.size > 0 and cnt.size > 0 :
                         combined_stats["道路统计"]["直方图"]["area"] = {"bins": edge.round(2).tolist(), "counts": cnt.tolist()}
                         fig, ax = plt.subplots();
                         ax.hist(s_roads, bins=edge, edgecolor="black");
                         ax.set_title(f"道路面积 分布 (分位数裁剪 {q})"); ax.set_xlabel("道路面积 (平方米)"); ax.set_ylabel("数量")
                         _PlotUtils.save(fig, self.out, "hist_道路面积.png")
                    else: print("    警告: 未能为道路面积生成有效的直方图数据。")
            except Exception as e: print(f"  错误: 分析道路面积时出错: {e}")
        else: print("  没有加载道路数据，跳过道路面积分析。")

        # --- 统计行政区和绿地数量 (不生成直方图) ---
        if self.admin_boundaries:
            combined_stats["行政区统计"]["数量"] = len(self.admin_boundaries)
            print(f"  统计了 {len(self.admin_boundaries)} 个行政区片段。")
        else:
            print("  没有加载行政区数据。")

        if self.green_spaces:
            combined_stats["绿地统计"]["数量"] = len(self.green_spaces)
            print(f"  统计了 {len(self.green_spaces)} 个绿地片段。")
        else:
            print("  没有加载绿地数据。")

        # 将合并的统计结果保存到 JSON 文件
        _Json.dump(combined_stats, self.out, "combined_stats.json")
        print("  合并的统计分析与直方图数据已保存到 combined_stats.json")


    # ---------- 导出方法 ----------
    def _export_3dm(self):
        """将加载和处理后的建筑、地块、道路、行政区、绿地数据导出为 Rhino 3DM 文件。"""
        print(f"  准备用于 Rhino 导出的数据模型...")
        model = DataModel(name="地图模型")

        # 添加研究边界底座
        if self.research_box_polygon:
            try:
                model.insert_element(DataElement(
                    self.research_box_polygon, layer="Base", start_height=-80, height=75))
                print(f"  已添加研究范围边界元素。")
            except Exception as e: print(f"  警告: 无法添加研究范围边界元素: {e}")

        # 添加地块
        print(f"  正在向模型添加 {len(self.lots)} 个最终地块片段...")
        lot_count = 0
        for lot in self.lots:
            try:
                layer_name = "GLA" if getattr(lot, 'is_GLA', False) else "LOT"
                model.insert_element(DataElement(
                    lot.geometry, layer=layer_name, start_height=-5, height=5.2))
                lot_count += 1
            except Exception as e:
                lot_id_str = getattr(lot, 'lot_id', getattr(lot, 'gla_id', '未知ID'))
                print(f"  警告: 无法将地块片段 (ID: {lot_id_str}) 添加到模型: {e}")
        print(f"  成功添加 {lot_count}/{len(self.lots)} 个地块片段。")

        # 添加道路
        print(f"  正在向模型添加 {len(self.roads)} 个最终道路片段...")
        road_count = 0
        for road in self.roads:
            try:
                model.insert_element(DataElement(
                    road.geometry, layer="ROAD", start_height=-5, height=5.0)) # 道路略高于地块
                road_count += 1
            except Exception as e:
                road_id_str = getattr(road, 'object_id', '未知ID')
                print(f"  警告: 无法将道路片段 (ID: {road_id_str}) 添加到模型: {e}")
        print(f"  成功添加 {road_count}/{len(self.roads)} 个道路片段。")

        # 添加行政区 (新增)
        print(f"  正在向模型添加 {len(self.admin_boundaries)} 个最终行政区片段...")
        admin_count = 0
        for admin in self.admin_boundaries:
            try:
                # 行政区通常是线划，但也可能是面，这里按面处理，放在比道路稍高
                model.insert_element(DataElement(
                    admin.geometry, layer="Land_Boundary", start_height=-5, height=4.5))
                admin_count += 1
            except Exception as e:
                admin_id_str = getattr(admin, 'object_id', '未知ID') # 假设 ID 属性名
                print(f"  警告: 无法将行政区片段 (ID: {admin_id_str}) 添加到模型: {e}")
        print(f"  成功添加 {admin_count}/{len(self.admin_boundaries)} 个行政区片段。")

        # 添加绿地 (新增)
        print(f"  正在向模型添加 {len(self.green_spaces)} 个最终绿地片段...")
        green_count = 0
        for green in self.green_spaces:
            try:
                # 绿地放在地块同高度或略高
                model.insert_element(DataElement(
                    green.geometry, layer="Landscape", start_height=-5, height=5.5)) # 略高于地块
                green_count += 1
            except Exception as e:
                green_id_str = getattr(green, 'object_id', '未知ID') # 假设 ID 属性名
                print(f"  警告: 无法将绿地片段 (ID: {green_id_str}) 添加到模型: {e}")
        print(f"  成功添加 {green_count}/{len(self.green_spaces)} 个绿地片段。")


        # 添加建筑
        print(f"  正在向模型添加 {len(self.buildings)} 个建筑...")
        building_count = 0
        for b in self.buildings:
            try:
                model.insert_element(DataElement(
                    b.geometry, layer=f"Building",
                    start_height=b.start_height, height=b.height))
                building_count += 1
            except Exception as e:
                 bldg_id_str = getattr(b, 'object_id', '未知ID')
                 print(f"  警告: 无法将建筑 (ID: {bldg_id_str}) 添加到模型: {e}")
        print(f"  成功添加 {building_count}/{len(self.buildings)} 个建筑。")

        print("  正在更新数据模型...")
        model.renew()

        model_path = self.out / "map_model.3dm"
        json_path = self.out / "map_model_json.3dm"
        print(f"  正在将 Rhino 3DM  / json 文件写入到: {model_path}")
        try:
            ARTShapelyDataExchanger.write_rhino_file(str(model_path), model)
            ARTShapelyDataExchanger.write_json_file(str(json_path), model)
            print(f"  导出成功！")
        except Exception as e:
            print(f"  错误: 导出 Rhino / json 文件时出错: {e}")