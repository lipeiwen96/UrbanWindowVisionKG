# -*- coding: utf-8 -*-
"""hk_map_processor.py – 统一处理 *建筑 + 地块* 并导出 Rhino 3DM

* 面向 CSDI Hong Kong：Building CSU、Lot / GLA、道路等
* 取消 @staticmethod，全部转实例方法，属性通过 **self.*** 持久；便于外部多次调用或继承
* 新增 **Lot**（或任何地块 GeoJSON）加载 → `self.lots`
* 建模阶段同时写入地块（低平台）与建筑（高体量）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.prepared import prep

matplotlib.use("Agg")

# -- project imports --------------------------------------------------------
from map_system.map_structure import MapBuilding, MapLot  # noqa: E402
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial
from art_public_modules.art_data_exchange.data_exchange import ARTShapelyDataExchanger
from map_system.map_cropper import MapCropper

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
        fig.savefig(out_dir / name, bbox_inches="tight", dpi=150)
        plt.close(fig) # 关闭图表释放内存

    @staticmethod
    def trimmed_hist(s: pd.Series, *, bins: int, q: Tuple[float, float]):
        """计算并返回修剪（按分位数）后的直方图数据"""
        if s.empty:
            print("    警告: 用于直方图的数据序列为空。")
            return np.array([]), np.array([])
        lo, hi = s.quantile(q).tolist(); clipped = s.clip(lo, hi)
        # print(f"    生成直方图: bins={bins}, 分位数范围=({q[0]}, {q[1]}), 裁剪范围=({lo:.2f}, {hi:.2f})") # 可选的详细输出
        return np.histogram(clipped, bins=bins)


class _Json:
    """JSON 文件操作辅助方法"""
    @staticmethod
    def dump(obj: Dict[str, Any], out_dir: Path, name: str) -> None:
        """将字典对象保存为 JSON 文件"""
        filepath = out_dir / name
        print(f"    正在保存 JSON 数据到: {filepath}")
        # 使用 utf-8 编码，确保中文字符正确写入
        with filepath.open("w", encoding="utf-8") as fp:
            json.dump(obj, fp, indent=4, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 主处理类
# ---------------------------------------------------------------------------


class MapProcessor:
    """加载 *建筑* + *地块* GeoJSON，进行分析并导出 3DM 文件。"""
    # ---------- 初始化 ----------
    def __init__(
        self,
        building_geojson: Union[str, Path],
        lot_geojson: Union[str, Path] | None = None,
        gla_geojson: Union[str, Path] | None = None,
        *,
        research_box: Optional[Tuple[float, float, float, float]] = None,
        min_building_area: float = 90,
        min_lot_area: float = 100,
        # 新增：进度提示频率
        progress_interval: int = 10000
    ) -> None:
        """
        初始化地图处理器。

        Args:
            building_geojson: 建筑 GeoJSON 文件路径。
            lot_geojson: LOT 地块 GeoJSON 文件路径 (可选)。
            gla_geojson: GLA 地块 GeoJSON 文件路径 (可选)。
            research_box: 研究范围边界框 (minx, miny, maxx, maxy) (可选)。
            min_building_area: 最小保留建筑面积阈值。
            min_lot_area: 最小保留地块面积阈值 (裁剪后)。
            progress_interval: 处理多少个要素后打印一次进度。
        """
        print("--- 初始化地图处理器 ---")
        self.src_bldg = Path(building_geojson).resolve()
        self.src_lot = Path(lot_geojson).resolve() if lot_geojson else None
        self.gla_geojson = Path(gla_geojson).resolve() if gla_geojson else None
        self.research_box = research_box
        self.min_building_area = min_building_area
        self.min_lot_area = min_lot_area
        self.progress_interval = progress_interval # 存储进度提示频率
        self.buildings: List[MapBuilding] = [] # 存储最终加载的建筑
        self.lots: List[MapLot] = [] # 存储最终加载的地块 (可能被裁剪)

        # 打印初始化参数
        print(f"  建筑 GeoJSON: {self.src_bldg}")
        print(f"  LOT 地块 GeoJSON: {self.src_lot if self.src_lot else '未提供'}")
        print(f"  GLA 地块 GeoJSON: {self.gla_geojson if self.gla_geojson else '未提供'}")
        if self.research_box:
            print(f"  研究范围 (minx, miny, maxx, maxy): {self.research_box}")
            # 创建 research_box_polygon 供后续裁剪和检查使用
            try:
                self.research_box_polygon = box(*self.research_box)
                print(f"  已创建研究范围边界多边形用于裁剪。")
            except Exception as e:
                print(f"  警告: 无法创建研究范围多边形: {e}。裁剪功能将禁用。")
                self.research_box_polygon = None
        else:
            print("  未定义研究范围 (裁剪功能禁用)。")
            self.research_box_polygon = None # 明确设为 None
        print(f"  最小建筑面积阈值: {self.min_building_area} 平方米")
        print(f"  最小地块面积阈值: {self.min_lot_area} 平方米")
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

        if analyse:
            print("\n--- 步骤 3: 分析数据 ---")
            self._analyse(bins=bins, q=q)
            print("--- 数据分析完成 ---")
        else:
            print("\n--- 步骤 3: 跳过数据分析 ---")

        print("\n--- 步骤 4: 导出为 3DM 文件 ---")
        self._export_3dm()
        print("--- 3DM 文件导出完成 ---")
        print("\n=== 地图处理流程执行完毕 ===")

    # ---------- 数据加载方法 ----------
    def _load_buildings(self):
        """加载建筑 GeoJSON 文件，根据条件过滤并存入 self.buildings。"""
        # 建筑加载逻辑: 使用 research_box 进行包含性检查 (contains)，不进行裁剪。
        if not self.src_bldg or not self.src_bldg.is_file():
             print(f"  错误: 建筑文件未找到或未指定 ({self.src_bldg})")
             return

        print(f"  正在读取建筑 GeoJSON: {self.src_bldg}")
        try:
            # 使用 stream=True 或 ijson 库可以处理超大文件，但这里为简化，一次性读入
            gj = json.loads(self.src_bldg.read_text(encoding="utf-8"))
            feats = gj.get("features", [])
            total_features = len(feats)
            print(f"  在建筑文件中找到 {total_features} 个要素。")
            if total_features == 0: return # 如果没有要素，直接返回
        except Exception as e:
            print(f"  错误: 读取或解析建筑 GeoJSON 时出错: {e}")
            return

        # 准备边界检查对象 (包含性检查)
        prep_bound = None
        if self.research_box_polygon: # 使用在 __init__ 中创建的多边形
             try:
                 from shapely.prepared import prep # 仅在此处导入 prep
                 prep_bound = prep(self.research_box_polygon)
                 print(f"  已准备研究范围用于检查建筑 *包含* 情况。")
             except Exception as e:
                 print(f"  警告: 无法准备几何图形用于建筑包含性检查: {e}。边界检查将跳过。")
                 prep_bound = None

        # 初始化计数器
        loaded_building_count = 0
        skipped_filter_count = 0
        skipped_boundary_count = 0
        skipped_error_count = 0

        print(f"  开始处理 {total_features} 个建筑要素...")
        for i, f in enumerate(feats):
            # 打印进度
            current_count = i + 1
            if current_count % self.progress_interval == 0 or current_count == total_features:
                print(f"    已处理 {current_count}/{total_features} 个建筑要素...")

            try:
                b = MapBuilding()
                b.init(row_data=f) # 假设 init 方法能处理初始化中的潜在错误

                # 基本几何形状和属性过滤
                # 注意 getattr 的使用，避免属性不存在时报错
                if (b.geometry is None or b.geometry.is_empty or
                    b.height <= 0.1 or b.area <= self.min_building_area or
                    getattr(b, 'status', None) != "A"):
                    skipped_filter_count += 1
                    continue # 跳过不满足条件的建筑

                # 边界检查 (包含性检查，非裁剪)
                if prep_bound and not prep_bound.contains(b.geometry):
                    skipped_boundary_count += 1
                    continue # 跳过在研究范围之外的建筑

                # 如果所有检查通过，则添加到列表
                self.buildings.append(b)
                loaded_building_count += 1

            except Exception as e:
                # print(f"  处理建筑要素 {current_count} 时出错: {e}。已跳过。") # 可以取消注释以查看详细错误
                skipped_error_count += 1
                continue # 跳过处理出错的要素

        print(f"  建筑要素处理完成。")
        print(f"  总要素数: {total_features}")
        print(f"  成功加载的建筑: {loaded_building_count}")
        print(f"  因过滤条件跳过: {skipped_filter_count}")
        print(f"  因超出边界跳过: {skipped_boundary_count}")
        print(f"  因处理错误跳过: {skipped_error_count}")


    def _load_lots(self):
        """加载 LOT 和 GLA 地块 GeoJSON 文件，使用 MapCropper 进行裁剪（如果定义了 research_box），
           并根据条件过滤，最终存入 self.lots。"""
        # 重置列表和计数器
        self.lots = []
        self.lot_id_list = [] # 追踪已添加地块的 ID，防止裁剪后产生重复
        total_lot_features = 0
        total_gla_features = 0
        added_lot_pieces = 0 # 记录最终添加到列表的地块片段数量
        processed_lot_features = 0 # 记录从 LOT 文件处理的要素数量
        processed_gla_features = 0 # 记录从 GLA 文件处理的要素数量

        # --- 处理 LOT GeoJSON ---
        if self.src_lot and self.src_lot.is_file():
            print(f"\n  正在读取 LOT GeoJSON: {self.src_lot}")
            try:
                gj_lot = json.loads(self.src_lot.read_text(encoding="utf-8"))
                lot_feats = gj_lot.get("features", [])
                total_lot_features = len(lot_feats)
                print(f"  在 LOT 文件中找到 {total_lot_features} 个要素。")
            except Exception as e:
                print(f"  错误: 读取或解析 LOT GeoJSON 时出错: {e}")
                lot_feats = [] # 如果文件无效，则处理空列表

            if total_lot_features > 0:
                print(f"  开始处理 {total_lot_features} 个 LOT 要素...")
                for i, f in enumerate(lot_feats):
                    current_count = i + 1
                    feature_id_str = f"LOT 要素 {current_count}/{total_lot_features}"
                    # 打印进度
                    if current_count % self.progress_interval == 0 or current_count == total_lot_features:
                        print(f"    已处理 {current_count}/{total_lot_features} 个 LOT 要素...")

                    try:
                        lot = MapLot()
                        lot.init(f) # 从要素数据初始化
                        processed_lot_features += 1

                        # 在裁剪前进行基本的有效性检查
                        if lot.geometry is None or lot.geometry.is_empty:
                            # print(f"  跳过 {feature_id_str}: 初始几何图形为空或无效。")
                            continue
                        lot_identifier = getattr(lot, 'lot_id', None)
                        if lot_identifier is None:
                            # print(f"  跳过 {feature_id_str}: 缺少 'lot_id' 属性。")
                            continue

                        # 决定需要处理的地块片段列表 (裁剪或原始)
                        lots_to_process: List[MapLot] = []
                        if self.research_box_polygon:
                            # --- 裁剪路径 ---
                            # print(f"  正在裁剪 {feature_id_str} (ID: {lot_identifier})...") # 可选的详细输出
                            # 将单个地块对象放入列表传递给裁剪器
                            clipped_lots = MapCropper.clip_to_box([lot], self.research_box_polygon)
                            if clipped_lots:
                                lots_to_process.extend(clipped_lots)
                            # else:
                            # print(f"  {feature_id_str} 与研究范围无有效交集。")
                        else:
                            # --- 非裁剪路径 ---
                            # 直接处理原始地块
                            lots_to_process.append(lot)

                        # 处理每个产生的地块片段 (原始的或裁剪后的)
                        for processed_lot in lots_to_process:
                            # 再次检查标识符，以防万一
                            current_id = getattr(processed_lot, 'lot_id', None)
                            if current_id is None: continue

                            # 对可能被裁剪的片段应用过滤器
                            if current_id in self.lot_id_list:
                                # print(f"  跳过来自 {feature_id_str} 的片段 (ID: {current_id}): 重复 ID。")
                                continue
                            # 检查面积是否满足最小阈值
                            if processed_lot.geometry.area < self.min_lot_area:
                                # print(f"  跳过来自 {feature_id_str} 的片段 (ID: {current_id}): 面积 {processed_lot.geometry.area:.2f} < {self.min_lot_area}。")
                                continue

                            # 如果所有检查通过，添加这个片段
                            self.lots.append(processed_lot)
                            self.lot_id_list.append(current_id) # 记录 ID 防止重复
                            added_lot_pieces += 1

                    except Exception as e:
                        print(f"  处理 {feature_id_str} 时出错: {e}。已跳过。")
                        continue # 跳过当前要素，处理下一个
                print(f"  LOT 要素处理完成。")

        elif self.src_lot:
            print(f"  警告: 指定的 LOT 文件未找到: {self.src_lot}")
        else:
            print("  未指定 LOT 文件，跳过 LOT 加载。")

        # --- 处理 GLA GeoJSON ---
        if self.gla_geojson and self.gla_geojson.is_file():
            print(f"\n  正在读取 GLA GeoJSON: {self.gla_geojson}")
            try:
                gj_gla = json.loads(self.gla_geojson.read_text(encoding="utf-8"))
                gla_feats = gj_gla.get("features", [])
                total_gla_features = len(gla_feats)
                print(f"  在 GLA 文件中找到 {total_gla_features} 个要素。")
            except Exception as e:
                print(f"  错误: 读取或解析 GLA GeoJSON 时出错: {e}")
                gla_feats = [] # 如果文件无效，则处理空列表

            if total_gla_features > 0:
                print(f"  开始处理 {total_gla_features} 个 GLA 要素...")
                for i, f in enumerate(gla_feats):
                    current_count = i + 1
                    feature_id_str = f"GLA 要素 {current_count}/{total_gla_features}"
                    # 打印进度
                    if current_count % self.progress_interval == 0 or current_count == total_gla_features:
                        print(f"    已处理 {current_count}/{total_gla_features} 个 GLA 要素...")

                    try:
                        lot = MapLot() # 仍然使用 MapLot 类
                        lot.init(f)    # 从 GLA 要素数据初始化
                        processed_gla_features += 1

                        # 基本有效性检查
                        if lot.geometry is None or lot.geometry.is_empty:
                            # print(f"  跳过 {feature_id_str}: 初始几何图形为空或无效。")
                            continue
                        # 确定唯一标识符 (优先使用 gla_id，否则使用 lot_id)
                        gla_identifier = getattr(lot, 'gla_id', getattr(lot, 'lot_id', None))
                        if gla_identifier is None:
                            # print(f"  跳过 {feature_id_str}: 缺少唯一标识符 ('gla_id' 或 'lot_id')。")
                            continue

                        # 决定需要处理的地块片段列表 (裁剪或原始)
                        lots_to_process: List[MapLot] = []
                        if self.research_box_polygon:
                            # --- 裁剪路径 ---
                            # print(f"  正在裁剪 {feature_id_str} (ID: {gla_identifier})...") # 可选的详细输出
                            clipped_lots = MapCropper.clip_to_box([lot], self.research_box_polygon)
                            if clipped_lots:
                                lots_to_process.extend(clipped_lots)
                            # else:
                            #    print(f"  {feature_id_str} 与研究范围无有效交集。")
                        else:
                            # --- 非裁剪路径 ---
                            lots_to_process.append(lot)

                        # 处理每个产生的地块片段
                        for processed_lot in lots_to_process:
                            # 再次获取标识符
                            current_id = getattr(processed_lot, 'gla_id', getattr(processed_lot, 'lot_id', None))
                            if current_id is None: continue

                            # 应用过滤器
                            if current_id in self.lot_id_list: # 检查是否与已添加的 LOT 或 GLA 重复
                                # print(f"  跳过来自 {feature_id_str} 的片段 (ID: {current_id}): 重复 ID。")
                                continue
                            if processed_lot.geometry.area < self.min_lot_area:
                                # print(f"  跳过来自 {feature_id_str} 的片段 (ID: {current_id}): 面积 {processed_lot.geometry.area:.2f} < {self.min_lot_area}。")
                                continue

                            # 如果所有检查通过，添加这个片段
                            # 可以在这里尝试设置 is_GLA 标志，如果 MapLot.init 没有做的话
                            # setattr(processed_lot, 'is_GLA', True)
                            self.lots.append(processed_lot)
                            self.lot_id_list.append(current_id) # 添加到主 ID 列表
                            added_lot_pieces += 1

                    except Exception as e:
                        print(f"  处理 {feature_id_str} 时出错: {e}。已跳过。")
                        continue # 跳过当前要素，处理下一个
                print(f"  GLA 要素处理完成。")

        elif self.gla_geojson:
            print(f"  警告: 指定的 GLA 文件未找到: {self.gla_geojson}")
        else:
            print("  未指定 GLA 文件，跳过 GLA 加载。")

        # --- 地块加载总结 ---
        print("\n  地块加载与裁剪总结:")
        print(f"  处理的 LOT 要素总数: {processed_lot_features}/{total_lot_features}")
        print(f"  处理的 GLA 要素总数: {processed_gla_features}/{total_gla_features}")
        print(f"  最终添加的地块片段总数 (裁剪和过滤后): {added_lot_pieces}")
        print(f"  添加的唯一地块 ID 总数: {len(self.lot_id_list)}")


    # ---------- 数据分析方法 ----------
    def _analyse(self, *, bins: int, q: Tuple[float, float]):
        """对加载的建筑数据进行统计分析并生成图表。"""
        # 分析作用于 self.buildings 列表 (这里未包含地块分析)
        print("  开始对加载的建筑数据进行分析...")
        if not self.buildings:
            print("  没有加载任何建筑数据，跳过分析。")
            return

        print(f"  正在分析 {len(self.buildings)} 个建筑...")
        try:
            # 从建筑对象列表创建 DataFrame
            df = pd.DataFrame([
                dict(height=max(b.height, 0), area=max(b.area, 0), start=max(b.start_height, 0),
                     storeys=getattr(b, "num_above_ground_storeys", 0), # 使用 getattr 获取可选属性
                     cat=getattr(b, 'category', '未知'), # 获取分类，提供默认值
                     typ=getattr(b, 'building_structure_type', '未知')) # 获取结构类型
                for b in self.buildings])
            # print(f"  已创建 DataFrame，形状: {df.shape}") # 可选输出
        except Exception as e:
             print(f"  错误: 创建用于分析的 DataFrame 时出错: {e}")
             return

        num_sum, hists = {}, {} # 存储数值总结和直方图数据
        # 对指定列进行分析
        for col in ["height", "area", "start"]:
             # print(f"  正在分析列: {col}") # 可选输出
             if col not in df.columns: continue # 跳过不存在的列
             s = df[col] # 获取数据序列
             if s.empty: continue # 跳过空序列

             # 计算描述性统计量
             desc = s.describe(percentiles=[.25, .5, .75])
             num_sum[col] = {k: round(float(desc[k]), 2) for k in
                             ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]}
             # print(f"    {col} 的数值总结: {num_sum[col]}") # 可选输出

             # 生成并保存直方图
             try:
                 cnt, edge = _PlotUtils.trimmed_hist(s, bins=bins, q=q) # 计算直方图数据
                 if edge.size > 0 and cnt.size > 0 : # 确保有数据
                     hists[col] = {"bins": edge.round(2).tolist(), "counts": cnt.tolist()}
                     fig, ax = plt.subplots(); # 创建图表
                     ax.hist(s, bins=edge, edgecolor="black"); # 绘制直方图
                     ax.set_title(f"{col} 分布 (分位数裁剪 {q})") # 设置标题
                     _PlotUtils.save(fig, self.out, f"hist_{col}.png") # 保存图表
                 # else: print(f"    跳过为 {col} 生成直方图 (无有效数据)。") # 可选输出
             except Exception as e:
                 print(f"    错误: 为 {col} 生成直方图时出错: {e}")

        # 将统计结果保存到 JSON 文件
        stats_data = {"分析的建筑数量":len(df),"数值总结":num_sum,"直方图":hists}
        _Json.dump(stats_data, self.out, "analysis_stats.json")
        print("  分析结果已保存到 analysis_stats.json")


    # ---------- 导出方法 ----------
    def _export_3dm(self):
        """将加载和处理后的建筑及地块数据导出为 Rhino 3DM 文件。"""
        # 导出作用于最终的 self.buildings 和 self.lots 列表
        print(f"  准备用于 Rhino 导出的数据模型...")
        model = DataModel(name="地图模型") # 创建数据模型实例

        # 添加研究边界底座 (如果存在)
        if self.research_box_polygon: # 检查多边形对象是否存在
            try:
                model.insert_element(DataElement(
                    self.research_box_polygon, # 使用多边形对象
                    layer="研究范围边界", # 图层名称
                    material=DataElementMaterial(color="0xe5ebf1"), # 材质颜色
                    start_height=-10, # 起始高度
                    height=10         # 高度
                 ))
                print(f"  已添加研究范围边界元素。")
            except Exception as e:
                print(f"  警告: 无法添加研究范围边界元素: {e}")

        # 添加地块 (使用 self.lots, 包含裁剪后的结果)
        print(f"  正在向模型添加 {len(self.lots)} 个最终地块片段...")
        lot_count = 0
        for lot in self.lots: # 遍历最终的地块列表
            try:
                # 尝试根据属性判断原始来源，设置不同图层
                # 假设 MapLot 对象在 init 或裁剪过程中保留了来源信息 (例如 is_GLA 标志)
                # 如果没有这个标志，可能需要根据 ID 格式或其他属性判断，或统一放入一个图层
                layer_name = "地块_GLA" if getattr(lot, 'is_GLA', False) else "地块_LOT"
                model.insert_element(DataElement(
                    lot.geometry, # 使用最终的 (可能被裁剪的) 几何图形
                    layer=layer_name,
                    material=DataElementMaterial(color="0xc0c0c0"), # 地块材质
                    start_height=0,  # 地块起始高度
                    height=0.2       # 地块厚度
                ))
                lot_count += 1
            except Exception as e:
                # 获取地块 ID 用于错误报告
                lot_id_str = getattr(lot, 'lot_id', getattr(lot, 'gla_id', '未知ID'))
                print(f"  警告: 无法将地块片段 (ID: {lot_id_str}) 添加到模型: {e}")
        print(f"  成功添加 {lot_count}/{len(self.lots)} 个地块片段。")

        # 添加建筑 (使用 self.buildings, 这里未经裁剪)
        print(f"  正在向模型添加 {len(self.buildings)} 个建筑...")
        building_count = 0
        for b in self.buildings:
            try:
                model.insert_element(DataElement(
                    b.geometry, # 建筑几何图形
                    layer=f"建筑_分类{getattr(b, 'category', '未知')}", # 根据分类设置图层
                    start_height=b.start_height, # 建筑起始高度
                    height=b.height              # 建筑高度
                ))
                building_count += 1
            except Exception as e:
                 bldg_id_str = getattr(b, 'object_id', '未知ID') # 获取建筑 ID
                 print(f"  警告: 无法将建筑 (ID: {bldg_id_str}) 添加到模型: {e}")
        print(f"  成功添加 {building_count}/{len(self.buildings)} 个建筑。")

        print("  正在更新数据模型...")
        model.renew() # 更新模型的边界等信息

        # 定义输出文件路径
        path = self.out / "map_model.3dm"
        print(f"  正在将 Rhino 3DM 文件写入到: {path}")
        try:
            # 调用导出函数写入文件
            ARTShapelyDataExchanger.write_rhino_file(str(path), model)
            print(f"  导出成功！")
        except Exception as e:
            print(f"  错误: 导出 Rhino 文件时出错: {e}")