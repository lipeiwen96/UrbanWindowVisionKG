# map_cropper.py (Optimized)
import copy
from typing import List, Optional

# 明确导入需要的 Shapely 类型
from shapely.geometry import Polygon, LineString, MultiPolygon, MultiLineString, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid # 用于更可靠地修复无效几何

# 假设这个导入路径是正确的
from map_system.map_structure import MapBaseGeometry


class MapCropper:
    """
    提供将地图几何对象列表高效裁剪到指定边界框的功能。
    - 完全在内部的几何体直接保留。
    - 完全在外部的几何体直接丢弃。
    - 仅对与边界相交的几何体执行裁剪操作。
    """

    @staticmethod
    def clip_to_box(map_geometry_list: List[MapBaseGeometry], clip_box: Polygon) -> List[MapBaseGeometry]:
        """
        将地图几何对象列表高效裁剪到指定的 clip_box 边界内。

        Args:
            map_geometry_list: 包含 MapBaseGeometry 对象的列表。
            clip_box: 用于裁剪的 Shapely Polygon 对象。

        Returns:
            一个新的列表，包含处理后的 MapBaseGeometry 对象。
        """
        clipped_results: List[MapBaseGeometry] = []

        # 预先检查 clip_box 是否有效
        if not clip_box.is_valid:
            # 尝试修复 clip_box，如果失败则无法进行裁剪
            clip_box = make_valid(clip_box)
            if not isinstance(clip_box, Polygon): # make_valid 可能返回 MultiPolygon 等
                # Handle error: Cannot proceed with invalid clip_box
                return [] # Return empty list or raise error

        for map_geo in map_geometry_list:
            geom = map_geo.geometry
            if geom is None or geom.is_empty:
                continue

            # 尝试修复输入几何体
            if not geom.is_valid:
                geom = make_valid(geom)
                # 如果修复后变成空的或非预期类型，则跳过
                if geom.is_empty or not isinstance(geom, (Polygon, LineString, MultiPolygon, MultiLineString, GeometryCollection)):
                     continue
                # 更新 map_geo 中的几何体引用，以便后续操作使用修复后的版本
                # 注意：这里直接修改可能影响原始列表，如果需要保持原始列表不变，应在开始时深拷贝
                map_geo.geometry = geom # Or handle this via deepcopy later

            # --- Optimization Logic ---
            # Case 1: 完全包含在 clip_box 内?
            # 使用 prepared geometry 可以加速 contains/intersects 判断，但对少量判断可能开销更大
            # prep_clip_box = prep(clip_box) # 如果 clip_box 固定且检查次数多，可以考虑
            try:
                if clip_box.contains(geom):
                    # 完全在内部，直接添加深拷贝
                    clipped_results.append(copy.deepcopy(map_geo))
                    continue # 处理下一个几何体
            except Exception:
                # contains 操作也可能因拓扑错误失败
                # Fallback to intersection logic below
                pass # Let it proceed to intersects check

            # Case 2: 与 clip_box 相交? (排除了完全包含的情况)
            try:
                if geom.intersects(clip_box):
                    # 与边界相交，执行裁剪
                    intersection_outcomes = MapCropper.single_geometry_intersection(map_geo, geom, clip_box) # 传入修复后的 geom
                    if intersection_outcomes:
                        clipped_results.extend(intersection_outcomes)
                # else: Case 3: 完全在外部，自动忽略，不添加到 clipped_results
            except Exception:
                # intersects 操作也可能失败
                continue # Skip this geometry if intersects check fails

        return clipped_results

    @staticmethod
    def single_geometry_intersection(
        original_map_geo: MapBaseGeometry,
        valid_geometry: BaseGeometry, # 传入已经验证/修复过的几何体
        clip_box: Polygon
        ) -> Optional[List[MapBaseGeometry]]:
        """
        计算单个几何体与 clip_box 的交集 (假设已经检查过 intersects)。

        Args:
            original_map_geo: 原始的 MapBaseGeometry 对象 (用于深拷贝属性)。
            valid_geometry: 已经验证/修复过的 Shapely 几何体。
            clip_box: 用于裁剪的 Shapely Polygon 对象。

        Returns:
            如果存在有效交集，则返回包含一个或多个裁剪后 MapBaseGeometry 对象的列表。
            否则返回 None。
        """
        try:
            # 直接执行交集计算，因为已确认相交且几何体有效
            outcome = valid_geometry.intersection(clip_box)
        except Exception:
            # Intersection 仍然可能失败
            return None

        # 处理交集结果
        processed_geometries = MapCropper.outcome_process(outcome)

        # 如果处理后得到有效几何列表
        if processed_geometries:
            outcome_map_geo_list = []
            for single_geometry in processed_geometries:
                # 创建原始对象的深拷贝，以保留所有属性
                map_geo_copy = copy.deepcopy(original_map_geo)
                # 将拷贝对象的几何替换为裁剪后的几何
                map_geo_copy.geometry = single_geometry
                outcome_map_geo_list.append(map_geo_copy)
            return outcome_map_geo_list
        else:
            return None

    @staticmethod
    def outcome_process(outcome: BaseGeometry) -> Optional[List[BaseGeometry]]:
        """
        处理 Shapely 操作结果，转换为有效、非空几何对象的列表。
        Args:
            outcome: Shapely 操作返回的几何对象。
        Returns:
            包含有效、非空几何对象的列表，如果结果为空或无效则返回 None。
        """
        if outcome is None or outcome.is_empty:
            return None

        geo_list = []
        # 检查是否是 Multi 类型或 GeometryCollection
        if isinstance(outcome, (MultiPolygon, MultiLineString, GeometryCollection)):
            for single_geometry in outcome.geoms:
                # 确保单个几何非空且有效
                if single_geometry and not single_geometry.is_empty:
                    # 尝试再次确保有效性，make_valid 可能产生非预期类型
                    valid_single = make_valid(single_geometry)
                    if not valid_single.is_empty and isinstance(valid_single, (Polygon, LineString)):
                        # 可选: 过滤小碎片
                        # if isinstance(valid_single, Polygon) and valid_single.area < 1e-9: continue
                        # if isinstance(valid_single, LineString) and valid_single.length < 1e-9: continue
                        geo_list.append(valid_single)
                    elif isinstance(valid_single, (MultiPolygon, MultiLineString, GeometryCollection)):
                        # 如果 make_valid 产生集合，递归处理或展平
                        inner_processed = MapCropper.outcome_process(valid_single)
                        if inner_processed:
                            geo_list.extend(inner_processed)

        # 检查是否是单个 Polygon 或 LineString
        elif isinstance(outcome, (Polygon, LineString)):
            if not outcome.is_empty:
                valid_single = make_valid(outcome) # 确保结果有效
                if not valid_single.is_empty and isinstance(valid_single, (Polygon, LineString)):
                    # 可选: 过滤小碎片
                    # if isinstance(valid_single, Polygon) and valid_single.area < 1e-9: continue
                    # if isinstance(valid_single, LineString) and valid_single.length < 1e-9: continue
                    geo_list.append(valid_single)
                elif isinstance(valid_single, (MultiPolygon, MultiLineString, GeometryCollection)):
                    inner_processed = MapCropper.outcome_process(valid_single)
                    if inner_processed:
                        geo_list.extend(inner_processed)

        return geo_list if geo_list else None
