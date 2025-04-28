import copy
from typing import List, Optional, Union, Tuple

# 明确导入需要的 Shapely 类型和基础类
from shapely.geometry import Polygon, LineString, MultiPolygon, MultiLineString, GeometryCollection, box
from shapely.geometry.base import BaseGeometry, GEOSException
from shapely.validation import make_valid
from shapely.ops import transform # 用于可能的坐标转换 (如果需要)


class RobustMapCropper:
    """
    提供将单个任意 Shapely 几何图形裁剪到指定边界框的功能，
    并返回分解后的有效基础几何图形列表。
    """

    @staticmethod
    def clip_geometry(
        input_geometry: Optional[BaseGeometry],
        clip_box: Polygon,
        filter_threshold: Optional[float] = 1e-9 # 设置阈值过滤小碎片，None 则不过滤
        ) -> List[Union[Polygon, LineString]]:
        """
        将单个输入几何图形裁剪到指定的 clip_box 边界内，并返回有效的基础几何图形列表。

        Args:
            input_geometry: 输入的 Shapely 几何对象 (可以是任何类型)。
            clip_box: 用于裁剪的 Shapely Polygon 对象。
            filter_threshold: 用于过滤微小几何体的阈值 (面积或长度)。
                              设置为 None 则不过滤。默认为 1e-9。

        Returns:
            一个列表，包含裁剪后有效的、非空的 Polygon 和 LineString 对象。
            如果输入无效、裁剪出错或无有效交集，则返回空列表 []。
        """
        # 1. 验证输入
        if input_geometry is None or input_geometry.is_empty:
            print("Warning: Input geometry is None or empty.")
            return []
        if clip_box is None or clip_box.is_empty or not isinstance(clip_box, Polygon):
             print("Error: Clip box is None, empty, or not a Polygon.")
             return []

        # 2. 确保几何体有效性 (尝试修复)
        valid_input_geom = RobustMapCropper._ensure_valid(input_geometry)
        valid_clip_box = RobustMapCropper._ensure_valid(clip_box)

        # 如果修复失败或类型不符
        if valid_input_geom is None:
            print("Warning: Input geometry is invalid and could not be fixed.")
            return []
        if not isinstance(valid_clip_box, Polygon):
             print("Error: Clip box became invalid after attempting to fix.")
             return [] # 必须是 Polygon 才能裁剪

        # 3. 计算交集 (使用 try-except 捕获 GEOS 错误)
        try:
            # 只有当输入几何图形与边界框可能相交时才进行实际计算
            if not valid_input_geom.disjoint(valid_clip_box):
                intersection_result = valid_input_geom.intersection(valid_clip_box)
            else:
                # 如果不相交，则结果为空
                return []
        except GEOSException as e:
            print(f"Error during intersection: {e}. Skipping geometry.")
            return []
        except Exception as e:
            print(f"Unexpected error during intersection: {e}. Skipping geometry.")
            return []

        # 4. 处理并分解交集结果
        final_geometries = RobustMapCropper._flatten_and_validate(intersection_result, filter_threshold)

        return final_geometries

    @staticmethod
    def _ensure_valid(geom: BaseGeometry) -> Optional[BaseGeometry]:
        """尝试确保几何体有效，如果无效则尝试修复。"""
        try:
            if geom.is_valid:
                return geom
            else:
                print(f"Warning: Geometry of type {geom.geom_type} is invalid, attempting to fix...")
                fixed_geom = make_valid(geom)
                # 检查修复后的结果是否有效且非空
                if fixed_geom is not None and not fixed_geom.is_empty and fixed_geom.is_valid:
                     print("  Successfully fixed.")
                     return fixed_geom
                else:
                     print("  Failed to fix or resulted in empty geometry.")
                     return None
        except Exception as e:
            print(f"Error during validation/fixing geometry: {e}")
            return None

    @staticmethod
    def _flatten_and_validate(
        geom: Optional[BaseGeometry],
        threshold: Optional[float]
        ) -> List[Union[Polygon, LineString]]:
        """
        递归地将几何对象分解为基础类型 (Polygon, LineString)，
        验证它们，并过滤掉空或过小的几何体。

        Args:
            geom: 输入的 Shapely 几何对象 (可能是 intersection 的结果)。
            threshold: 过滤阈值 (面积或长度)，None 表示不过滤。

        Returns:
            一个包含有效、非空、过滤后的 Polygon 和 LineString 对象的列表。
        """
        if geom is None or geom.is_empty:
            return []

        valid_parts = []

        # 处理 GeometryCollection 或 Multi* 类型
        if isinstance(geom, GeometryCollection):
            for part in geom.geoms:
                valid_parts.extend(RobustMapCropper._flatten_and_validate(part, threshold))
        elif isinstance(geom, (MultiPolygon, MultiLineString)):
             for part in geom.geoms:
                 # 这里 part 已经是 Polygon 或 LineString，直接处理
                 valid_parts.extend(RobustMapCropper._flatten_and_validate(part, threshold))
        # 处理基础类型 Polygon 和 LineString
        elif isinstance(geom, (Polygon, LineString)):
            # 再次确保有效性，因为 intersection 可能产生轻微无效的结果
            valid_geom = RobustMapCropper._ensure_valid(geom)
            if valid_geom and isinstance(valid_geom, (Polygon, LineString)): # 确保修复后类型正确
                # 过滤微小几何体
                is_too_small = False
                if threshold is not None:
                    if isinstance(valid_geom, Polygon) and valid_geom.area < threshold:
                        is_too_small = True
                    elif isinstance(valid_geom, LineString) and valid_geom.length < threshold:
                         is_too_small = True

                if not is_too_small:
                    valid_parts.append(valid_geom)
            elif valid_geom and isinstance(valid_geom, (GeometryCollection, MultiPolygon, MultiLineString)):
                 # 如果修复后又变成了集合，递归处理
                 valid_parts.extend(RobustMapCropper._flatten_and_validate(valid_geom, threshold))

        # 其他几何类型 (如 Point, MultiPoint) 在此逻辑下会被忽略

        return valid_parts


# --- 示例用法 ---
if __name__ == '__main__':
    # 1. 定义裁剪边界框
    minx, miny, maxx, maxy = 0, 0, 10, 10
    clip_polygon = box(minx, miny, maxx, maxy)
    print(f"Clip Box: {clip_polygon.wkt[:100]}...")

    # 2. 创建一些示例输入几何图形
    poly_inside = Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
    poly_intersect = Polygon([(8, 8), (12, 8), (12, 12), (8, 12)])
    poly_outside = Polygon([(11, 11), (15, 11), (15, 15), (11, 15)])
    line_intersect = LineString([(5, 5), (15, 15)])
    line_outside = LineString([(11, 1), (15, 5)])
    multi_poly_intersect = MultiPolygon([
        Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]), # 完全在内部
        Polygon([(9, 9), (11, 9), (11, 11), (9, 11)]) # 部分相交
    ])
    geom_collection_intersect = GeometryCollection([
        Polygon([(1, 6), (3, 6), (3, 8), (1, 8)]), # 完全在内部
        LineString([(7, 1), (11, 5)]) # 部分相交
    ])
    invalid_poly = Polygon([(0,0), (10,0), (10,10), (5,5), (0,10), (0,0)]) # 自相交，无效

    geometries_to_test = {
        "poly_inside": poly_inside,
        "poly_intersect": poly_intersect,
        "poly_outside": poly_outside,
        "line_intersect": line_intersect,
        "line_outside": line_outside,
        "multi_poly_intersect": multi_poly_intersect,
        "geom_collection_intersect": geom_collection_intersect,
        "invalid_poly_intersect": invalid_poly,
        "empty_geom": Polygon(),
        "none_geom": None
    }

    # 3. 测试裁剪函数
    cropper = RobustMapCropper() # 虽然方法是静态的，但可以实例化

    for name, geom in geometries_to_test.items():
        print(f"\n--- Clipping '{name}' (Type: {type(geom).__name__ if geom else 'None'}) ---")
        clipped_list = cropper.clip_geometry(geom, clip_polygon, filter_threshold=1e-9)

        if clipped_list:
            print(f"  Result ({len(clipped_list)} geometries):")
            for i, clipped_geom in enumerate(clipped_list):
                print(f"    {i+1}: Type={clipped_geom.geom_type}, WKT={clipped_geom.wkt[:60]}...")
        else:
            print("  Result: [] (No valid intersection or input was invalid/empty)")

