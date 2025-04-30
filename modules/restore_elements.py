# restore_elements.py
import uuid
from dataclasses import dataclass, field
from typing import AnyStr, List, Dict, Optional, Set
import time
import os
import json
from modules.utils import log_time, load_json_data
from shapely.wkt import loads as wkt_loads
from shapely.geometry.base import BaseGeometry
from shapely.errors import WKTReadingError, ShapelyError


@dataclass(order=True)
class RestoreElements:
    # Shapely 几何比较的容差 (如果需要可以调整)
    GEOMETRY_COMPARISON_TOLERANCE = 1e-9

    # --- 默认高度配置 ---
    # 存储非 Building 类型的默认高度和起始高度
    DEFAULT_HEIGHTS = {
        # 类型名: (start_height, height)
        "Landscape": (-5.0, 6),
        "Base": (-80.0, 75.0),
        "ROAD": (-5, 5),
        "GLA": (-5, 5.5),
        "LOT": (-5, 5.5),
        "Land_Boundary": (-5, 4.2),
        # 可以根据需要添加更多类型及其默认值
    }
    # 未在上面列出的类型的通用默认值
    GENERIC_DEFAULT_START_HEIGHT = 0.0
    GENERIC_DEFAULT_HEIGHT = 1.0

    @staticmethod
    def run(raw_json_path: str, processed_json_path: str, output_json_path: Optional[str] = None) -> Optional[
        List[Dict]]:
        start_time_run = time.perf_counter()
        print(f"--- 开始执行语义恢复 (优化版 v2 - 增强可视性) ---")
        raw_data = load_json_data(raw_json_path)
        processed_data = load_json_data(processed_json_path)

        if not raw_data or not processed_data:
            print("错误: 无法加载输入 JSON 文件。")
            return None

        # 调用核心恢复逻辑
        enriched_elements = RestoreElements.restore_semantics(raw_data, processed_data)

        if enriched_elements is None:  # 检查 restore_semantics 是否失败
            print("错误: 语义恢复过程失败。")
            return None

        # --- 导出结果到 JSON 文件 ---
        if output_json_path:
            print(f"\n步骤 3: 导出富集数据到 JSON 文件: {output_json_path}")
            start_export_time = time.perf_counter()
            try:
                # 确保输出目录存在
                output_dir = os.path.dirname(output_json_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"  已创建输出目录: {output_dir}")

                # 构建输出 JSON 结构
                output_json_data = {
                    # 从原始数据或处理过的数据中获取元数据，如果可用
                    "model_id": raw_data.get("model_id", processed_data.get("model_id", str(uuid.uuid4()))),
                    "model_name": f"{raw_data.get('model_name', 'Enriched Model')} (Restored)",
                    "model_elements": enriched_elements
                }
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_json_data, f, indent=4, ensure_ascii=False)
                log_time(f"成功导出富集数据 JSON。", start_export_time)

            except IOError as e:
                print(f"错误: 无法写入 JSON 文件 {output_json_path}: {e}")
            except Exception as e:
                print(f"错误: 导出 JSON 时发生未知错误: {e}")
        else:
            print("\n步骤 3: 跳过导出 JSON 文件 (未提供输出路径)。")

        log_time(f"--- 完成语义恢复运行 ---", start_time_run)
        return enriched_elements

    @staticmethod
    def parse_geometry(wkt_string: Optional[str], element_id: str = "N/A") -> Optional[BaseGeometry]:
        """
        安全地将 WKT 字符串解析为 Shapely 几何对象。
        (保持不变)
        """
        if not wkt_string: return None
        try:
            geom = wkt_loads(wkt_string)
            if geom.is_empty: return None
            return geom
        except (WKTReadingError, ShapelyError, TypeError, Exception):
            # print(f"警告: 无法解析元素 {element_id} 的 WKT 几何") # 减少打印，只在总结中报告
            return None

    @staticmethod
    def restore_semantics(raw_data: Dict, processed_data: Dict) -> Optional[List[Dict]]:
        """
        核心恢复逻辑：按类型分组，条件化匹配 Building，应用默认高度。
        增加了详细的过程打印。
        """
        start_time = time.perf_counter()
        print("核心恢复逻辑: 开始恢复简化数据的语义信息 (优化版：类型分组 + 条件匹配)...")
        enriched_elements: List[Dict] = []
        # --- 统计计数器 ---
        stats = {
            "found_building": 0,
            "not_found_building": 0,
            "default_applied": 0,
            "missing_key": 0,
            "processed_geom_error": 0,
            "raw_building_parse_error": 0  # 在预处理中累加
        }
        # ---------------------

        if not raw_data or "model_elements" not in raw_data: print("错误: 原始数据无效"); return None
        if not processed_data or "model_elements" not in processed_data: print("错误: 简化数据无效"); return None

        raw_elements = raw_data.get("model_elements", [])
        processed_elements = processed_data.get("model_elements", [])
        total_processed = len(processed_elements)
        total_raw = len(raw_elements)

        # --- 1. 预处理原始数据 ---
        start_preprocess_time = time.perf_counter()
        print(f"\n步骤 1: 预处理 {total_raw} 个原始数据...")
        raw_elements_by_type: Dict[str, List[Dict]] = {}
        raw_elements_missing_id = 0

        for raw_element in raw_elements:
            elem_type = raw_element.get("element_type")
            elem_id = raw_element.get("element_id")
            if not elem_type: continue
            if not elem_id: raw_elements_missing_id += 1
            if elem_type not in raw_elements_by_type: raw_elements_by_type[elem_type] = []

            if elem_type == "Building":
                geom_wkt = raw_element.get("element_geometry")
                geom_obj = RestoreElements.parse_geometry(geom_wkt, elem_id or "raw_building_N/A")
                if geom_obj:
                    raw_elements_by_type[elem_type].append({'geom_obj': geom_obj, 'data': raw_element, 'used': False})
                else:
                    stats["raw_building_parse_error"] += 1
            # else: # 非 Building 类型目前不存储，因为我们只匹配 Building
            #     pass

        valid_raw_buildings_count = len(raw_elements_by_type.get('Building', []))
        log_time(f"完成原始数据预处理。找到 {valid_raw_buildings_count} 个有效 Building 几何可供匹配。",
                 start_preprocess_time)
        if stats["raw_building_parse_error"] > 0:
            print(f"  警告: 预处理期间无法解析 {stats['raw_building_parse_error']} 个原始 Building 几何。")
        if raw_elements_missing_id > 0:
            print(f"  警告: {raw_elements_missing_id} 个原始元素缺少 'element_id'，可能影响 Building 匹配的唯一性。")

        # --- 2. 处理简化数据 ---
        print(f"\n步骤 2: 处理 {total_processed} 个简化元素...")
        last_print_time = start_preprocess_time  # 用于控制打印频率
        print_interval = 2.0  # 每隔多少秒打印一次详细进度

        for i, processed_element in enumerate(processed_elements):
            current_time = time.perf_counter()
            # --- 详细进度打印 (取代之前的按数量打印) ---
            if current_time - last_print_time > print_interval or (i + 1) == total_processed:
                progress_percent = ((i + 1) / total_processed) * 100
                elapsed_loop = current_time - start_preprocess_time  # Loop time
                rate = (i + 1) / elapsed_loop if elapsed_loop > 0 else 0
                print(f"  进度: {i + 1}/{total_processed} ({progress_percent:.1f}%) | "
                      f"速率: {rate:.1f} elem/s | "
                      f"匹配Building: {stats['found_building']} | "
                      f"未匹配Building: {stats['not_found_building']} | "
                      f"应用默认: {stats['default_applied']} | "
                      f"错误/跳过: {stats['missing_key'] + stats['processed_geom_error']}")
                last_print_time = current_time
            # ---------------------------------------------

            processed_element_type = processed_element.get("element_type")
            processed_element_geom_wkt = processed_element.get("element_geometry")
            processed_element_id_debug = processed_element.get("element_id", f"proc_{i}")

            if not processed_element_type or not processed_element_geom_wkt:
                stats["missing_key"] += 1
                continue

            # --- 分类型处理 ---
            if processed_element_type == "Building":
                # print(f"    [处理 Building {processed_element_id_debug}] 尝试匹配几何...") # 过于详细，注释掉
                processed_geom_obj = RestoreElements.parse_geometry(processed_element_geom_wkt,
                                                                    processed_element_id_debug)
                if processed_geom_obj is None:
                    stats["processed_geom_error"] += 1
                    continue

                found_match_for_building = False
                raw_buildings = raw_elements_by_type.get("Building", [])
                for raw_building_entry in raw_buildings:
                    if raw_building_entry['used']: continue

                    raw_geom_obj = raw_building_entry['geom_obj']
                    raw_element_data = raw_building_entry['data']
                    raw_element_id = raw_element_data.get("element_id", "N/A")

                    try:
                        if processed_geom_obj.equals(raw_geom_obj):
                            # print(f"      -> 成功匹配 Raw ID: {raw_element_id}") # 过于详细
                            enriched_elements.append(raw_element_data.copy())
                            stats["found_building"] += 1
                            raw_building_entry['used'] = True
                            found_match_for_building = True
                            break
                    except (ShapelyError, Exception) as cmp_e:
                        print(
                            f"警告: 比较几何时出错 (将跳过此原始 Building): Proc={processed_element_id_debug}, Raw={raw_element_id}: {cmp_e}")
                        # Mark raw entry as unusable or log more details if needed
                        raw_building_entry['used'] = True  # Avoid re-comparing problematic geometry

                if not found_match_for_building:
                    stats["not_found_building"] += 1
                    # print(f"    [处理 Building {processed_element_id_debug}] 未找到几何匹配项。") # 过于详细

            else:
                # --- 处理非 Building 类型 ---
                # print(f"    [处理 {processed_element_type} {processed_element_id_debug}] 应用默认高度...") # 过于详细
                default_height_info = RestoreElements.DEFAULT_HEIGHTS.get(processed_element_type)
                enriched_elem = processed_element.copy()

                if default_height_info:
                    start_h, h = default_height_info
                    enriched_elem["element_start_height"] = start_h
                    enriched_elem["element_height"] = h
                else:
                    enriched_elem["element_start_height"] = RestoreElements.GENERIC_DEFAULT_START_HEIGHT
                    enriched_elem["element_height"] = RestoreElements.GENERIC_DEFAULT_HEIGHT

                enriched_elem.setdefault("element_custom_semantics", {})
                enriched_elem.setdefault("element_id",
                                         f"proc_{processed_element_type.lower()}_{i}")  # Ensure an ID exists

                enriched_elements.append(enriched_elem)
                stats["default_applied"] += 1

        # --- 最终总结 ---
        log_time(f"核心恢复逻辑完成。", start_time)
        print("\n--- 语义恢复总结 ---")
        print(f"  成功匹配 Building:         {stats['found_building']}")
        print(f"  未找到匹配 Building:     {stats['not_found_building']}")
        print(f"  应用默认高度元素:        {stats['default_applied']}")
        print(f"  缺少键跳过元素:          {stats['missing_key']}")
        print(f"  简化几何解析错误:      {stats['processed_geom_error']}")
        print(f"  原始 Building 几何解析错误:{stats['raw_building_parse_error']}")
        print(f"----------------------")
        print(f"  最终生成 {len(enriched_elements)} 个富集元素。")
        print(f"----------------------")

        return enriched_elements