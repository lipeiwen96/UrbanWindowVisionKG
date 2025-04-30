# ```python
# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import math
import random
import json
import warnings # 用于处理潜在的 Shapely/Pyproj 警告

import numpy as np
from PIL import Image
from numba import njit, prange, cuda
from shapely.geometry import Polygon, MultiPolygon # 用于解析 WKT
from shapely.wkt import loads as wkt_loads # 用于加载 WKT 字符串
from shapely.errors import WKTReadingError
import pyproj # 用于坐标转换

# --- GPU 可用性检查 (来自你的 demo) ---
_GPU_AVAILABLE = False
try:
    if cuda.is_available():
        try:
            cuda.detect()
            device = cuda.get_current_device()
            print(f"CUDA 可用: True")
            print(f"使用 GPU: {device.name.decode()}")
            _GPU_AVAILABLE = True
        except Exception as e_detect:
            print(f"检测 CUDA 设备或获取名称时出错: {e_detect}")
            _GPU_AVAILABLE = True # 尝试使用
    else:
        print("CUDA 不可用: 未检测到兼容设备或驱动程序。")
except ImportError:
    print("Numba CUDA 扩展未安装或导入失败。")
except Exception as e:
    print(f"CUDA 初始化期间发生未知错误: {e}")

# --- 常量定义 (大部分来自你的 demo, 稍作调整) ---
GEOMETRY_DTYPE = np.float32
INDEX_DTYPE = np.int32
COLOR_DTYPE = np.uint8
LABEL_DTYPE = np.int32
DEPTH_DTYPE = np.float32
POINT_DTYPE = np.float32
BVH_NODE_DTYPE = np.float32
INF = GEOMETRY_DTYPE(np.inf)
INTERSECTION_EPSILON = GEOMETRY_DTYPE(1e-5)
SHADOW_BIAS = GEOMETRY_DTYPE(1e-3) # 保持增大的 bias

# --- 光照参数 (保持不变) ---
SUN_DIRECTION = np.array([0.6, 0.8, -0.4], dtype=GEOMETRY_DTYPE)
SUN_DIRECTION /= np.linalg.norm(SUN_DIRECTION)
SUN_INTENSITY = GEOMETRY_DTYPE(1.0)
AMBIENT_LIGHT = GEOMETRY_DTYPE(0.2)
SPECULAR_COLOR = np.array([1.0, 1.0, 1.0], dtype=GEOMETRY_DTYPE)
SPECULAR_EXPONENT = GEOMETRY_DTYPE(32.0)
Ks = GEOMETRY_DTYPE(0.4)

# --- BVH 参数 (保持不变) ---
BVH_MAX_LEAF_SIZE = 4
BVH_NODE_FIELDS = 8

# --- 语义标签和颜色 (与 demo 保持一致, 添加新类型) ---
# 注意：这些标签需要映射到你的 JSON 数据中的 element_type
LABEL_MAP = {
    0: "ground/landscape", # 对应 Landscape, Base, LOT?
    1: "building",         # 对应 Building
    2: "road",             # 对应 ROAD
    3: "water",            # 如果有水体数据的话
    4: "skyscraper",       # 可以用于特别高的建筑
    5: "tree",             # 对应生成的树木
    6: "mountain",         # 对应 OBJ 地形
    7: "sky"               # 天空标签
}
SKY_LABEL = 7
DEFAULT_SKY_COLOR = np.array([135, 206, 235], dtype=COLOR_DTYPE)

# 全局颜色映射 (根据需要调整)
C = {
    0: np.array([80, 160, 80], dtype=COLOR_DTYPE),   # ground/landscape (绿色)
    1: np.array([200, 200, 200], dtype=COLOR_DTYPE), # building (灰色)
    2: np.array([100, 100, 100], dtype=COLOR_DTYPE), # road (深灰)
    3: np.array([90, 140, 210], dtype=COLOR_DTYPE),  # water (蓝色)
    4: np.array([210, 80, 80], dtype=COLOR_DTYPE),   # skyscraper (红色) - 用于区分
    5: np.array([0, 140, 0], dtype=COLOR_DTYPE),     # tree (深绿)
    6: np.array([120, 120, 120], dtype=COLOR_DTYPE), # mountain (灰色)
    SKY_LABEL: DEFAULT_SKY_COLOR                     # sky (淡蓝)
}
PALETTE_BACKGROUND_COLOR = (0, 0, 0)

# --- 文件路径 ---
# 假设脚本位于项目根目录或能够找到 'library' 子目录的位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.join(BASE_DIR, "library")
# 检查 library 目录是否存在，如果不存在，则尝试上一级目录
if not os.path.exists(LIBRARY_DIR):
    LIBRARY_DIR = os.path.join(os.path.dirname(BASE_DIR), "library")
    if not os.path.exists(LIBRARY_DIR):
        print(f"错误：无法找到 'library' 目录。请确保脚本相对于 'library' 的位置正确。")
        # 可以选择退出或设置默认路径
        # exit() # 或者设置默认路径

HK_MAP_DIR = os.path.join(LIBRARY_DIR, "HK_map", "processed", "project")
TREE_DIR = os.path.join(LIBRARY_DIR, "HK_map", "row_map", "greening")

RAW_MODEL_JSON_PATH = os.path.join(HK_MAP_DIR, "map_model.json")
SIMPLIFIED_MODEL_JSON_PATH = os.path.join(HK_MAP_DIR, "project_simplified_only_crv.json")
NORTH_MOUNTAINS_OBJ_PATH = os.path.join(HK_MAP_DIR, "北侧地形.obj")
SOUTH_MOUNTAINS_OBJ_PATH = os.path.join(HK_MAP_DIR, "南侧地形.obj")
TREE_POINT_PATH = os.path.join(TREE_DIR, "tree20250203_converted.json")
print(f"Base directory: {BASE_DIR}")
print(f"Library directory: {LIBRARY_DIR}")
print(f"Raw model path: {RAW_MODEL_JSON_PATH}")

# --- 坐标系统 ---
# 源坐标系: WGS84 (经纬度) - 树木数据
CRS_WGS84 = "EPSG:4326"
# 目标坐标系: 香港1980方格网 (米) - 建筑/地形数据
CRS_HK1980 = "EPSG:2326"
# 创建转换器
try:
    # 确认 HK1980 的轴顺序。EPSG:2326 通常是 Northing, Easting (Y, X)
    # 但 pyproj 的 always_xy=True 会强制输出为 (X, Y) 顺序
    transformer_latlon_to_hk = pyproj.Transformer.from_crs(CRS_WGS84, CRS_HK1980, always_xy=True) # lon, lat -> x, y
    print(f"坐标转换器 WGS84 ({CRS_WGS84}) -> HK1980 ({CRS_HK1980}) 创建成功。")
except pyproj.exceptions.CRSError as e:
    print(f"错误: 初始化 pyproj 转换器失败: {e}")
    print("请确保 pyproj 已安装且 CRS 定义有效。")
    transformer_latlon_to_hk = None

# --- 工具函数 (来自 demo) ---
def _now() -> float:
    return time.perf_counter()

def log_step(title: str, t0: float) -> None:
    print(f"    {title} 耗时 {_now() - t0:.3f}s")

RAND = np.random.default_rng(seed=0)

# --- [新] 数据加载函数 ---
def load_json_file(filepath):
    """加载 JSON 文件。"""
    print(f"    正在加载 JSON 文件: {filepath}")
    if not os.path.exists(filepath):
        print(f"    错误: 文件未找到 {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        num_features = len(data.get('features', []))
        num_elements = len(data.get('model_elements', []))
        if num_features > 0:
             print(f"    成功加载 {num_features} 个要素")
        if num_elements > 0:
             print(f"    成功加载 {num_elements} 个模型元素")
        if num_features == 0 and num_elements == 0:
             print(f"    警告: JSON 文件似乎不包含 'features' 或 'model_elements'。")
        return data
    except json.JSONDecodeError as e:
        print(f"    错误: 解析 JSON 文件失败 {filepath}: {e}")
        return None
    except Exception as e:
        print(f"    加载 JSON 文件时发生未知错误 {filepath}: {e}")
        return None

def load_obj_file_simple(filepath):
    """简单的 OBJ 文件加载器 (仅顶点和面)。"""
    print(f"    正在加载 OBJ 文件: {filepath}")
    if not os.path.exists(filepath):
        print(f"    错误: 文件未找到 {filepath}")
        return [], []
    vertices = []
    faces = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('v '):
                    try:
                        parts = line.split()
                        # 假设顶点是 x, y, z
                        vertices.append([GEOMETRY_DTYPE(p) for p in parts[1:4]])
                    except (ValueError, IndexError):
                        print(f"    警告: 无法解析 OBJ 中的顶点行: {line.strip()}")
                elif line.startswith('f '):
                    try:
                        parts = line.split()
                        # 处理面定义 (例如 'f v1 v2 v3' 或 'f v1/vt1/vn1 ...')
                        face_indices = []
                        for part in parts[1:]:
                            # 只取顶点索引 (第一个数字)
                            v_index = int(part.split('/')[0])
                            # OBJ 索引是 1-based, 转换为 0-based
                            face_indices.append(v_index - 1)
                        # 假设是三角形，如果不是则需要三角化
                        if len(face_indices) == 3:
                            faces.append(tuple(face_indices))
                        elif len(face_indices) > 3:
                            # 简单的扇形三角化 (仅适用于凸多边形)
                            for i in range(1, len(face_indices) - 1):
                                faces.append((face_indices[0], face_indices[i], face_indices[i+1]))
                            # print(f"    警告: OBJ 中的面不是三角形，已尝试扇形三角化: {line.strip()}")
                        # else:
                             # print(f"    警告: OBJ 中的面少于 3 个顶点: {line.strip()}") # 太过频繁
                             # pass
                    except (ValueError, IndexError):
                        print(f"    警告: 无法解析 OBJ 中的面行: {line.strip()}")

        print(f"    成功加载 {len(vertices)} 个顶点和 {len(faces)} 个面。")
        return np.array(vertices, dtype=GEOMETRY_DTYPE), faces
    except Exception as e:
        print(f"    加载 OBJ 文件时发生未知错误 {filepath}: {e}")
        return [], []

# --- [新] 几何处理函数 ---
def parse_wkt_polygon(wkt_string):
    """使用 Shapely 解析 WKT POLYGON 字符串。"""
    if not wkt_string or not isinstance(wkt_string, str):
        return None
    try:
        # 移除可能存在的前导/尾随空格
        wkt_string = wkt_string.strip()
        # 检查是否以 POLYGON 开头（不区分大小写）
        if not wkt_string.upper().startswith("POLYGON"):
             # print(f"    警告: WKT 字符串看起来不像 Polygon: {wkt_string[:50]}...")
             return None

        geom = wkt_loads(wkt_string)
        if isinstance(geom, Polygon):
            if geom.is_empty: return None
            exterior_coords = list(geom.exterior.coords)
            if len(exterior_coords) < 4: return None # 需要至少3个不同点+闭合点
            if exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            if len(exterior_coords) < 3: return None # 至少需要3个顶点
            return np.array(exterior_coords, dtype=GEOMETRY_DTYPE)
        elif isinstance(geom, MultiPolygon):
            if geom.is_empty: return None
            # print(f"    警告: WKT 是 MultiPolygon，仅使用第一个 Polygon。")
            first_poly = geom.geoms[0]
            if first_poly.is_empty: return None
            exterior_coords = list(first_poly.exterior.coords)
            if len(exterior_coords) < 4: return None
            if exterior_coords[0] == exterior_coords[-1]:
                exterior_coords = exterior_coords[:-1]
            if len(exterior_coords) < 3: return None
            return np.array(exterior_coords, dtype=GEOMETRY_DTYPE)
        else:
            # print(f"    警告: WKT 几何不是 Polygon 或 MultiPolygon: {type(geom)}")
            return None
    except WKTReadingError as e:
        # print(f"    错误: 解析 WKT 字符串失败: {e} - String: {wkt_string[:100]}...")
        return None
    except Exception as e:
        # print(f"    解析 WKT 时发生未知错误: {e} - String: {wkt_string[:100]}...")
        return None

def triangulate_polygon_simple(poly_coords_2d):
    """简单的扇形三角化 (仅适用于凸多边形)。"""
    n = len(poly_coords_2d)
    if n < 3:
        return [] # 不能三角化少于 3 个顶点
    triangles = []
    v0_idx = 0
    for i in range(1, n - 1):
        v1_idx = i
        v2_idx = i + 1
        triangles.append((v0_idx, v1_idx, v2_idx))
    return triangles

def extrude_polygon(poly_coords_2d, start_height, height):
    """将 2D 多边形拉伸成 3D 网格 (顶点和面)。"""
    num_verts_base = len(poly_coords_2d)
    if num_verts_base < 3:
        return np.empty((0, 3), dtype=GEOMETRY_DTYPE), [] # 无效多边形

    verts = np.zeros((num_verts_base * 2, 3), dtype=GEOMETRY_DTYPE)
    faces = []

    # 创建底部和顶部顶点
    verts[:num_verts_base, :2] = poly_coords_2d # 底部 XY
    verts[:num_verts_base, 2] = start_height   # 底部 Z
    verts[num_verts_base:, :2] = poly_coords_2d # 顶部 XY
    verts[num_verts_base:, 2] = start_height + height # 顶部 Z

    # 创建侧面 (四边形 -> 三角形)
    for i in range(num_verts_base):
        i0_bottom = i
        i1_bottom = (i + 1) % num_verts_base # 环绕
        i0_top = i + num_verts_base
        i1_top = i1_bottom + num_verts_base

        # 侧面四边形的两个三角形
        # 确保顶点顺序一致（例如，逆时针从外部看）
        faces.append((i0_bottom, i1_bottom, i1_top)) # Tri 1
        faces.append((i0_bottom, i1_top, i0_top))    # Tri 2

    # 创建底部和顶部面 (需要三角化)
    # TODO: 替换为更健壮的三角化方法！
    base_tris_indices = triangulate_polygon_simple(poly_coords_2d)

    # 底部面 (顶点顺序需要反转以使法线朝下)
    for tri_indices in base_tris_indices:
        # 原始索引是 0 到 num_verts_base-1
        # 反转顺序 (v0, v2, v1)
        faces.append((tri_indices[0], tri_indices[2], tri_indices[1]))

    # 顶部面 (保持原始顺序以使法线朝上)
    for tri_indices in base_tris_indices:
        # 索引偏移 num_verts_base
        faces.append((tri_indices[0] + num_verts_base,
                      tri_indices[1] + num_verts_base,
                      tri_indices[2] + num_verts_base))

    return verts, faces

# 使用 demo 中的 create_cone_tree (稍作修改以匹配参数)
def create_cone_tree_mesh(base_center_xy, base_z, height, radius, segments=10):
    """创建简化的树 (圆柱树干 + 圆锥树冠)。"""
    cx, cy = base_center_xy # 假设输入是 HK1980 的 x, y
    cz = base_z # 假设 z 是地面高度
    trunk_height = height * 0.3
    trunk_radius = radius * 0.15
    crown_height = height * 0.7
    crown_radius = radius
    if trunk_height <= 0 or crown_height <= 0 or trunk_radius <=0 or crown_radius <=0:
        # print(f"    警告: 无效的树木尺寸，跳过。H={height}, R={radius}")
        return np.empty((0, 3), dtype=GEOMETRY_DTYPE), []


    # --- 树干 (圆柱) ---
    trunk_center_z = cz + trunk_height / 2.0
    trunk_segments = max(6, segments // 2)
    trunk_verts = []
    # 底部和顶部顶点
    for i in range(trunk_segments):
        angle = 2.0 * np.pi * i / trunk_segments
        x = cx + trunk_radius * np.cos(angle)
        y = cy + trunk_radius * np.sin(angle) # 在 XY 平面创建圆
        trunk_verts.append((x, y, cz)) # 底部顶点
        trunk_verts.append((x, y, cz + trunk_height)) # 顶部顶点
    trunk_verts = np.asarray(trunk_verts, dtype=GEOMETRY_DTYPE)
    num_trunk_verts = len(trunk_verts)

    trunk_faces = []
    # 侧面
    for i in range(trunk_segments):
        i0_bottom = 2 * i
        i1_bottom = (i0_bottom + 2) % (2 * trunk_segments) # 环绕
        i0_top = i0_bottom + 1
        i1_top = i1_bottom + 1
        trunk_faces.append((i0_bottom, i1_bottom, i1_top))
        trunk_faces.append((i0_bottom, i1_top, i0_top))

    # --- 树冠 (圆锥) ---
    crown_base_z = cz + trunk_height
    apex_z = crown_base_z + crown_height
    apex_vertex = np.array([cx, cy, apex_z], dtype=GEOMETRY_DTYPE)

    cone_base_verts = []
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        x = cx + crown_radius * np.cos(angle)
        y = cy + crown_radius * np.sin(angle) # XY 平面
        cone_base_verts.append((x, y, crown_base_z))
    cone_base_verts = np.asarray(cone_base_verts, dtype=GEOMETRY_DTYPE)

    cone_verts = np.vstack((apex_vertex.reshape(1, 3), cone_base_verts))
    num_cone_verts = len(cone_verts)

    cone_faces = []
    apex_idx_local = 0
    for i in range(segments):
        base_idx_local_1 = i + 1
        base_idx_local_2 = (i + 1) % segments + 1
        # 确保顶点顺序使法线朝外
        cone_faces.append((apex_idx_local, base_idx_local_2, base_idx_local_1))

    # --- 合并 ---
    all_verts = np.vstack((trunk_verts, cone_verts))
    # 调整圆锥面索引
    adjusted_cone_faces = [(f[0] + num_trunk_verts, f[1] + num_trunk_verts, f[2] + num_trunk_verts)
                           for f in cone_faces]
    all_faces = trunk_faces + adjusted_cone_faces

    return all_verts, all_faces

# --- [修改] 场景构建函数 ---
def build_hk_scene():
    """构建香港场景几何体，分配标签和颜色。"""
    tris_vertices_tuples = [] # (v0, v1, v2) 存储三角形顶点坐标元组
    tris_labels = []          # 存储每个三角形的语义标签
    tris_colors = []          # 存储每个三角形的基础颜色
    element_ids = []          # [可选] 存储每个三角形所属的原始元素 ID
    building_data = {}        # 存储建筑信息以供视点生成 {element_id: data}

    # --- 1. 加载并合并模型数据 ---
    print("    加载原始和简化模型 JSON...")
    raw_model_data = load_json_file(RAW_MODEL_JSON_PATH)
    simplified_model_data = load_json_file(SIMPLIFIED_MODEL_JSON_PATH)

    if not raw_model_data or not raw_model_data.get('model_elements'):
        print("    错误: 原始模型文件无效或缺少 'model_elements'。")
        return [], [], [], [], [], [], [], {} # 返回空数据
    if not simplified_model_data or not simplified_model_data.get('model_elements'):
        print("    错误: 简化模型文件无效或缺少 'model_elements'。")
        # 也许可以只使用原始数据？但这违背了简化的目的。
        return [], [], [], [], [], [], [], {} # 返回空数据

    # 创建原始数据的查找字典
    raw_data_lookup = {elem['element_id']: elem for elem in raw_model_data.get('model_elements', []) if 'element_id' in elem}
    print(f"    创建了包含 {len(raw_data_lookup)} 个元素的原始数据查找表。")

    # --- 2. 处理简化模型中的元素 ---
    print("    处理简化模型元素并生成网格...")
    processed_ids = set()
    skipped_types = set()
    skipped_invalid_geom = 0
    elements_to_process = simplified_model_data.get('model_elements', [])
    print(f"    将处理 {len(elements_to_process)} 个简化模型元素。")

    for i, simplified_elem in enumerate(elements_to_process):
        if i % 5000 == 0 and i > 0: # 每处理 5000 个元素打印一次进度
             print(f"      处理元素 {i}/{len(elements_to_process)}...")

        elem_id = simplified_elem.get('element_id')
        if not elem_id: continue
        if elem_id in processed_ids: continue

        # 从原始数据中查找完整信息
        raw_elem = raw_data_lookup.get(elem_id)
        if not raw_elem:
            raw_elem = simplified_elem # 回退

        # 获取必要的属性
        wkt_geom = raw_elem.get('element_geometry')
        elem_type = raw_elem.get('element_type', 'Unknown').upper()
        # 尝试将高度和起始高度转换为浮点数
        try:
            height = float(raw_elem.get('element_height', 0.0))
        except (ValueError, TypeError):
            height = 0.0
        try:
            start_height = float(raw_elem.get('element_start_height', 0.0))
        except (ValueError, TypeError):
            start_height = 0.0

        custom_semantics = raw_elem.get('element_custom_semantics', {})

        if not wkt_geom: continue

        # 解析 WKT 获取 2D 坐标
        poly_coords_2d = parse_wkt_polygon(wkt_geom)
        if poly_coords_2d is None or len(poly_coords_2d) < 3:
            skipped_invalid_geom += 1
            continue

        # 确定语义标签和颜色
        label = -1
        if elem_type == 'BUILDING':
            label = 1
            if height > 100.0: label = 4
            # 存储建筑信息
            building_data[elem_id] = {
                'coords': poly_coords_2d,
                'height': height,
                'start_height': start_height,
                'num_storeys': custom_semantics.get('num_above_ground_storeys'),
                'id': elem_id
            }
        elif elem_type == 'ROAD':
            label = 2
        elif elem_type in ['LANDSCAPE', 'LOT', 'BASE']:
            # 稍微放宽地面/景观的条件
            if height < 1.0 and start_height < 5.0:
                 label = 0
            else:
                 skipped_types.add(f"{elem_type}(H:{height:.1f},SH:{start_height:.1f})")
                 continue
        elif elem_type == 'WATER':
            label = 3
        elif elem_type in ['LAND_BOUNDARY', 'GLA']: # 显式跳过这些类型
             skipped_types.add(elem_type)
             continue
        else:
            skipped_types.add(elem_type)
            continue

        color = C.get(label, PALETTE_BACKGROUND_COLOR)

        # 拉伸多边形生成 3D 网格
        # TODO: 替换 triangulate_polygon_simple 为更健壮的方法
        verts, faces = extrude_polygon(poly_coords_2d, start_height, height)

        if len(verts) == 0 or len(faces) == 0: continue

        # 将生成的三角形添加到场景列表
        for face_indices in faces:
            try:
                v0_idx, v1_idx, v2_idx = face_indices
                v0 = verts[v0_idx]
                v1 = verts[v1_idx]
                v2 = verts[v2_idx]
                # 检查退化三角形 (面积接近零)
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross_prod = np.cross(edge1, edge2)
                area_sq = np.dot(cross_prod, cross_prod)
                if area_sq < (INTERSECTION_EPSILON ** 2):
                    # print(f"    警告: 跳过元素 {elem_id} 中的退化三角形。")
                    continue

                tris_vertices_tuples.append((v0, v1, v2))
                tris_labels.append(label)
                tris_colors.append(color)
                element_ids.append(elem_id)
            except IndexError:
                print(f"    错误: 处理元素 {elem_id} 的面时出现索引错误。面: {face_indices}, 顶点数: {len(verts)}")
                continue

        processed_ids.add(elem_id)

    print(f"    处理完成。跳过的类型: {skipped_types}")
    print(f"    因无效几何跳过: {skipped_invalid_geom} 个元素。")
    print(f"    已处理 {len(processed_ids)} 个唯一元素 ID。")
    print(f"    收集到 {len(building_data)} 个建筑元素的信息。")

    # --- 3. 加载并添加山体地形 ---
    print("    加载并添加山体地形 OBJ...")
    obj_label = 6 # 山脉标签
    obj_color = C[obj_label]

    for obj_path in [NORTH_MOUNTAINS_OBJ_PATH, SOUTH_MOUNTAINS_OBJ_PATH]:
        verts_obj, faces_obj = load_obj_file_simple(obj_path)
        if len(verts_obj) > 0 and len(faces_obj) > 0:
            print(f"    添加来自 {os.path.basename(obj_path)} 的 {len(faces_obj)} 个三角形...")
            for face_indices in faces_obj:
                 try:
                     v0_idx, v1_idx, v2_idx = face_indices
                     v0 = verts_obj[v0_idx]
                     v1 = verts_obj[v1_idx]
                     v2 = verts_obj[v2_idx]
                     # 检查退化三角形
                     edge1 = v1 - v0; edge2 = v2 - v0
                     cross_prod = np.cross(edge1, edge2)
                     area_sq = np.dot(cross_prod, cross_prod)
                     if area_sq < (INTERSECTION_EPSILON ** 2): continue

                     tris_vertices_tuples.append((v0, v1, v2))
                     tris_labels.append(obj_label)
                     tris_colors.append(obj_color)
                     element_ids.append(f"terrain_{os.path.basename(obj_path)}") # 标记来源
                 except IndexError:
                     print(f"    错误: 处理 OBJ {os.path.basename(obj_path)} 的面时出现索引错误。面: {face_indices}, 顶点数: {len(verts_obj)}")
                     continue
        else:
             print(f"    警告: 未能从 {obj_path} 加载有效的地形数据。")


    # --- 4. 加载并添加树木 ---
    print("    加载、转换并添加树木...")
    tree_data = load_json_file(TREE_POINT_PATH)
    tree_label = 5 # 树木标签
    tree_color = C[tree_label]
    added_trees = 0
    skipped_trees = 0
    skipped_coord_errors = 0
    skipped_invalid_size = 0

    # TODO: 实现地面高度查找。暂时使用固定高度。
    # 可以创建一个包含所有地面/景观/道路三角形的简化 BVH
    # 然后对每个树木位置进行光线投射以找到 Z 值。
    DEFAULT_GROUND_Z = 0.5 # 临时地面高度

    if tree_data and 'features' in tree_data and transformer_latlon_to_hk:
        print(f"    处理 {len(tree_data['features'])} 个树木要素...")
        target_tree_count = 10000 # 增加目标树木数量，如果需要
        for i, feature in enumerate(tree_data['features']):
            if added_trees >= target_tree_count:
                print(f"    已达到目标树木数量 ({target_tree_count})，停止添加。")
                break

            if i % 1000 == 0 and i > 0:
                 print(f"      处理树木 {i}/{len(tree_data['features'])}...")

            geom = feature.get('geometry')
            props = feature.get('properties')
            if not geom or geom.get('type') != 'Point' or not props:
                skipped_trees += 1
                continue

            try:
                 lon, lat = geom['coordinates']
                 # 尝试转换属性为数值
                 height = float(props.get('Height_M'))
                 crown_spread = float(props.get('Crown_Spread_M'))
            except (ValueError, TypeError, IndexError):
                 skipped_invalid_size += 1
                 continue # 跳过属性无效的树

            if height <= 0 or crown_spread <= 0:
                skipped_invalid_size += 1
                continue

            try:
                # 坐标转换
                hk_x, hk_y = transformer_latlon_to_hk.transform(lon, lat)
            except Exception as e:
                # print(f"    警告: 树木坐标转换失败 ({lon}, {lat}): {e}")
                skipped_coord_errors += 1
                continue

            # 获取地面高度 (使用临时值)
            base_z = DEFAULT_GROUND_Z
            radius = crown_spread / 2.0

            # 创建树木网格
            verts_tree, faces_tree = create_cone_tree_mesh((hk_x, hk_y), base_z, height, radius)

            if len(verts_tree) > 0 and len(faces_tree) > 0:
                for face_indices in faces_tree:
                    try:
                        v0_idx, v1_idx, v2_idx = face_indices
                        v0 = verts_tree[v0_idx]
                        v1 = verts_tree[v1_idx]
                        v2 = verts_tree[v2_idx]
                        # 检查退化三角形
                        edge1 = v1 - v0; edge2 = v2 - v0
                        cross_prod = np.cross(edge1, edge2)
                        area_sq = np.dot(cross_prod, cross_prod)
                        if area_sq < (INTERSECTION_EPSILON ** 2): continue

                        tris_vertices_tuples.append((v0, v1, v2))
                        tris_labels.append(tree_label)
                        tris_colors.append(tree_color)
                        element_ids.append(f"tree_{props.get('TMCP_Tree_ID', i)}") # 标记来源
                    except IndexError:
                         print(f"    错误: 处理树木 {i} 的面时出现索引错误。面: {face_indices}, 顶点数: {len(verts_tree)}")
                         continue
                added_trees += 1
            else:
                skipped_trees += 1

        print(f"    添加了 {added_trees} 棵树。")
        if skipped_trees > 0: print(f"    因未知原因跳过 {skipped_trees} 棵树。")
        if skipped_coord_errors > 0: print(f"    因坐标转换错误跳过 {skipped_coord_errors} 棵树。")
        if skipped_invalid_size > 0: print(f"    因无效尺寸或属性跳过 {skipped_invalid_size} 棵树。")

    elif not transformer_latlon_to_hk:
         print("    警告: 坐标转换器不可用，跳过添加树木。")
    else:
        print("    警告: 未能加载树木数据或数据格式无效，跳过添加树木。")


    # --- 5. 完成场景数据转换 ---
    N = len(tris_vertices_tuples) # 三角形总数
    if N == 0:
        print("错误: 场景构建结果为 0 个三角形!")
        return [], [], [], [], [], [], [], {} # 返回空

    print(f"    场景构建完成。总三角形数: {N}")

    # 将元组列表转换为 numpy 数组 (与 demo 类似)
    v0s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    e1s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    e2s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    normals = np.empty((N, 3), dtype=GEOMETRY_DTYPE)

    # 使用 Numba 加速转换和法线计算
    @njit(parallel=True, fastmath=True)
    def finalize_geometry_parallel(n_tris, tris_tuples, v0s_out, e1s_out, e2s_out, normals_out):
        for i in prange(n_tris):
            v0_tuple, v1_tuple, v2_tuple = tris_tuples[i]
            # 手动解包元组以帮助 Numba
            v0_np = np.array([v0_tuple[0], v0_tuple[1], v0_tuple[2]], dtype=GEOMETRY_DTYPE)
            v1_np = np.array([v1_tuple[0], v1_tuple[1], v1_tuple[2]], dtype=GEOMETRY_DTYPE)
            v2_np = np.array([v2_tuple[0], v2_tuple[1], v2_tuple[2]], dtype=GEOMETRY_DTYPE)

            v0s_out[i] = v0_np
            edge1 = v1_np - v0_np
            edge2 = v2_np - v0_np
            e1s_out[i] = edge1
            e2s_out[i] = edge2

            # 计算面法线
            normal_x = edge1[1] * edge2[2] - edge1[2] * edge2[1]
            normal_y = edge1[2] * edge2[0] - edge1[0] * edge2[2]
            normal_z = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            norm_len_sq = normal_x**2 + normal_y**2 + normal_z**2

            if norm_len_sq > INTERSECTION_EPSILON**2:
                inv_norm_len = 1.0 / math.sqrt(norm_len_sq)
                normals_out[i, 0] = normal_x * inv_norm_len
                normals_out[i, 1] = normal_y * inv_norm_len
                normals_out[i, 2] = normal_z * inv_norm_len
            else:
                # 退化三角形，默认为向上 (假设 Z 向上)
                # *** 重要: 如果 HK1980 是 Y 向上, 改为 [0.0, 1.0, 0.0] ***
                normals_out[i, 0] = 0.0
                normals_out[i, 1] = 0.0
                normals_out[i, 2] = 1.0

    # 需要将元组列表转换为 Numba 可处理的类型，例如对象数组或结构化数组
    # 为了简单起见，暂时不在 Numba 中处理元组列表，保持原始循环
    # 如果性能是瓶颈，需要重构 tris_vertices_tuples 的存储方式
    print("    正在转换几何数据并计算法线...")
    t_finalize = _now()
    for i, (v0_tuple, v1_tuple, v2_tuple) in enumerate(tris_vertices_tuples):
        v0_np = np.array(v0_tuple, dtype=GEOMETRY_DTYPE)
        v1_np = np.array(v1_tuple, dtype=GEOMETRY_DTYPE)
        v2_np = np.array(v2_tuple, dtype=GEOMETRY_DTYPE)

        v0s[i] = v0_np
        edge1 = np.subtract(v1_np, v0_np)
        edge2 = np.subtract(v2_np, v0_np)
        e1s[i] = edge1
        e2s[i] = edge2

        # 计算面法线
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len > INTERSECTION_EPSILON:
            normals[i] = normal / norm_len
        else:
            # 退化三角形，默认为向上 (假设 Z 向上)
            # *** 重要: 如果 HK1980 是 Y 向上, 改为 [0.0, 1.0, 0.0] ***
            normals[i] = np.array([0.0, 0.0, 1.0], dtype=GEOMETRY_DTYPE)
    log_step("几何数据转换和法线计算", t_finalize)


    labels_np = np.array(tris_labels, dtype=LABEL_DTYPE)
    colors_np = np.array(tris_colors, dtype=COLOR_DTYPE)
    prim_indices = np.arange(N, dtype=INDEX_DTYPE) # 原始索引 [0, ..., N-1]

    # 返回所有必要的几何数据和建筑信息
    return v0s, e1s, e2s, normals, labels_np, colors_np, prim_indices, building_data


# --- [修改] 视点生成函数 ---
def generate_window_viewpoints(building_data, num_target_buildings=5, default_floor_height=3.0, min_segment_length=1.0):
    """为选定的建筑生成窗口视点 (改进版：基于多边形段)。"""
    viewpoints = []
    if not building_data:
        print("    警告: 没有建筑数据可用于生成视点。")
        return viewpoints

    print(f"    尝试为最多 {num_target_buildings} 个建筑生成视点...")

    # 选择目标建筑 (例如，最高的几个)
    # 按高度对建筑进行排序 (降序)
    sorted_buildings = sorted(building_data.values(), key=lambda b: b.get('height', 0), reverse=True)

    target_buildings = sorted_buildings[:num_target_buildings]
    print(f"    选择了 {len(target_buildings)} 个建筑进行视点生成:")
    for b in target_buildings:
         print(f"      - ID: {b['id']}, Height: {b.get('height', 0):.1f}m")


    processed_building_count = 0
    for building in target_buildings:
        b_id = building['id']
        coords_2d = building['coords'] # [[x1, y1], [x2, y2], ...]
        height = building['height']
        start_height = building['start_height']
        num_storeys = building['num_storeys']

        if height <= 0:
            print(f"      跳过建筑 {b_id} (无效高度: {height})")
            continue
        if len(coords_2d) < 3:
             print(f"      跳过建筑 {b_id} (顶点数少于 3)")
             continue

        print(f"    处理建筑 {b_id} (Height: {height:.1f}m)...")
        processed_building_count += 1
        building_viewpoints_count = 0

        # 估算楼层数 (如果缺少)
        if num_storeys is None or not isinstance(num_storeys, (int, float)) or num_storeys <= 0:
            num_storeys_calc = max(1, int(round(height / default_floor_height)))
            # print(f"      建筑 {b_id}: 缺少楼层数，估算为 {num_storeys_calc} 层")
        else:
            num_storeys_calc = int(num_storeys)
            # print(f"      建筑 {b_id}: {num_storeys_calc} 层")


        # --- 遍历建筑轮廓的线段 ---
        num_verts = len(coords_2d)
        for i in range(num_verts):
            p1 = coords_2d[i]          # 当前点 (x, y)
            p2 = coords_2d[(i + 1) % num_verts] # 下一个点 (环绕)

            # 计算线段向量和长度
            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)

            # 跳过非常短的线段
            if segment_len < min_segment_length:
                continue

            # 计算线段中点 (立面上的点)
            mid_point_2d = (p1 + p2) / 2.0

            # 计算线段的法线 (向外) - 假设 Z 向上
            # *** 重要: 如果 HK1980 是 Y 向上, 法线计算需要调整 ***
            # 2D 法线: (dy, -dx) 或 (-dy, dx)
            dx = segment_vec[0]
            dy = segment_vec[1]
            # 选择一个法线方向，例如 (dy, -dx)
            normal_2d = np.array([dy, -dx], dtype=GEOMETRY_DTYPE)
            normal_2d /= np.linalg.norm(normal_2d) # 归一化

            # TODO: 需要检查法线是否真的指向外部。
            # 可以通过检查多边形中心点与线段中点+法线的关系来判断。
            # 暂时假设计算出的法线是向外的。

            # 3D 法线 (相机方向)
            cam_dir = np.array([normal_2d[0], normal_2d[1], 0.0], dtype=GEOMETRY_DTYPE)

            # 为此立面段的每一层生成视点
            for floor in range(1, num_storeys_calc + 1):
                # 计算窗口中心点的 Z 坐标 (楼层中点)
                window_z = start_height + (floor - 0.5) * default_floor_height

                # 计算相机原点 (稍微向内移动)
                offset_distance = 0.5 # 向内移动 0.5 米 (避免穿墙)
                cam_o_3d = np.array([mid_point_2d[0], mid_point_2d[1], window_z], dtype=GEOMETRY_DTYPE)
                cam_o_3d -= cam_dir * offset_distance # 向内移动

                # 相机向上向量 (假设 Z 轴是世界向上)
                # *** 重要: 如果 HK1980 是 Y 向上, 改为 [0.0, 1.0, 0.0] ***
                cam_up_vec = np.array([0.0, 0.0, 1.0], dtype=GEOMETRY_DTYPE)

                # 存储视点信息
                viewpoint = {
                    "building_id": b_id,
                    "floor": floor,
                    "face_segment_index": i, # 标记是哪个立面段
                    "cam_o": cam_o_3d,
                    "cam_dir": cam_dir,
                    "cam_up_vec": cam_up_vec
                }
                viewpoints.append(viewpoint)
                building_viewpoints_count += 1
        print(f"      为建筑 {b_id} 生成了 {building_viewpoints_count} 个视点。")


    print(f"    总共为 {processed_building_count} 个建筑生成了 {len(viewpoints)} 个视点。")
    return viewpoints


# --- BVH 构建 (来自 demo, 无需大改) ---
# int32_to_float32_bits, float32_to_int32_bits (不变)
@njit
def int32_to_float32_bits(val_int32):
    int_array = np.array([val_int32], dtype=INDEX_DTYPE)
    return int_array.view(BVH_NODE_DTYPE)[0]
@njit
def float32_to_int32_bits(val_float32):
    float_array = np.array([val_float32], dtype=BVH_NODE_DTYPE)
    return float_array.view(INDEX_DTYPE)[0]

# calculate_tri_aabb_numba, calculate_bounds (不变)
@njit(fastmath=True)
def calculate_tri_aabb_numba(v0, e1, e2):
    v1 = v0 + e1
    v2 = v0 + e2
    min_coord = np.empty(3, dtype=GEOMETRY_DTYPE)
    max_coord = np.empty(3, dtype=GEOMETRY_DTYPE)
    for k in range(3):
        min_coord[k] = min(v0[k], v1[k], v2[k])
        max_coord[k] = max(v0[k], v1[k], v2[k])
    return min_coord, max_coord
@njit(fastmath=True)
def calculate_bounds(indices, tri_aabbs_min, tri_aabbs_max):
    num_tris = len(indices)
    if num_tris == 0:
        return (np.full(3, INF, dtype=BVH_NODE_DTYPE),
                np.full(3, -INF, dtype=BVH_NODE_DTYPE))
    first_idx = indices[0]
    global_min = tri_aabbs_min[first_idx].copy()
    global_max = tri_aabbs_max[first_idx].copy()
    for i in range(1, num_tris):
        idx = indices[i]
        current_min = tri_aabbs_min[idx]
        current_max = tri_aabbs_max[idx]
        for k in range(3):
            global_min[k] = min(global_min[k], current_min[k])
            global_max[k] = max(global_max[k], current_max[k])
    return global_min, global_max

# recursive_build_numba (不变)
@njit
def recursive_build_numba(current_node_idx, nodes_used, flat_nodes, prim_indices,
                          start_idx, end_idx,
                          tri_aabbs_min, tri_aabbs_max, tri_centers):
    num_prims = end_idx - start_idx
    node = flat_nodes[current_node_idx]
    indices_slice = prim_indices[start_idx:end_idx]
    aabb_min, aabb_max = calculate_bounds(indices_slice, tri_aabbs_min, tri_aabbs_max)
    node[0:3] = aabb_min
    node[3:6] = aabb_max

    if num_prims <= BVH_MAX_LEAF_SIZE:
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx))
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims))
        return

    extent = aabb_max - aabb_min
    split_axis = np.argmax(extent)
    if extent[split_axis] < INTERSECTION_EPSILON:
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx))
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims))
        return

    split_coord = (aabb_min[split_axis] + aabb_max[split_axis]) * 0.5
    mid_point = start_idx
    for i in range(start_idx, end_idx):
        prim_idx = prim_indices[i]
        if tri_centers[prim_idx, split_axis] < split_coord:
            prim_indices[i], prim_indices[mid_point] = prim_indices[mid_point], prim_indices[i]
            mid_point += 1

    if mid_point == start_idx or mid_point == end_idx:
        mid_point = start_idx + num_prims // 2

    left_child_idx = nodes_used[0]
    nodes_used[0] += 1
    right_child_idx = nodes_used[0]
    nodes_used[0] += 1

    node[6] = int32_to_float32_bits(INDEX_DTYPE(left_child_idx))
    node[7] = int32_to_float32_bits(INDEX_DTYPE(right_child_idx))

    recursive_build_numba(left_child_idx, nodes_used, flat_nodes, prim_indices,
                          start_idx, mid_point,
                          tri_aabbs_min, tri_aabbs_max, tri_centers)
    recursive_build_numba(right_child_idx, nodes_used, flat_nodes, prim_indices,
                          mid_point, end_idx,
                          tri_aabbs_min, tri_aabbs_max, tri_centers)

# build_bvh (不变)
def build_bvh(v0s, e1s, e2s, prim_indices_in):
    N = len(prim_indices_in)
    if N == 0:
        return np.empty((0, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE), np.empty(0, dtype=INDEX_DTYPE)

    t0_precompute = _now()
    tri_aabbs_min = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_aabbs_max = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_centers = np.empty((N, 3), dtype=BVH_NODE_DTYPE)

    @njit(parallel=True, fastmath=True)
    def precompute_bounds_centers_parallel(num_tris, v0s_n, e1s_n, e2s_n,
                                           aabbs_min_out, aabbs_max_out, centers_out):
        for i in prange(num_tris):
            v0, e1, e2 = v0s_n[i], e1s_n[i], e2s_n[i]
            aabb_min, aabb_max = calculate_tri_aabb_numba(v0, e1, e2)
            aabbs_min_out[i] = aabb_min
            aabbs_max_out[i] = aabb_max
            centers_out[i] = (aabb_min + aabb_max) * 0.5

    precompute_bounds_centers_parallel(N, v0s, e1s, e2s,
                                       tri_aabbs_min, tri_aabbs_max, tri_centers)
    log_step('BVH 预计算', t0_precompute)

    max_nodes = max(1, 2 * N - 1)
    flat_nodes = np.zeros((max_nodes, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE)
    ordered_prim_indices = np.copy(prim_indices_in)
    nodes_used = np.array([1], dtype=INDEX_DTYPE)

    t0_recursive = _now()
    try:
        recursive_build_numba(0, nodes_used, flat_nodes, ordered_prim_indices,
                              0, N,
                              tri_aabbs_min, tri_aabbs_max, tri_centers)
    except Exception as e:
         print(f"    BVH 递归构建期间出错: {e}")
         # 即使出错，也尝试返回部分构建的 BVH
         actual_nodes_used = nodes_used[0]
         flat_nodes = flat_nodes[:actual_nodes_used]
         print(f"    警告: BVH 构建可能未完成。使用了 {actual_nodes_used} 个节点。")
         return flat_nodes, ordered_prim_indices # 返回部分结果

    log_step('BVH 递归构建', t0_recursive)

    actual_nodes_used = nodes_used[0]
    flat_nodes = flat_nodes[:actual_nodes_used]
    print(f"    BVH 构建完成: 使用了 {actual_nodes_used} 个节点。")
    if actual_nodes_used > 0:
        root_min = flat_nodes[0, 0:3]
        root_max = flat_nodes[0, 3:6]
        # 调整根节点边界打印格式
        print(f"    根节点 AABB Min: [{root_min[0]:.2f}, {root_min[1]:.2f}, {root_min[2]:.2f}]")
        print(f"    根节点 AABB Max: [{root_max[0]:.2f}, {root_max[1]:.2f}, {root_max[2]:.2f}]")


    return flat_nodes, ordered_prim_indices


# --- 光线追踪核心 (来自 demo, 无需大改) ---
# intersect_ray_triangle_cpu, intersect_ray_aabb_cpu, trace_shadow_ray_cpu (不变)
@njit(fastmath=True)
def intersect_ray_triangle_cpu(orig, dir, v0, e1, e2):
    h = np.cross(dir, e2)
    a = np.dot(e1, h)
    if abs(a) < INTERSECTION_EPSILON: return INF
    f = GEOMETRY_DTYPE(1.0) / a
    s = orig - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0: return INF
    q = np.cross(s, e1)
    v = f * np.dot(dir, q)
    if v < 0.0 or u + v > 1.0: return INF
    t = f * np.dot(e2, q)
    return t if t > INTERSECTION_EPSILON else INF

@njit(fastmath=True)
def intersect_ray_aabb_cpu(orig, dir_inv, tmin_global, node_aabb_min, node_aabb_max):
    t_near = -INF; t_far = INF
    for k in range(3):
        inv_d = dir_inv[k]; aabb_min_k = node_aabb_min[k]; aabb_max_k = node_aabb_max[k]
        t1 = (aabb_min_k - orig[k]) * inv_d
        t2 = (aabb_max_k - orig[k]) * inv_d
        if t1 > t2: t1, t2 = t2, t1
        t_near = max(t_near, t1); t_far = min(t_far, t2)
        if t_near >= t_far or t_far < INTERSECTION_EPSILON or t_near >= tmin_global: return False
    return True

@njit(fastmath=True)
def trace_shadow_ray_cpu(shadow_orig, shadow_dir, max_dist,
                         flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered):
    num_nodes = flat_nodes.shape[0]; num_tris_total = len(v0s_reordered)
    if num_nodes == 0: return False
    dir_inv = np.empty(3, dtype=GEOMETRY_DTYPE)
    for k in range(3):
        dir_k = shadow_dir[k]
        dir_inv[k] = math.copysign(INF, dir_k) if abs(dir_k) < INTERSECTION_EPSILON else GEOMETRY_DTYPE(1.0) / dir_k

    BVH_CPU_STACK_SIZE = 64
    node_stack = np.empty(BVH_CPU_STACK_SIZE, dtype=INDEX_DTYPE)
    stack_ptr = 0; node_stack[stack_ptr] = 0; stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1; node_idx = node_stack[stack_ptr]
        if node_idx < 0 or node_idx >= num_nodes: continue
        node = flat_nodes[node_idx]; node_aabb_min = node[0:3]; node_aabb_max = node[3:6]
        if not intersect_ray_aabb_cpu(shadow_orig, dir_inv, max_dist, node_aabb_min, node_aabb_max): continue

        info_bits = node[7]; info_val = float32_to_int32_bits(info_bits)
        if info_val < 0: # Leaf
            prim_count = -info_val; prim_offset_bits = node[6]; prim_offset = float32_to_int32_bits(prim_offset_bits)
            for p_local_idx in range(prim_count):
                p_idx = prim_offset + p_local_idx
                if p_idx < num_tris_total:
                    t = intersect_ray_triangle_cpu(shadow_orig, shadow_dir,
                                                   v0s_reordered[p_idx], e1s_reordered[p_idx], e2s_reordered[p_idx])
                    if t < max_dist: return True # Occluded
        else: # Internal
            left_child_idx_bits = node[6]; left_child_idx = float32_to_int32_bits(left_child_idx_bits)
            right_child_idx = info_val
            if stack_ptr + 2 <= BVH_CPU_STACK_SIZE:
                if right_child_idx >= 0 and right_child_idx < num_nodes: node_stack[stack_ptr] = right_child_idx; stack_ptr += 1
                if left_child_idx >= 0 and left_child_idx < num_nodes: node_stack[stack_ptr] = left_child_idx; stack_ptr += 1
    return False # Not occluded

# raytrace_cpu_bvh (不变)
@njit(parallel=True, fastmath=True)
def raytrace_cpu_bvh(
    flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
    normals_reordered, labels_reordered, colors_reordered,
    cam_o, cam_dir, right, up, screen_w, screen_h, W, H
):
    rgb = np.zeros((H, W, 3), dtype=COLOR_DTYPE)
    depth = np.full((H, W), INF, dtype=DEPTH_DTYPE)
    sem = np.full((H, W), SKY_LABEL, dtype=LABEL_DTYPE)
    pts = np.full((H, W, 3), np.nan, dtype=POINT_DTYPE)

    inv_W = GEOMETRY_DTYPE(1.0 / W); inv_H = GEOMETRY_DTYPE(1.0 / H)
    num_nodes = flat_nodes.shape[0]; num_tris_total = len(v0s_reordered)
    if num_nodes == 0: return rgb, depth, sem, pts

    sun_dir_cpu = SUN_DIRECTION; ambient_light_cpu = AMBIENT_LIGHT
    specular_color_cpu = SPECULAR_COLOR; specular_exponent_cpu = SPECULAR_EXPONENT
    ks_cpu = Ks; shadow_bias_cpu = SHADOW_BIAS
    default_sky_color_cpu = DEFAULT_SKY_COLOR

    for i in prange(H): # Parallel loop over rows
        node_stack = np.empty(64, dtype=INDEX_DTYPE)
        d = np.empty(3, dtype=GEOMETRY_DTYPE); dir_inv = np.empty(3, dtype=GEOMETRY_DTYPE)
        hit_normal = np.empty(3, dtype=GEOMETRY_DTYPE); hit_point = np.empty(3, dtype=GEOMETRY_DTYPE)
        shadow_ray_origin = np.empty(3, dtype=GEOMETRY_DTYPE); view_dir = np.empty(3, dtype=GEOMETRY_DTYPE)
        reflect_dir = np.empty(3, dtype=GEOMETRY_DTYPE); base_color_float = np.empty(3, dtype=GEOMETRY_DTYPE)
        final_color_float = np.empty(3, dtype=GEOMETRY_DTYPE)

        for j in range(W): # Loop over columns
            # 1. Ray direction
            u_norm = (GEOMETRY_DTYPE(j) + 0.5) * inv_W - 0.5
            v_norm = (GEOMETRY_DTYPE(i) + 0.5) * inv_H - 0.5
            d[0] = cam_dir[0] + u_norm * screen_w * right[0] - v_norm * screen_h * up[0]
            d[1] = cam_dir[1] + u_norm * screen_w * right[1] - v_norm * screen_h * up[1]
            d[2] = cam_dir[2] + u_norm * screen_w * right[2] - v_norm * screen_h * up[2]
            norm_sq = d[0]**2 + d[1]**2 + d[2]**2
            if norm_sq < INTERSECTION_EPSILON**2:
                rgb[i, j, :] = default_sky_color_cpu; continue
            inv_norm = GEOMETRY_DTYPE(1.0) / math.sqrt(norm_sq); d *= inv_norm
            for k in range(3):
                 dir_inv[k] = math.copysign(INF, d[k]) if abs(d[k]) < INTERSECTION_EPSILON else GEOMETRY_DTYPE(1.0) / d[k]

            # 2. BVH Traversal
            tmin = INF; hit_prim_idx = -1
            stack_ptr = 0; node_stack[stack_ptr] = 0; stack_ptr += 1
            while stack_ptr > 0:
                stack_ptr -= 1; node_idx = node_stack[stack_ptr]
                if node_idx < 0 or node_idx >= num_nodes: continue
                node = flat_nodes[node_idx]; node_aabb_min = node[0:3]; node_aabb_max = node[3:6]
                if not intersect_ray_aabb_cpu(cam_o, dir_inv, tmin, node_aabb_min, node_aabb_max): continue

                info_bits = node[7]; info_val = float32_to_int32_bits(info_bits)
                if info_val < 0: # Leaf
                    prim_count = -info_val; prim_offset_bits = node[6]; prim_offset = float32_to_int32_bits(prim_offset_bits)
                    for p_local_idx in range(prim_count):
                        p_idx = prim_offset + p_local_idx
                        if p_idx < num_tris_total:
                            t = intersect_ray_triangle_cpu(cam_o, d,
                                                           v0s_reordered[p_idx], e1s_reordered[p_idx], e2s_reordered[p_idx])
                            if t < tmin:
                                tmin = t; hit_prim_idx = p_idx
                                hit_normal = normals_reordered[p_idx] # Copy normal
                else: # Internal
                    left_child_idx_bits = node[6]; right_child_idx = info_val
                    left_child_idx = float32_to_int32_bits(left_child_idx_bits)
                    if stack_ptr + 2 <= 64:
                        if right_child_idx >= 0 and right_child_idx < num_nodes: node_stack[stack_ptr] = right_child_idx; stack_ptr += 1
                        if left_child_idx >= 0 and left_child_idx < num_nodes: node_stack[stack_ptr] = left_child_idx; stack_ptr += 1

            # 3. Shading
            if hit_prim_idx >= 0: # Hit
                depth[i, j] = tmin
                sem[i, j] = labels_reordered[hit_prim_idx]
                hit_point = cam_o + tmin * d
                pts[i, j, :] = hit_point

                color_uint8 = colors_reordered[hit_prim_idx]
                base_color_float[0] = GEOMETRY_DTYPE(color_uint8[0]) / 255.0
                base_color_float[1] = GEOMETRY_DTYPE(color_uint8[1]) / 255.0
                base_color_float[2] = GEOMETRY_DTYPE(color_uint8[2]) / 255.0

                if np.dot(hit_normal, d) > 0.0: hit_normal = -hit_normal

                light_dot_normal = max(0.0, np.dot(hit_normal, sun_dir_cpu))
                diffuse_intensity = SUN_INTENSITY * light_dot_normal

                shadow_ray_origin = hit_point + hit_normal * shadow_bias_cpu
                is_occluded = trace_shadow_ray_cpu(shadow_ray_origin, sun_dir_cpu, INF,
                                                   flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered)
                shadow_factor = GEOMETRY_DTYPE(0.0) if is_occluded else GEOMETRY_DTYPE(1.0)

                view_dir = cam_o - hit_point
                view_dir_norm = np.linalg.norm(view_dir)
                if view_dir_norm > INTERSECTION_EPSILON: view_dir /= view_dir_norm
                else: view_dir = -d

                reflect_dot = 2.0 * np.dot(hit_normal, sun_dir_cpu)
                reflect_dir = reflect_dot * hit_normal - sun_dir_cpu

                specular_dot_view = max(0.0, np.dot(view_dir, reflect_dir))
                specular_intensity = ks_cpu * (specular_dot_view ** specular_exponent_cpu)

                final_color_float = base_color_float * (ambient_light_cpu + diffuse_intensity * shadow_factor) + specular_color_cpu * specular_intensity * shadow_factor

                final_r = max(0.0, min(1.0, final_color_float[0]))
                final_g = max(0.0, min(1.0, final_color_float[1]))
                final_b = max(0.0, min(1.0, final_color_float[2]))
                rgb[i, j, 0] = int(final_r * 255.0)
                rgb[i, j, 1] = int(final_g * 255.0)
                rgb[i, j, 2] = int(final_b * 255.0)
            else: # Sky
                rgb[i, j, :] = default_sky_color_cpu
                # depth, sem, pts already initialized

    return rgb, depth, sem, pts

# GPU Kernels (ray_tri_intersect_gpu, ray_aabb_intersect_gpu, trace_shadow_ray_gpu_device, raytrace_cuda_bvh_kernel)
# 这些函数直接从你的 demo 复制过来，因为它们的核心逻辑不变
if _GPU_AVAILABLE:
    @cuda.jit(device=True, inline=True)
    def ray_tri_intersect_gpu(orig, dir, v0, e1, e2):
        eps_gpu = INTERSECTION_EPSILON; inf_gpu = GEOMETRY_DTYPE(1e20)
        h0 = dir[1] * e2[2] - dir[2] * e2[1]; h1 = dir[2] * e2[0] - dir[0] * e2[2]; h2 = dir[0] * e2[1] - dir[1] * e2[0]
        a = e1[0] * h0 + e1[1] * h1 + e1[2] * h2
        if abs(a) < eps_gpu: return inf_gpu
        f = GEOMETRY_DTYPE(1.0) / a
        s0 = orig[0] - v0[0]; s1 = orig[1] - v0[1]; s2 = orig[2] - v0[2]
        u = f * (s0 * h0 + s1 * h1 + s2 * h2)
        if u < 0.0 or u > 1.0: return inf_gpu
        q0 = s1 * e1[2] - s2 * e1[1]; q1 = s2 * e1[0] - s0 * e1[2]; q2 = s0 * e1[1] - s1 * e1[0]
        v = f * (dir[0] * q0 + dir[1] * q1 + dir[2] * q2)
        if v < 0.0 or u + v > 1.0: return inf_gpu
        t = f * (e2[0] * q0 + e2[1] * q1 + e2[2] * q2)
        return t if t > eps_gpu else inf_gpu

    @cuda.jit(device=True, inline=True)
    def ray_aabb_intersect_gpu(orig, dir_inv, t_min_global, node_aabb_min, node_aabb_max):
        t_near = -INF; t_far = INF; eps_aabb = INTERSECTION_EPSILON
        for k in range(3):
            inv_d = dir_inv[k]; aabb_min_k = node_aabb_min[k]; aabb_max_k = node_aabb_max[k]
            t1 = (aabb_min_k - orig[k]) * inv_d; t2 = (aabb_max_k - orig[k]) * inv_d
            if t1 > t2: t1, t2 = t2, t1
            t_near = max(t_near, t1); t_far = min(t_far, t2)
            if t_near >= t_far or t_far < eps_aabb or t_near >= t_min_global: return False
        return True

    @cuda.jit(device=True)
    def trace_shadow_ray_gpu_device(shadow_orig, shadow_dir, max_dist,
                                    flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered):
        num_nodes = flat_nodes.shape[0]; num_tris_total_gpu = v0s_reordered.shape[0]
        if num_nodes == 0: return False
        node_stack = cuda.local.array(64, dtype=INDEX_DTYPE); dir_inv = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        node_aabb_min_local = cuda.local.array(3, dtype=BVH_NODE_DTYPE); node_aabb_max_local = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        inf_gpu = GEOMETRY_DTYPE(1e20); eps_gpu = INTERSECTION_EPSILON

        for k in range(3):
            dir_k = shadow_dir[k]
            dir_inv[k] = math.copysign(inf_gpu, dir_k) if abs(dir_k) < eps_gpu else GEOMETRY_DTYPE(1.0) / dir_k

        stack_ptr = 0; node_stack[stack_ptr] = 0; stack_ptr += 1
        while stack_ptr > 0:
            stack_ptr -= 1; node_idx = node_stack[stack_ptr]
            if node_idx < 0 or node_idx >= num_nodes: continue

            node_aabb_min_local[0] = flat_nodes[node_idx, 0]; node_aabb_min_local[1] = flat_nodes[node_idx, 1]; node_aabb_min_local[2] = flat_nodes[node_idx, 2]
            node_aabb_max_local[0] = flat_nodes[node_idx, 3]; node_aabb_max_local[1] = flat_nodes[node_idx, 4]; node_aabb_max_local[2] = flat_nodes[node_idx, 5]

            if not ray_aabb_intersect_gpu(shadow_orig, dir_inv, max_dist, node_aabb_min_local, node_aabb_max_local): continue

            info_bits_f = flat_nodes[node_idx, 7]; info_val = info_bits_f.view(INDEX_DTYPE)
            if info_val < 0: # Leaf
                prim_count = -info_val; prim_offset_bits_f = flat_nodes[node_idx, 6]; prim_offset = prim_offset_bits_f.view(INDEX_DTYPE)
                for p_idx_offset in range(prim_count):
                    current_prim_idx = prim_offset + p_idx_offset
                    if current_prim_idx < num_tris_total_gpu:
                        tri_v0 = v0s_reordered[current_prim_idx]; tri_e1 = e1s_reordered[current_prim_idx]; tri_e2 = e2s_reordered[current_prim_idx]
                        t = ray_tri_intersect_gpu(shadow_orig, shadow_dir, tri_v0, tri_e1, tri_e2)
                        if t < max_dist: return True # Occluded
            else: # Internal
                left_child_idx_bits_f = flat_nodes[node_idx, 6]; left_child_idx = left_child_idx_bits_f.view(INDEX_DTYPE)
                right_child_idx = info_val
                if stack_ptr + 2 <= 64:
                    if right_child_idx >= 0 and right_child_idx < num_nodes: node_stack[stack_ptr] = right_child_idx; stack_ptr += 1
                    if left_child_idx >= 0 and left_child_idx < num_nodes: node_stack[stack_ptr] = left_child_idx; stack_ptr += 1
        return False # Not occluded

    @cuda.jit
    def raytrace_cuda_bvh_kernel(
        flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
        normals_reordered, labels_reordered, colors_reordered,
        cam_o, cam_dir, right, up, sun_dir_gpu, specular_color_gpu,
        scr_w, scr_h, W, H,
        rgb, depth, sem, pts
    ):
        i, j = cuda.grid(2)
        if i >= H or j >= W: return

        inf_gpu = GEOMETRY_DTYPE(1e20); eps_gpu = INTERSECTION_EPSILON
        ambient_light_gpu = AMBIENT_LIGHT; specular_exponent_gpu = SPECULAR_EXPONENT
        ks_gpu = Ks; shadow_bias_gpu = SHADOW_BIAS; sky_label_gpu = SKY_LABEL
        default_sky_r = DEFAULT_SKY_COLOR[0]; default_sky_g = DEFAULT_SKY_COLOR[1]; default_sky_b = DEFAULT_SKY_COLOR[2]

        d = cuda.local.array(3, dtype=GEOMETRY_DTYPE); dir_inv = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        node_aabb_min = cuda.local.array(3, dtype=BVH_NODE_DTYPE); node_aabb_max = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        hit_normal = cuda.local.array(3, dtype=GEOMETRY_DTYPE); hit_point = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        base_color_float = cuda.local.array(3, dtype=GEOMETRY_DTYPE); shadow_ray_origin = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        view_dir = cuda.local.array(3, dtype=GEOMETRY_DTYPE); reflect_dir = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        final_color_float = cuda.local.array(3, dtype=GEOMETRY_DTYPE)

        num_tris_total_gpu = v0s_reordered.shape[0]; num_nodes = flat_nodes.shape[0]

        # 1. Ray direction
        u_norm = (GEOMETRY_DTYPE(j) + 0.5) / W - 0.5; v_norm = (GEOMETRY_DTYPE(i) + 0.5) / H - 0.5
        d[0] = cam_dir[0] + u_norm * scr_w * right[0] - v_norm * scr_h * up[0]
        d[1] = cam_dir[1] + u_norm * scr_w * right[1] - v_norm * scr_h * up[1]
        d[2] = cam_dir[2] + u_norm * scr_w * right[2] - v_norm * scr_h * up[2]
        nrm_sq = d[0]**2 + d[1]**2 + d[2]**2
        if nrm_sq < eps_gpu**2:
             rgb[i, j, 0] = default_sky_r; rgb[i, j, 1] = default_sky_g; rgb[i, j, 2] = default_sky_b
             depth[i, j] = inf_gpu; sem[i, j] = sky_label_gpu; return
        inv_nrm = GEOMETRY_DTYPE(1.0) / math.sqrt(nrm_sq); d[0] *= inv_nrm; d[1] *= inv_nrm; d[2] *= inv_nrm
        for k in range(3):
            dir_k = d[k]; dir_inv[k] = math.copysign(inf_gpu, dir_k) if abs(dir_k) < eps_gpu else GEOMETRY_DTYPE(1.0) / dir_k

        # 2. Init traversal
        tmin = inf_gpu; hit_prim_idx = -1
        if num_nodes == 0:
            rgb[i, j, 0] = default_sky_r; rgb[i, j, 1] = default_sky_g; rgb[i, j, 2] = default_sky_b
            depth[i, j] = inf_gpu; sem[i, j] = sky_label_gpu; return

        BVH_GPU_STACK_SIZE = 64
        node_stack = cuda.local.array(BVH_GPU_STACK_SIZE, dtype=INDEX_DTYPE)
        stack_ptr = 0; node_stack[stack_ptr] = 0; stack_ptr += 1

        # 3. BVH Traversal
        while stack_ptr > 0:
            stack_ptr -= 1; node_idx = node_stack[stack_ptr]
            if node_idx < 0 or node_idx >= num_nodes: continue
            node_aabb_min[0] = flat_nodes[node_idx, 0]; node_aabb_min[1] = flat_nodes[node_idx, 1]; node_aabb_min[2] = flat_nodes[node_idx, 2]
            node_aabb_max[0] = flat_nodes[node_idx, 3]; node_aabb_max[1] = flat_nodes[node_idx, 4]; node_aabb_max[2] = flat_nodes[node_idx, 5]
            if not ray_aabb_intersect_gpu(cam_o, dir_inv, tmin, node_aabb_min, node_aabb_max): continue

            info_bits_f = flat_nodes[node_idx, 7]; info_val = info_bits_f.view(INDEX_DTYPE)
            if info_val < 0: # Leaf
                prim_count = -info_val; prim_offset_bits_f = flat_nodes[node_idx, 6]; prim_offset = prim_offset_bits_f.view(INDEX_DTYPE)
                for p_idx_offset in range(prim_count):
                    current_prim_idx = prim_offset + p_idx_offset
                    if current_prim_idx < num_tris_total_gpu:
                        tri_v0 = v0s_reordered[current_prim_idx]; tri_e1 = e1s_reordered[current_prim_idx]; tri_e2 = e2s_reordered[current_prim_idx]
                        t = ray_tri_intersect_gpu(cam_o, d, tri_v0, tri_e1, tri_e2)
                        if t < tmin:
                            tmin = t; hit_prim_idx = current_prim_idx
                            hit_normal[0] = normals_reordered[current_prim_idx, 0]; hit_normal[1] = normals_reordered[current_prim_idx, 1]; hit_normal[2] = normals_reordered[current_prim_idx, 2]
            else: # Internal
                left_child_idx_bits_f = flat_nodes[node_idx, 6]; left_child_idx = left_child_idx_bits_f.view(INDEX_DTYPE)
                right_child_idx = info_val
                if stack_ptr + 2 <= BVH_GPU_STACK_SIZE:
                    if right_child_idx >= 0 and right_child_idx < num_nodes: node_stack[stack_ptr] = right_child_idx; stack_ptr += 1
                    if left_child_idx >= 0 and left_child_idx < num_nodes: node_stack[stack_ptr] = left_child_idx; stack_ptr += 1

        # 4. Shading
        if hit_prim_idx >= 0: # Hit
            depth[i, j] = tmin
            sem[i, j] = labels_reordered[hit_prim_idx]
            hit_point[0] = cam_o[0] + d[0] * tmin; hit_point[1] = cam_o[1] + d[1] * tmin; hit_point[2] = cam_o[2] + d[2] * tmin
            pts[i, j, 0] = hit_point[0]; pts[i, j, 1] = hit_point[1]; pts[i, j, 2] = hit_point[2]

            color_uint8 = colors_reordered[hit_prim_idx]
            base_color_float[0] = GEOMETRY_DTYPE(color_uint8[0]) / 255.0; base_color_float[1] = GEOMETRY_DTYPE(color_uint8[1]) / 255.0; base_color_float[2] = GEOMETRY_DTYPE(color_uint8[2]) / 255.0

            normal_dot_ray = hit_normal[0] * d[0] + hit_normal[1] * d[1] + hit_normal[2] * d[2]
            if normal_dot_ray > 0.0: hit_normal[0] = -hit_normal[0]; hit_normal[1] = -hit_normal[1]; hit_normal[2] = -hit_normal[2]

            light_dot_normal = max(0.0, hit_normal[0] * sun_dir_gpu[0] + hit_normal[1] * sun_dir_gpu[1] + hit_normal[2] * sun_dir_gpu[2])
            diffuse_intensity = SUN_INTENSITY * light_dot_normal

            shadow_ray_origin[0] = hit_point[0] + hit_normal[0] * shadow_bias_gpu; shadow_ray_origin[1] = hit_point[1] + hit_normal[1] * shadow_bias_gpu; shadow_ray_origin[2] = hit_point[2] + hit_normal[2] * shadow_bias_gpu
            is_occluded = trace_shadow_ray_gpu_device(shadow_ray_origin, sun_dir_gpu, inf_gpu,
                                                      flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered)
            shadow_factor = GEOMETRY_DTYPE(0.0) if is_occluded else GEOMETRY_DTYPE(1.0)

            view_dir[0] = cam_o[0] - hit_point[0]; view_dir[1] = cam_o[1] - hit_point[1]; view_dir[2] = cam_o[2] - hit_point[2]
            view_dir_norm_sq = view_dir[0]**2 + view_dir[1]**2 + view_dir[2]**2
            if view_dir_norm_sq > eps_gpu**2:
                inv_view_norm = GEOMETRY_DTYPE(1.0) / math.sqrt(view_dir_norm_sq)
                view_dir[0] *= inv_view_norm; view_dir[1] *= inv_view_norm; view_dir[2] *= inv_view_norm

            reflect_dot = 2.0 * (hit_normal[0] * sun_dir_gpu[0] + hit_normal[1] * sun_dir_gpu[1] + hit_normal[2] * sun_dir_gpu[2])
            reflect_dir[0] = reflect_dot * hit_normal[0] - sun_dir_gpu[0]; reflect_dir[1] = reflect_dot * hit_normal[1] - sun_dir_gpu[1]; reflect_dir[2] = reflect_dot * hit_normal[2] - sun_dir_gpu[2]

            specular_dot_view = max(0.0, view_dir[0] * reflect_dir[0] + view_dir[1] * reflect_dir[1] + view_dir[2] * reflect_dir[2])
            specular_intensity = ks_gpu * (specular_dot_view ** specular_exponent_gpu)

            final_color_float[0] = base_color_float[0] * (ambient_light_gpu + diffuse_intensity * shadow_factor) + specular_color_gpu[0] * specular_intensity * shadow_factor
            final_color_float[1] = base_color_float[1] * (ambient_light_gpu + diffuse_intensity * shadow_factor) + specular_color_gpu[1] * specular_intensity * shadow_factor
            final_color_float[2] = base_color_float[2] * (ambient_light_gpu + diffuse_intensity * shadow_factor) + specular_color_gpu[2] * specular_intensity * shadow_factor

            final_r = max(0.0, min(1.0, final_color_float[0])); final_g = max(0.0, min(1.0, final_color_float[1])); final_b = max(0.0, min(1.0, final_color_float[2]))
            rgb[i, j, 0] = int(final_r * 255.0); rgb[i, j, 1] = int(final_g * 255.0); rgb[i, j, 2] = int(final_b * 255.0)

        else: # Sky
            rgb[i, j, 0] = default_sky_r; rgb[i, j, 1] = default_sky_g; rgb[i, j, 2] = default_sky_b
            depth[i, j] = inf_gpu; sem[i, j] = sky_label_gpu


# --- 输出函数 (来自 demo, save_combined_obj 可能需要调整以处理原始数据) ---
# save_parameters_json (不变)
def save_parameters_json(filename, params):
    # print(f"    将参数保存到 JSON: {filename}") # 循环中太频繁
    try:
        params_serializable = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray): params_serializable[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)): params_serializable[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)): params_serializable[key] = int(value)
            elif isinstance(value, dict):
                nested_serializable = {}
                for nk, nv in value.items():
                    if isinstance(nv, np.ndarray): nested_serializable[nk] = nv.tolist()
                    elif isinstance(nv, (np.float32, np.float64)): nested_serializable[nk] = float(nv)
                    elif isinstance(nv, (np.int32, np.int64)): nested_serializable[nk] = int(nv)
                    else: nested_serializable[nk] = nv
                params_serializable[key] = nested_serializable
            else: params_serializable[key] = value
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(params_serializable, f, indent=4)
        # print(f"    成功将参数保存到 {filename}")
    except TypeError as e: print(f"    错误: 无法将参数序列化为 JSON ({filename}): {e}")
    except IOError as e: print(f"    错误: 无法写入 JSON 文件 {filename}: {e}")
    except Exception as e: print(f"    保存 JSON 时发生未知错误 ({filename}): {e}")

# save_combined_obj (可能需要调整以使用原始 v0s, labels 等进行分组)
# 为了简化，暂时移除 OBJ 保存，因为需要更复杂的处理来映射回原始标签
# def save_combined_obj(...):
#    pass


# --- 主执行流程 ---
def main():
    global _GPU_AVAILABLE # 允许修改
    total_t0 = _now()
    print(f"--- 开始执行: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    # --- 输出目录设置 ---
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outd_base = os.path.join(BASE_DIR, 'output_hk_window', ts) # 主输出目录
    try:
        os.makedirs(outd_base, exist_ok=True)
        print(f"主输出目录: {outd_base}")
    except OSError as e:
        print(f"错误: 无法创建主输出目录 {outd_base}: {e}")
        return

    # --- [1] 场景生成 ---
    print('\n[1] 构建香港场景...')
    t0_scene = _now()
    # build_hk_scene 返回原始几何数据和建筑信息
    v0s, e1s, e2s, normals, labels, colors, prim_indices, building_data_for_views = build_hk_scene()
    num_triangles = len(v0s)
    log_step('场景构建', t0_scene)
    if num_triangles == 0:
        print("错误: 场景构建结果为 0 个三角形。正在退出。")
        return
    print(f"    场景包含 {num_triangles} 个三角形。")

    # --- [2] 构建 BVH ---
    print('\n[2] 构建 BVH...')
    t0_bvh = _now()
    flat_nodes, ordered_prim_indices = build_bvh(v0s, e1s, e2s, prim_indices)
    log_step('BVH 构建', t0_bvh)
    if flat_nodes.shape[0] == 0 and num_triangles > 0:
        print("警告: BVH 构建失败或为空，但场景中有三角形。渲染可能会非常慢或失败。")
    elif flat_nodes.shape[0] == 0 and num_triangles == 0:
         print("场景和 BVH 均为空。")
         # 也许应该在这里退出？

    # --- [3] 重新排序几何数据 ---
    print('\n[3] 根据 BVH 重新排序几何数据...')
    t0_reorder = _now()
    # 确保 BVH 构建成功且索引数量匹配
    if flat_nodes.shape[0] > 0 and len(ordered_prim_indices) == num_triangles:
        try:
            v0s_reordered = v0s[ordered_prim_indices]
            e1s_reordered = e1s[ordered_prim_indices]
            e2s_reordered = e2s[ordered_prim_indices]
            normals_reordered = normals[ordered_prim_indices]
            labels_reordered = labels[ordered_prim_indices]
            colors_reordered = colors[ordered_prim_indices]
            print("    几何数据重新排序成功。")
        except IndexError as e:
             print(f"错误: 几何数据重新排序期间发生索引错误: {e}。回退到原始顺序。")
             v0s_reordered, e1s_reordered, e2s_reordered = v0s, e1s, e2s
             normals_reordered, labels_reordered, colors_reordered = normals, labels, colors
    else:
        if num_triangles > 0:
             print(f"警告: BVH 排序索引计数 ({len(ordered_prim_indices)}) 与三角形计数 ({num_triangles}) 不匹配，或 BVH 为空。使用原始顺序。")
        v0s_reordered, e1s_reordered, e2s_reordered = v0s, e1s, e2s
        normals_reordered, labels_reordered, colors_reordered = normals, labels, colors
    log_step('几何数据重新排序', t0_reorder)

    # --- [4] 生成视点 ---
    print('\n[4] 生成窗口视点...')
    t0_views = _now()
    # 选择 5 栋建筑进行渲染
    viewpoints = generate_window_viewpoints(building_data_for_views, num_target_buildings=5)
    log_step('视点生成', t0_views)
    if not viewpoints:
        print("错误: 未能生成任何视点。检查建筑数据和视点生成逻辑。正在退出。")
        return

    # --- [5] 循环渲染每个视点 ---
    print(f'\n[5] 开始渲染 {len(viewpoints)} 个视点...')
    W, H = 1024, 768 # 渲染分辨率 (可以调小以加快测试)
    fov_degrees = 70.0 # 窗口的视野范围
    fov_radians = np.deg2rad(fov_degrees)
    aspect_ratio = float(W) / float(H)
    # 假设相机距离屏幕平面为 1 个单位
    screen_h_cam = GEOMETRY_DTYPE(2.0 * np.tan(fov_radians / 2.0))
    screen_w_cam = GEOMETRY_DTYPE(screen_h_cam * aspect_ratio)

    # 预上传 GPU 数据 (如果使用 GPU)
    d_flat_nodes, d_v0s, d_e1s, d_e2s, d_normals, d_labels, d_colors = None, None, None, None, None, None, None
    d_sun_dir, d_specular_color = None, None
    # 只有当 BVH 和几何体都有效时才尝试 GPU
    use_gpu_rendering = _GPU_AVAILABLE and flat_nodes.shape[0] > 0 and len(v0s_reordered) > 0

    if use_gpu_rendering:
        print("    预上传数据到 GPU...")
        try:
            t_upload_start = _now()
            d_flat_nodes = cuda.to_device(flat_nodes)
            d_v0s = cuda.to_device(v0s_reordered)
            d_e1s = cuda.to_device(e1s_reordered)
            d_e2s = cuda.to_device(e2s_reordered)
            d_normals = cuda.to_device(normals_reordered)
            d_labels = cuda.to_device(labels_reordered)
            d_colors = cuda.to_device(colors_reordered)
            d_sun_dir = cuda.to_device(SUN_DIRECTION)
            d_specular_color = cuda.to_device(SPECULAR_COLOR)
            log_step('GPU 数据预上传', t_upload_start)
        except Exception as e:
            print(f"    GPU 数据上传失败: {e}。将回退到 CPU 渲染。")
            use_gpu_rendering = False
            _GPU_AVAILABLE = False # 禁用 GPU

    # 渲染循环
    total_render_time = 0
    for vp_idx, vp in enumerate(viewpoints):
        t0_vp_loop = _now()
        # 简化输出文件名中的 face 信息
        face_info = f"Seg{vp['face_segment_index']}"
        print(f"\n  --- 渲染视点 {vp_idx + 1}/{len(viewpoints)} (Building: {vp['building_id']}, Floor: {vp['floor']}, {face_info}) ---")
        t0_vp_render = _now()

        # 设置当前视点的相机参数
        cam_o = vp['cam_o'].astype(GEOMETRY_DTYPE)
        cam_dir = vp['cam_dir'].astype(GEOMETRY_DTYPE) # 已经是归一化的
        cam_up_vec = vp['cam_up_vec'].astype(GEOMETRY_DTYPE)

        # 计算相机框架
        right = np.cross(cam_dir, cam_up_vec)
        norm_right = np.linalg.norm(right)
        if norm_right < INTERSECTION_EPSILON:
            print("    警告: 相机方向与向上向量平行，调整右向量。")
            # *** 重要: 检查 Z 向上还是 Y 向上 ***
            if abs(cam_dir[2]) > 1.0 - INTERSECTION_EPSILON: # 假设 Z 向上
                 alt_up = np.array([0.0, 1.0, 0.0], dtype=GEOMETRY_DTYPE) # 使用 Y 作为替代
                 right = np.cross(alt_up, cam_dir)
            else: # 水平但向上向量无效
                 right = np.array([1.0, 0.0, 0.0], dtype=GEOMETRY_DTYPE) # 默认 X
            norm_right = np.linalg.norm(right)
            if norm_right < INTERSECTION_EPSILON:
                 print(f"错误: 无法为视点 {vp_idx+1} 计算有效右向量，跳过此视点。")
                 continue # 跳过这个视点
        right /= norm_right
        up = np.cross(right, cam_dir) # 重新计算精确的向上向量

        # 创建此视点的输出子目录
        # 使用 building_id 创建子目录，避免 ID 过长或包含非法字符
        building_dir_name = "".join(c for c in vp['building_id'] if c.isalnum() or c in ('-', '_')).rstrip()
        vp_out_dir = os.path.join(outd_base, building_dir_name)
        try:
             os.makedirs(vp_out_dir, exist_ok=True)
        except OSError as e:
             print(f"    错误: 无法创建视点输出目录 {vp_out_dir}: {e}。跳过此视点。")
             continue

        base_filename = f"F{vp['floor']:03d}_{face_info}" # 移除时间戳，因为主目录有时间戳

        # 执行光线追踪
        rgb, depth, sem_lbl, pts = None, None, None, None
        render_device = "N/A"
        if use_gpu_rendering:
            render_device = "GPU"
            # print("    使用 GPU 渲染...")
            try:
                # 在循环内分配 GPU 输出数组可能更安全，避免潜在的内存问题
                t_gpu_alloc_start = _now()
                d_rgb = cuda.device_array((H, W, 3), dtype=COLOR_DTYPE)
                d_depth = cuda.to_device(np.full((H, W), INF, dtype=DEPTH_DTYPE))
                d_sem = cuda.to_device(np.full((H, W), SKY_LABEL, dtype=LABEL_DTYPE))
                d_pts = cuda.to_device(np.full((H, W, 3), np.nan, dtype=POINT_DTYPE))
                cuda.synchronize() # 确保分配完成
                # log_step('  GPU 输出内存分配', t_gpu_alloc_start)

                threads_per_block = (16, 16)
                blocks_per_grid_x = (W + threads_per_block[1] - 1) // threads_per_block[1]
                blocks_per_grid_y = (H + threads_per_block[0] - 1) // threads_per_block[0]
                blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)

                # print(f"    启动 CUDA 内核: Grid={blocks_per_grid}, Block={threads_per_block}")
                t_kernel_start = _now()
                raytrace_cuda_bvh_kernel[blocks_per_grid, threads_per_block](
                    d_flat_nodes, d_v0s, d_e1s, d_e2s,
                    d_normals, d_labels, d_colors,
                    cam_o, cam_dir, right, up, # CPU 参数
                    d_sun_dir, d_specular_color, # GPU 光照参数
                    screen_w_cam, screen_h_cam, W, H,
                    d_rgb, d_depth, d_sem, d_pts # GPU 输出数组
                )
                cuda.synchronize()
                kernel_time = _now() - t_kernel_start
                # print(f"    GPU 内核执行耗时: {kernel_time:.3f}s")

                t_download_start = _now()
                rgb = d_rgb.copy_to_host()
                depth = d_depth.copy_to_host()
                sem_lbl = d_sem.copy_to_host()
                pts = d_pts.copy_to_host()
                # log_step('  GPU 数据下载', t_download_start)

                # 清理 GPU 内存 (可选，但有助于大型循环)
                del d_rgb, d_depth, d_sem, d_pts
                # cuda.current_context().memory_manager.deallocations.clear() # 更激进的清理

            except Exception as e:
                print(f"    GPU 渲染视点 {vp_idx+1} 失败: {e}。回退到 CPU。")
                use_gpu_rendering = False # 对后续视点也禁用 GPU
                _GPU_AVAILABLE = False
                rgb, depth, sem_lbl, pts = None, None, None, None # 重置
                render_device = "GPU_Failed"


        # 如果 GPU 失败或未启用，则使用 CPU
        if not use_gpu_rendering or render_device == "GPU_Failed":
             render_device = "CPU"
             print("    使用 CPU 渲染...")
             t_cpu_start = _now()
             rgb, depth, sem_lbl, pts = raytrace_cpu_bvh(
                 flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
                 normals_reordered, labels_reordered, colors_reordered,
                 cam_o, cam_dir, right, up,
                 screen_w_cam, screen_h_cam, W, H
             )
             log_step('  CPU 光线追踪执行', t_cpu_start)

        vp_render_time = _now() - t0_vp_render
        total_render_time += vp_render_time
        print(f"    视点 {vp_idx + 1} 渲染完成 (使用 {render_device})。")
        log_step(f'  视点 {vp_idx + 1} 渲染', t0_vp_render)


        # --- [6] 保存当前视点的输出图像 ---
        # print(f"    保存视点 {vp_idx + 1} 的输出图像...")
        t0_save_img = _now()

        # 保存 RGB 视图
        if rgb is not None:
            fn_view = os.path.join(vp_out_dir, f'{base_filename}_view.png')
            try: Image.fromarray(rgb).save(fn_view)
            except Exception as e: print(f"    保存视图图像失败 {fn_view}: {e}")
        # 保存深度图
        if depth is not None:
            fn_depth = os.path.join(vp_out_dir, f'{base_filename}_depth.png')
            try:
                valid_depth = depth[np.isfinite(depth)]
                if len(valid_depth) > 0:
                    dmin = np.min(valid_depth); dmax_vis = np.percentile(valid_depth, 99.5)
                    dmax_vis = min(dmax_vis, 1000.0) # 限制可视化最大值
                    if dmax_vis <= dmin: dmax_vis = dmin + 1.0
                    scale = max(dmax_vis - dmin, INTERSECTION_EPSILON)
                    depth_normalized = (depth - dmin) / scale
                    depth_clipped = np.clip(depth_normalized * 254.0, 0, 254)
                    dmap = np.where(np.isfinite(depth), depth_clipped, 255).astype(np.uint8)
                    Image.fromarray(dmap, 'L').save(fn_depth)
                else: Image.new('L', (W, H), 0).save(fn_depth) # 保存黑色图像
            except Exception as e: print(f"    保存深度图像失败 {fn_depth}: {e}")
        # 保存语义图
        if sem_lbl is not None:
            fn_sem = os.path.join(vp_out_dir, f'{base_filename}_semantic.png')
            try:
                sem_img = np.zeros((H, W, 3), dtype=COLOR_DTYPE)
                unique_labels_render = np.unique(sem_lbl)
                for label_id in unique_labels_render:
                     sem_img[sem_lbl == label_id] = C.get(label_id, PALETTE_BACKGROUND_COLOR)
                Image.fromarray(sem_img).save(fn_sem)
            except Exception as e: print(f"    保存语义图像失败 {fn_sem}: {e}")

        # log_step(f'  视点 {vp_idx + 1} 图像导出', t0_save_img) # 太频繁

        # --- [7] 保存当前视点的参数 ---
        # print(f"    保存视点 {vp_idx + 1} 的参数...")
        # t0_json = _now()
        fn_json = os.path.join(vp_out_dir, f'{base_filename}_params.json')
        vp_params = {
            "timestamp": ts,
            "viewpoint_index": vp_idx,
            "building_id": vp['building_id'],
            "floor": vp['floor'],
            "face_segment_index": vp['face_segment_index'],
            "resolution": {"width": W, "height": H},
            "fov_degrees": fov_degrees,
            "camera": {
                "origin": cam_o, "direction": cam_dir, "world_up": cam_up_vec,
                "frame_right": right, "frame_up": up,
                "screen_width_world": screen_w_cam, "screen_height_world": screen_h_cam,
            },
            "render_device": render_device,
            "render_time_seconds": vp_render_time,
        }
        save_parameters_json(fn_json, vp_params)
        # log_step(f'  视点 {vp_idx + 1} JSON 参数导出', t0_json) # 太频繁

        log_step(f'总计视点 {vp_idx + 1} 处理', t0_vp_loop)


    # --- 完成 ---
    print('\n[完成]')
    total_time = _now() - total_t0
    avg_render_time = total_render_time / len(viewpoints) if viewpoints else 0
    print(f"总执行时间: {total_time:.2f}s")
    print(f"渲染视点数: {len(viewpoints)}")
    print(f"平均渲染时间: {avg_render_time:.2f}s / 视点")
    print(f"输出已保存至: {outd_base}")


# --- 脚本入口点 ---
if __name__ == '__main__':
    # 添加警告过滤器，避免 Shapely/Pyproj 的一些常见但无害的警告刷屏
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module='scipy') # 忽略 scipy 版本警告
    # warnings.filterwarnings("ignore", message="invalid value encountered in cast") # Numba 可能会产生
    main()
# ```
# ```
#