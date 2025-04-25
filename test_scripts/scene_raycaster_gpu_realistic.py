# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import math # 数学运算所需
import random # 随机选择所需
import json # JSON 导出所需

import numpy as np
from PIL import Image
from numba import njit, prange, cuda

# -- GPU 可用性检查 --------------------------------------------------
_GPU_AVAILABLE = False # 默认为 False
try:
    # 检查 CUDA 设备是否可用且可以创建上下文
    if cuda.is_available():
        # 尝试检测并获取设备名称
        try:
            cuda.detect()  # 如果成功，将打印设备信息
            device = cuda.get_current_device()
            print(f"CUDA 可用: True")
            print(f"使用 GPU: {device.name.decode()}")
            _GPU_AVAILABLE = True
        except Exception as e_detect:
            print(f"检测 CUDA 设备或获取名称时出错: {e_detect}")
            print("CUDA 可能仍然可用，但信息不完整。")
            _GPU_AVAILABLE = True # 无论如何都尝试一下
    else:
        print("CUDA 不可用: 未检测到兼容设备或驱动程序。")

except ImportError:
    print("Numba CUDA 扩展未安装或导入失败。")
except Exception as e:
    print(f"CUDA 初始化期间发生未知错误: {e}")


# --- 常量定义 ---
GEOMETRY_DTYPE = np.float32
INDEX_DTYPE = np.int32
COLOR_DTYPE = np.uint8
LABEL_DTYPE = np.int32
DEPTH_DTYPE = np.float32
POINT_DTYPE = np.float32
BVH_NODE_DTYPE = np.float32 # BVH 节点数据类型 (用于 AABB)
INF = GEOMETRY_DTYPE(np.inf)

# --- 相交测试的 Epsilon 和 Bias ---
INTERSECTION_EPSILON = GEOMETRY_DTYPE(1e-5) # 浮点比较的 Epsilon
# --- 修改: 增大 Shadow Bias 以减少阴影痤疮 ---
SHADOW_BIAS = GEOMETRY_DTYPE(1e-3)          # 阴影光线的偏移量，避免自相交 (从 1e-4 增大)

# --- 光照参数 ---
SUN_DIRECTION = np.array([0.6, 0.8, -0.4], dtype=GEOMETRY_DTYPE) # 调整后的太阳方向
SUN_DIRECTION /= np.linalg.norm(SUN_DIRECTION) # 归一化太阳方向
SUN_INTENSITY = GEOMETRY_DTYPE(1.0)
AMBIENT_LIGHT = GEOMETRY_DTYPE(0.2) # 稍微增加的环境光

# --- Phong 着色参数 (用于提升真实感) ---
SPECULAR_COLOR = np.array([1.0, 1.0, 1.0], dtype=GEOMETRY_DTYPE) # 白色高光
SPECULAR_EXPONENT = GEOMETRY_DTYPE(32.0) # 光泽度因子
Ks = GEOMETRY_DTYPE(0.4) # 镜面反射系数

# --- BVH 参数 ---
BVH_MAX_LEAF_SIZE = 4 # BVH 叶节点中的最大图元数
BVH_NODE_FIELDS = 8   # 每个 BVH 节点的字段数 (min_x,y,z, max_x,y,z, info1, info2)

# --- 语义标签和颜色 ---
SKY_LABEL = 7 # 专门为天空定义的新标签
LABEL_MAP = {
    0: "ground",
    1: "building",
    2: "road",
    3: "water",
    4: "skyscraper", # 包括棱柱体
    5: "tree",
    6: "mountain",
    SKY_LABEL: "sky"
}
DEFAULT_SKY_COLOR = np.array([135, 206, 235], dtype=COLOR_DTYPE) # 淡蓝色天空

# 全局颜色映射 (将标签 ID 映射到 RGB 颜色)
C = {
    0: np.array([80, 160, 80], dtype=COLOR_DTYPE),   # 0: 地面/草地 (更绿)
    1: np.array([200, 200, 200], dtype=COLOR_DTYPE), # 1: 建筑
    2: np.array([220, 180, 50], dtype=COLOR_DTYPE),  # 2: 道路
    3: np.array([90, 140, 210], dtype=COLOR_DTYPE),  # 3: 水面
    4: np.array([210, 80, 80], dtype=COLOR_DTYPE),   # 4: 摩天楼/特殊建筑
    5: np.array([0, 140, 0], dtype=COLOR_DTYPE),     # 5: 树木
    6: np.array([120, 120, 120], dtype=COLOR_DTYPE), # 6: 山脉
    SKY_LABEL: DEFAULT_SKY_COLOR                     # 7: 天空
}
# 语义调色板图像的背景色 (用于未分配标签的情况，应该不常发生)
PALETTE_BACKGROUND_COLOR = (0, 0, 0)

# --- 工具函数 ---
def _now() -> float:
    """高精度时间戳。"""
    return time.perf_counter()

def log_step(title: str, t0: float) -> None:
    """记录一个步骤的持续时间。"""
    print(f"    {title} 耗时 {_now() - t0:.3f}s")

# --- 可复现的随机数生成器 ---
RAND = np.random.default_rng(seed=0)

# --- 几何体创建函数 ---
# (几何体创建逻辑无需重大更改，保留以保持完整性)
# (添加了注释以提高清晰度)

def create_box(center, size):
    """创建长方体的顶点和面。"""
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    verts = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ], dtype=GEOMETRY_DTYPE)
    # 面定义为顶点索引的元组 (0-based)
    faces = [
        (0, 1, 2), (0, 2, 3), # 底面 (2 个三角形)
        (4, 6, 5), (4, 7, 6), # 顶面 (2 个三角形)
        (0, 4, 5), (0, 5, 1), # 前面 (2 个三角形)
        (3, 2, 6), (3, 6, 7), # 后面 (2 个三角形)
        (0, 3, 7), (0, 7, 4), # 左面 (2 个三角形)
        (1, 5, 6), (1, 6, 2), # 右面 (2 个三角形)
    ] # 总共 12 个三角形
    return verts, faces

def create_prism(center, height, radius, sides=5):
    """创建棱柱体的顶点和面。"""
    cx, cy, cz = center
    half_h = height / 2.0
    verts = []
    # 为每条边创建底部和顶部顶点
    for i in range(sides):
        angle = 2.0 * np.pi * i / sides
        x = cx + radius * np.cos(angle)
        z = cz + radius * np.sin(angle)
        verts.append((x, cy - half_h, z)) # 底部顶点
        verts.append((x, cy + half_h, z)) # 顶部顶点
    verts = np.asarray(verts, dtype=GEOMETRY_DTYPE)

    faces = []
    # 创建侧面 (由两个三角形组成的四边形)
    for i in range(sides):
        i0_bottom = 2 * i
        i1_bottom = (2 * (i + 1)) % (2 * sides) # 对最后一个侧面进行环绕处理
        i0_top = i0_bottom + 1
        i1_top = i1_bottom + 1
        # 侧面四边形的三角形 1
        faces.append((i0_bottom, i1_bottom, i1_top))
        # 侧面四边形的三角形 2
        faces.append((i0_bottom, i1_top, i0_top))
    # 注意: 这个棱柱体没有顶部/底部封盖面。如果需要可以添加。
    return verts, faces # 2 * sides 个三角形

def create_skyscraper(center, base_width, height, levels=8, taper=0.85):
    """创建多层、逐渐变细的摩天楼。"""
    verts, faces = [], []
    cx, cy, cz = center
    current_y = cy # 从基础 y 坐标开始
    current_width = base_width
    segment_height = height / float(levels)

    for level in range(levels):
        segment_center_y = current_y + segment_height / 2.0
        segment_center = (cx, segment_center_y, cz)
        segment_size = (current_width, segment_height, current_width)

        # 为当前层创建长方体
        v_seg, f_seg = create_box(segment_center, segment_size)

        # 添加分段的顶点和面，调整索引
        vertex_offset = len(verts)
        verts.extend(v_seg)
        faces.extend([(a + vertex_offset, b + vertex_offset, c + vertex_offset)
                      for a, b, c in f_seg])

        # 为下一层更新
        current_y += segment_height
        current_width *= taper # 应用锥度

    return np.asarray(verts, dtype=GEOMETRY_DTYPE), faces # 12 * levels 个三角形

def create_cylinder(center, height, radius, *, segments: int = 16):
    """创建圆柱体的顶点和面 (无封盖)。"""
    cx, cy, cz = center
    verts = []
    half_h = height / 2.0
    # 创建底部和顶部顶点
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        x = cx + radius * np.cos(angle)
        z = cz + radius * np.sin(angle)
        verts.append((x, cy - half_h, z)) # 底部顶点
        verts.append((x, cy + half_h, z)) # 顶部顶点
    verts = np.asarray(verts, dtype=GEOMETRY_DTYPE)

    faces = []
    # 创建侧面
    for i in range(segments):
        i0_bottom = 2 * i
        i1_bottom = (i0_bottom + 2) % (2 * segments) # 环绕处理
        i0_top = i0_bottom + 1
        i1_top = i1_bottom + 1
        # 三角形 1
        faces.append((i0_bottom, i1_bottom, i1_top))
        # 三角形 2
        faces.append((i0_bottom, i1_top, i0_top))
    # 注意: 没有顶部/底部封盖。
    return verts, faces # 2 * segments 个三角形

def create_cone_tree(base_center, height, radius, *, segments: int = 10):
    """创建简化的树 (圆柱树干 + 圆锥树冠)。"""
    cx, cy, cz = base_center
    trunk_height = height * 0.3
    trunk_radius = radius * 0.15
    crown_height = height * 0.7
    crown_radius = radius

    # 创建树干
    trunk_center_y = cy + trunk_height / 2.0
    trunk_segments = max(6, segments // 2)
    trunk_verts, trunk_faces = create_cylinder(
        center=(cx, trunk_center_y, cz),
        height=trunk_height, radius=trunk_radius, segments=trunk_segments)
    num_trunk_verts = len(trunk_verts)

    # 创建树冠 (圆锥)
    crown_base_y = cy + trunk_height
    apex_y = crown_base_y + crown_height
    apex_vertex = np.array([cx, apex_y, cz], dtype=GEOMETRY_DTYPE)

    cone_base_verts = []
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        x = cx + crown_radius * np.cos(angle)
        z = cz + crown_radius * np.sin(angle)
        cone_base_verts.append((x, crown_base_y, z))
    cone_base_verts = np.asarray(cone_base_verts, dtype=GEOMETRY_DTYPE)

    # 合并圆锥的顶点和基座顶点
    cone_verts = np.vstack((apex_vertex.reshape(1, 3), cone_base_verts))
    num_cone_verts = len(cone_verts)

    cone_faces = []
    apex_idx_local = 0 # 圆锥顶点在 cone_verts 中的索引
    for i in range(segments):
        base_idx_local_1 = i + 1 # +1 因为顶点是索引 0
        base_idx_local_2 = (i + 1) % segments + 1 # 环绕处理, +1 是顶点的偏移量
        cone_faces.append((apex_idx_local, base_idx_local_1, base_idx_local_2))

    # 合并树干和树冠的几何体
    all_verts = np.vstack((trunk_verts, cone_verts))

    # 调整圆锥面索引以考虑树干顶点
    adjusted_cone_faces = []
    for face in cone_faces:
        adjusted_face = (face[0] + num_trunk_verts,
                         face[1] + num_trunk_verts,
                         face[2] + num_trunk_verts)
        adjusted_cone_faces.append(adjusted_face)

    all_faces = trunk_faces + adjusted_cone_faces
    # 注意: 圆锥底部未闭合。
    return all_verts, all_faces

def create_mountain_range(x_start, x_end, z_pos, *, segs=200, depth=25, h_min=18, h_max=50):
    """使用噪声创建山脉段。"""
    xs = np.linspace(x_start, x_end, segs + 1, dtype=GEOMETRY_DTYPE)

    # 生成高度变化的噪声
    num_noise_points = max(5, segs // 10) # 较少的点以获得更平滑的噪声
    noise_xs = np.linspace(x_start, x_end, num_noise_points)
    noise_ys_raw = RAND.uniform(-1.0, 1.0, noise_xs.shape)
    # 插值噪声以匹配段数
    noise_ys = np.interp(xs, noise_xs, noise_ys_raw)
    # 将噪声范围 [-1, 1] 映射到高度范围 [h_min, h_max]
    base_heights = np.interp(noise_ys, (-1.0, 1.0), (h_min, h_max))

    verts = []
    faces = []
    zero_y = GEOMETRY_DTYPE(0.0)
    z_pos_dtype = GEOMETRY_DTYPE(z_pos)
    depth_dtype = GEOMETRY_DTYPE(depth)

    # 创建山脉的段 (类似挤压轮廓)
    for i in range(segs):
        x0, x1 = xs[i], xs[i + 1]
        h0, h1 = base_heights[i], base_heights[i + 1]
        h0_dtype, h1_dtype = GEOMETRY_DTYPE(h0), GEOMETRY_DTYPE(h1)

        # 一个段的顶点 (前底、前顶、后底、后顶)
        v_seg = [
            (x0, zero_y,   z_pos_dtype),           # 0: 前底左
            (x1, zero_y,   z_pos_dtype),           # 1: 前底右
            (x0, h0_dtype, z_pos_dtype),           # 2: 前顶左
            (x1, h1_dtype, z_pos_dtype),           # 3: 前顶右
            (x0, zero_y,   z_pos_dtype + depth_dtype), # 4: 后底左
            (x1, zero_y,   z_pos_dtype + depth_dtype), # 5: 后底右
            (x0, h0_dtype, z_pos_dtype + depth_dtype), # 6: 后顶左
            (x1, h1_dtype, z_pos_dtype + depth_dtype)  # 7: 后顶右
        ]
        idx0 = len(verts) # 该段顶点的起始索引
        verts.extend(v_seg)

        # 创建带偏移索引的面的函数
        f = lambda a, b, c: (idx0 + a, idx0 + b, idx0 + c)

        # 为该段创建面 (四边形 -> 每个 2 个三角形)
        faces.append(f(0, 1, 3)) # 前面 三角形 1
        faces.append(f(0, 3, 2)) # 前面 三角形 2
        faces.append(f(4, 6, 7)) # 后面 三角形 1
        faces.append(f(4, 7, 5)) # 后面 三角形 2
        faces.append(f(0, 4, 6)) # 左面 三角形 1
        faces.append(f(0, 6, 2)) # 左面 三角形 2
        faces.append(f(1, 3, 7)) # 右面 三角形 1
        faces.append(f(1, 7, 5)) # 右面 三角形 2
        faces.append(f(2, 6, 7)) # 顶面 三角形 1
        faces.append(f(2, 7, 3)) # 顶面 三角形 2
        # 注意: 未创建底面。

    return np.asarray(verts, dtype=GEOMETRY_DTYPE), faces # 10 * segs 个三角形

# --- 场景构建 ---
def build_scene():
    """构建整个场景几何体，分配标签和颜色。"""
    # 在转换为 numpy 数组之前存储三角形数据的列表
    tris_vertices = [] # 存储每个三角形的 (v0, v1, v2) 元组
    tris_labels = []   # 存储每个三角形的语义标签
    tris_colors = []   # 存储每个三角形的基础颜色

    # 目标对象数量 (可调整以改变场景复杂度)
    target_num_buildings = 5000
    target_num_skyscrapers = 500
    target_num_prisms = 500
    target_num_trees = 10000
    mountain_segs1 = 250
    mountain_segs2 = 200

    print(f"    目标对象: 建筑={target_num_buildings}, "
          f"摩天楼={target_num_skyscrapers}, 棱柱={target_num_prisms}, "
          f"树木={target_num_trees}")

    # 将对象添加到场景列表的辅助函数
    def add_object(verts, faces, label, color_map):
        """将来自 verts/faces 的三角形添加到场景列表中。"""
        color = color_map[label]
        for face_indices in faces:
            # 确保索引有效
            if all(idx < len(verts) for idx in face_indices):
                if len(face_indices) == 3: # 只处理三角形
                    a, b, c = face_indices
                    # 为此三角形附加顶点、标签和颜色
                    tris_vertices.append((verts[a], verts[b], verts[c]))
                    tris_labels.append(label)
                    tris_colors.append(color)
            # else: print(f"警告: 标签 {label} 中存在无效面索引") # 可选警告

    # --- 场景尺寸和地面 ---
    world_radius = 400.0
    ground_size = (world_radius * 2.2, 0.01, world_radius * 2.2)
    min_coord = -world_radius
    max_coord = world_radius
    ground_y = -0.1 # 地平面的 Y 水平
    base_y = ground_y + 0.01 # 放置在地面上的对象的基础 Y

    print("    生成地面...")
    ground_center = (0.0, ground_y + 0.005, 0.0) # 略高于 ground_y
    g_v, g_f = create_box(ground_center, ground_size)
    add_object(g_v, g_f, 0, C) # 标签 0: ground

    # --- 湖泊 ---
    print("    生成湖泊...")
    lake_y = base_y + 0.02 # 略高于 base_y
    lake_center = (150.0, lake_y, 200.0)
    lake_size = (120.0, 0.01, 90.0)
    lake_v, lake_f = create_box(lake_center, lake_size)
    # 只添加湖泊长方体的顶面 (来自 create_box 的索引 2 和 3)
    add_object(lake_v, [lake_f[i] for i in [2, 3]], 3, C) # 标签 3: water
    # 定义湖泊边界用于碰撞检查
    lake_x_min = lake_center[0] - lake_size[0] / 2.0
    lake_x_max = lake_center[0] + lake_size[0] / 2.0
    lake_z_min = lake_center[2] - lake_size[2] / 2.0
    lake_z_max = lake_center[2] + lake_size[2] / 2.0

    # --- 道路网格 ---
    print("    生成道路网格...")
    road_y = base_y + 0.03 # 略高于湖泊
    road_w = 5.0 # 道路宽度
    road_h = 0.01 # 道路高度 (厚度)
    num_roads = 15
    # 创建网格线的坐标
    grid_coords = np.linspace(min_coord * 0.9, max_coord * 0.9, num_roads, dtype=GEOMETRY_DTYPE)
    road_len = world_radius * 2.0 + road_w # 确保道路跨越区域

    # 沿 X 轴添加道路
    for x in grid_coords:
        v, f = create_box((x, road_y, 0.0), (road_w, road_h, road_len))
        add_object(v, f, 2, C) # 标签 2: road
    # 沿 Z 轴添加道路
    for z in grid_coords:
        v, f = create_box((0.0, road_y, z), (road_len, road_h, road_w))
        add_object(v, f, 2, C) # 标签 2: road

    # --- 建筑集群 ---
    # 定义建筑集群的中心 (x, z) 和半径
    clusters = [
        (-200.0, -150.0, 80.0), (50.0, -250.0, 90.0), (250.0, -50.0, 70.0),
        (100.0, 180.0, 100.0), (-180.0, 220.0, 60.0)
    ]
    num_clusters = len(clusters)
    print(f"    定义了 {num_clusters} 个建筑集群。")

    # 在集群半径内获取随机点的辅助函数
    def get_random_point_in_cluster(cluster_info, bias_center=False):
        center_x, center_z, radius = cluster_info
        angle = RAND.uniform(0.0, 2.0 * np.pi)
        # 对某些建筑 (例如摩天楼) 向中心偏移
        max_dist_factor = 0.6 if bias_center else 1.0
        # 使用 sqrt 以获得更均匀的面积分布
        distance = np.sqrt(RAND.uniform(0.0, max_dist_factor)) * radius
        x = center_x + distance * np.cos(angle)
        z = center_z + distance * np.sin(angle)
        return x, z

    # --- 放置建筑、摩天楼、棱柱体 ---
    building_count = 0
    skyscraper_count = 0
    prism_count = 0
    tree_count = 0

    # 放置普通建筑
    print("    放置普通建筑...")
    attempts = 0
    max_attempts_building = target_num_buildings * 5 # 允许比目标更多的尝试次数
    while building_count < target_num_buildings and attempts < max_attempts_building:
        attempts += 1
        chosen_cluster_info = random.choice(clusters)
        x, z = get_random_point_in_cluster(chosen_cluster_info, bias_center=False)

        # 碰撞检查
        is_in_lake = (lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max)
        if is_in_lake: continue

        is_on_road = False
        road_clearance = road_w # 距离道路中心所需的间隙
        for gx in grid_coords:
            if abs(x - gx) < road_clearance + RAND.uniform(1.0, 3.0): # 添加一些随机缓冲区
                is_on_road = True
                break
        if not is_on_road:
            for gz in grid_coords:
                if abs(z - gz) < road_clearance + RAND.uniform(1.0, 3.0):
                    is_on_road = True
                    break
        if is_on_road: continue

        # 如果检查通过，创建并添加建筑
        w = RAND.uniform(6.0, 15.0) # 随机宽度/深度
        h = RAND.uniform(10.0, 25.0) # 随机高度
        center_y = base_y + h / 2.0
        v, f = create_box((x, center_y, z), (w, h, w))
        add_object(v, f, 1, C) # 标签 1: building
        building_count += 1

    if attempts >= max_attempts_building:
        print(f"    警告: 达到最大尝试次数。放置了 {building_count}/{target_num_buildings} 个建筑。")

    # 放置摩天楼 (逻辑类似，参数/偏移不同)
    print("    放置摩天楼...")
    attempts = 0
    max_attempts_skyscraper = target_num_skyscrapers * 5
    while skyscraper_count < target_num_skyscrapers and attempts < max_attempts_skyscraper:
        attempts += 1
        chosen_cluster_info = random.choice(clusters)
        # 向集群中心偏移放置位置
        x, z = get_random_point_in_cluster(chosen_cluster_info, bias_center=True)

        # 碰撞检查 (与建筑类似，可能间隙更大)
        is_in_lake = (lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max)
        if is_in_lake: continue
        is_on_road = False
        road_clearance = road_w * 1.2 # 稍大的间隙
        for gx in grid_coords:
            if abs(x - gx) < road_clearance: is_on_road = True; break
        if not is_on_road:
            for gz in grid_coords:
                if abs(z - gz) < road_clearance: is_on_road = True; break
        if is_on_road: continue

        # 创建摩天楼
        base_w = RAND.uniform(8.0, 14.0)
        h = RAND.uniform(50.0, 90.0)
        levels = RAND.integers(8, 15)
        taper = RAND.uniform(0.85, 0.98)
        v, f = create_skyscraper((x, base_y, z), base_w, h, levels=levels, taper=taper)
        add_object(v, f, 4, C) # 标签 4: skyscraper
        skyscraper_count += 1

    if attempts >= max_attempts_skyscraper:
        print(f"    警告: 达到最大尝试次数。放置了 {skyscraper_count}/{target_num_skyscrapers} 个摩天楼。")

    # 放置棱柱建筑
    print("    放置棱柱建筑...")
    attempts = 0
    max_attempts_prism = target_num_prisms * 5
    while prism_count < target_num_prisms and attempts < max_attempts_prism:
        attempts += 1
        chosen_cluster_info = random.choice(clusters)
        x, z = get_random_point_in_cluster(chosen_cluster_info, bias_center=False)

        # 碰撞检查
        is_in_lake = (lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max)
        if is_in_lake: continue
        is_on_road = False
        road_clearance = road_w * 1.1
        for gx in grid_coords:
            if abs(x - gx) < road_clearance: is_on_road = True; break
        if not is_on_road:
            for gz in grid_coords:
                if abs(z - gz) < road_clearance: is_on_road = True; break
        if is_on_road: continue

        # 创建棱柱体
        r = RAND.uniform(5.0, 9.0) # 半径
        h = RAND.uniform(15.0, 35.0) # 高度
        sides = RAND.integers(5, 9) # 边数
        center_y = base_y + h / 2.0
        v, f = create_prism((x, center_y, z), h, r, sides=sides)
        add_object(v, f, 4, C) # 对这些特殊建筑也使用标签 4
        prism_count += 1

    if attempts >= max_attempts_prism:
        print(f"    警告: 达到最大尝试次数。放置了 {prism_count}/{target_num_prisms} 个棱柱。")


    # --- 放置树木 ---
    print("    放置树木...")
    # 树木放置范围更广，避开道路和湖泊
    tree_placement_radius = world_radius * 1.1
    attempts = 0
    max_attempts_tree = target_num_trees * 3 # 每棵树允许较少的重试次数
    while tree_count < target_num_trees and attempts < max_attempts_tree:
        attempts += 1
        # 在更宽的半径内随机放置位置
        x = RAND.uniform(-tree_placement_radius, tree_placement_radius)
        z = RAND.uniform(-tree_placement_radius, tree_placement_radius)

        # 碰撞检查
        is_in_lake = (lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max)
        if is_in_lake: continue

        is_on_road = False
        road_clearance_tree = road_w / 1.5 # 树木的间隙可以小一些
        for gx in grid_coords:
             if abs(x - gx) < road_clearance_tree: is_on_road = True; break
        if not is_on_road:
            for gz in grid_coords:
                if abs(z - gz) < road_clearance_tree: is_on_road = True; break
        if is_on_road: continue

        # 创建树木
        h = RAND.uniform(7.0, 14.0)
        r = RAND.uniform(1.8, 3.0)
        segs = RAND.integers(9, 15)
        v, f = create_cone_tree((x, base_y, z), h, r, segments=segs)
        add_object(v, f, 5, C) # 标签 5: tree
        tree_count += 1

    if attempts >= max_attempts_tree:
        print(f"    警告: 达到最大尝试次数。放置了 {tree_count}/{target_num_trees} 棵树。")

    # --- 生成山脉 ---
    print("    生成山脉...")
    # 将山脉放置在更远的位置
    mountain_z1 = max_coord + 80.0
    mountain_z2 = mountain_z1 + 60.0
    mountain_width = world_radius * 1.5
    mountain_depth1 = 50.0
    mountain_depth2 = 40.0
    mountain_h_min1, mountain_h_max1 = 30.0, 85.0
    mountain_h_min2, mountain_h_max2 = 25.0, 60.0

    # 创建第一个山脉
    m_v1, m_f1 = create_mountain_range(
        x_start=-mountain_width, x_end=mountain_width, z_pos=mountain_z1,
        segs=mountain_segs1, depth=mountain_depth1, h_min=mountain_h_min1, h_max=mountain_h_max1)
    m_v1[:, 1] += base_y # 根据地面水平抬高
    add_object(m_v1, m_f1, 6, C) # 标签 6: mountain

    # 创建第二个更远的山脉
    m_v2, m_f2 = create_mountain_range(
        x_start=-mountain_width, x_end=mountain_width, z_pos=mountain_z2,
        segs=mountain_segs2, depth=mountain_depth2, h_min=mountain_h_min2, h_max=mountain_h_max2)
    m_v2[:, 1] += base_y # 抬高
    add_object(m_v2, m_f2, 6, C) # 标签 6: mountain

    # --- 完成场景数据 ---
    N = len(tris_vertices) # 三角形总数
    if N == 0:
        print("警告: 场景为空!")
        # 返回具有正确形状和数据类型的空数组
        return (np.empty((0, 3), dtype=GEOMETRY_DTYPE), # v0s
                np.empty((0, 3), dtype=GEOMETRY_DTYPE), # e1s
                np.empty((0, 3), dtype=GEOMETRY_DTYPE), # e2s
                np.empty((0, 3), dtype=GEOMETRY_DTYPE), # normals
                np.empty(0, dtype=LABEL_DTYPE),         # labels
                np.empty((0, 3), dtype=COLOR_DTYPE),    # colors
                np.empty(0, dtype=INDEX_DTYPE))         # prim_indices

    print(f"    实际对象: 建筑={building_count}, 摩天楼={skyscraper_count}, "
          f"棱柱={prism_count}, 树木={tree_count}")
    print(f"    总三角形数: {N}")

    # 将列表转换为 numpy 数组
    v0s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    e1s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    e2s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    normals = np.empty((N, 3), dtype=GEOMETRY_DTYPE) # 面法线

    for i, (v0, v1, v2) in enumerate(tris_vertices):
        v0_np = np.array(v0, dtype=GEOMETRY_DTYPE)
        v1_np = np.array(v1, dtype=GEOMETRY_DTYPE)
        v2_np = np.array(v2, dtype=GEOMETRY_DTYPE)

        v0s[i] = v0_np
        edge1 = np.subtract(v1_np, v0_np)
        edge2 = np.subtract(v2_np, v0_np)
        e1s[i] = edge1
        e2s[i] = edge2

        # 计算面法线
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len > INTERSECTION_EPSILON: # 避免归一化零向量
            normals[i] = normal / norm_len
        else:
            # 对退化三角形默认为向上向量
            normals[i] = np.array([0.0, 1.0, 0.0], dtype=GEOMETRY_DTYPE)

    labels = np.array(tris_labels, dtype=LABEL_DTYPE)
    colors = np.array(tris_colors, dtype=COLOR_DTYPE)
    prim_indices = np.arange(N, dtype=INDEX_DTYPE) # 原始索引 [0, 1, ..., N-1]

    # 返回所有必要的几何数据
    return v0s, e1s, e2s, normals, labels, colors, prim_indices


# --- BVH 构建 ---
# (BVH 构建逻辑基本保持不变，略作清理)

@njit(fastmath=True)
def calculate_tri_aabb_numba(v0, e1, e2):
    """计算三角形的轴对齐包围盒 (AABB) (Numba)。"""
    v1 = v0 + e1
    v2 = v0 + e2
    min_coord = np.empty(3, dtype=GEOMETRY_DTYPE)
    max_coord = np.empty(3, dtype=GEOMETRY_DTYPE)
    for k in range(3): # 遍历 x, y, z 轴
        min_coord[k] = min(v0[k], v1[k], v2[k])
        max_coord[k] = max(v0[k], v1[k], v2[k])
    return min_coord, max_coord

@njit(fastmath=True)
def calculate_bounds(indices, tri_aabbs_min, tri_aabbs_max):
    """计算一组三角形的组合 AABB (Numba)。"""
    num_tris = len(indices)
    if num_tris == 0:
        # 如果没有三角形，则返回无限边界
        return (np.full(3, INF, dtype=BVH_NODE_DTYPE),
                np.full(3, -INF, dtype=BVH_NODE_DTYPE))

    # 使用第一个三角形的边界进行初始化
    first_idx = indices[0]
    global_min = tri_aabbs_min[first_idx].copy()
    global_max = tri_aabbs_max[first_idx].copy()

    # 扩展边界以包含所有其他三角形
    for i in range(1, num_tris):
        idx = indices[i]
        current_min = tri_aabbs_min[idx]
        current_max = tri_aabbs_max[idx]
        for k in range(3): # 遍历 x, y, z
            global_min[k] = min(global_min[k], current_min[k])
            global_max[k] = max(global_max[k], current_max[k])
    return global_min, global_max

# --- Numba BVH 节点的位操作 ---
# 使用位转换将整数数据 (索引、计数) 存储在 float32 字段中
@njit
def int32_to_float32_bits(val_int32):
    """将 int32 位重新解释为 float32 位。"""
    int_array = np.array([val_int32], dtype=INDEX_DTYPE)
    return int_array.view(BVH_NODE_DTYPE)[0]

@njit
def float32_to_int32_bits(val_float32):
    """将 float32 位重新解释为 int32 位。"""
    float_array = np.array([val_float32], dtype=BVH_NODE_DTYPE)
    return float_array.view(INDEX_DTYPE)[0]
# ---

@njit
def recursive_build_numba(current_node_idx, nodes_used, flat_nodes, prim_indices,
                          start_idx, end_idx, # 该节点的图元范围
                          tri_aabbs_min, tri_aabbs_max, tri_centers):
    """递归构建 BVH 结构 (Numba)。"""
    num_prims = end_idx - start_idx
    node = flat_nodes[current_node_idx] # 获取当前节点数组切片

    # 计算该节点中图元的边界
    indices_slice = prim_indices[start_idx:end_idx]
    aabb_min, aabb_max = calculate_bounds(indices_slice, tri_aabbs_min, tri_aabbs_max)
    node[0:3] = aabb_min # 将最小边界存储在节点字段 0, 1, 2 中
    node[3:6] = aabb_max # 将最大边界存储在节点字段 3, 4, 5 中

    # --- 叶节点条件 ---
    if num_prims <= BVH_MAX_LEAF_SIZE:
        # 标记为叶节点：存储起始索引和负计数
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx)) # 信息 1: 起始索引
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims)) # 信息 2: 负图元计数
        return # 停止递归

    # --- 内部节点：确定分割轴 ---
    extent = aabb_max - aabb_min
    split_axis = np.argmax(extent) # 沿最长轴分割

    # 如果范围非常小，则无论如何都将其设为叶节点以避免问题
    if extent[split_axis] < INTERSECTION_EPSILON:
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx))
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims))
        return

    # --- 划分图元 ---
    split_coord = (aabb_min[split_axis] + aabb_max[split_axis]) * 0.5 # 在中点分割
    mid_point = start_idx # 分隔左右图元的索引

    # 使用三角形中心进行划分 (简单的表面积启发式近似)
    for i in range(start_idx, end_idx):
        prim_idx = prim_indices[i]
        if tri_centers[prim_idx, split_axis] < split_coord:
            # 将图元索引交换到左侧分区
            prim_indices[i], prim_indices[mid_point] = prim_indices[mid_point], prim_indices[i]
            mid_point += 1

    # 处理分割失败的情况 (所有图元都在一侧)
    if mid_point == start_idx or mid_point == end_idx:
        mid_point = start_idx + num_prims // 2 # 回退到简单的中位数分割

    # --- 分配并递归处理子节点 ---
    left_child_idx = nodes_used[0] # 获取下一个可用节点索引
    nodes_used[0] += 1             # 增加节点计数器
    right_child_idx = nodes_used[0]
    nodes_used[0] += 1

    # 在当前节点中存储子节点索引
    node[6] = int32_to_float32_bits(INDEX_DTYPE(left_child_idx))  # 信息 1: 左子节点索引
    node[7] = int32_to_float32_bits(INDEX_DTYPE(right_child_idx)) # 信息 2: 右子节点索引 (正数 = 内部节点)

    # 递归构建左右子节点
    recursive_build_numba(left_child_idx, nodes_used, flat_nodes, prim_indices,
                          start_idx, mid_point, # 图元 [start_idx, mid_point)
                          tri_aabbs_min, tri_aabbs_max, tri_centers)
    recursive_build_numba(right_child_idx, nodes_used, flat_nodes, prim_indices,
                          mid_point, end_idx, # 图元 [mid_point, end_idx)
                          tri_aabbs_min, tri_aabbs_max, tri_centers)


def build_bvh(v0s, e1s, e2s, prim_indices_in):
    """构建 BVH 加速结构。"""
    N = len(prim_indices_in) # 图元 (三角形) 数量
    if N == 0:
        # 如果没有图元，则返回空的 BVH
        return np.empty((0, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE), np.empty(0, dtype=INDEX_DTYPE)

    t0_precompute = _now()
    # --- 预计算所有三角形的 AABB 和中心 ---
    tri_aabbs_min = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_aabbs_max = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_centers = np.empty((N, 3), dtype=BVH_NODE_DTYPE)

    @njit(parallel=True, fastmath=True)
    def precompute_bounds_centers_parallel(num_tris, v0s_n, e1s_n, e2s_n,
                                           aabbs_min_out, aabbs_max_out, centers_out):
        """用于预计算的并行 Numba 函数。"""
        for i in prange(num_tris): # 并行循环
            v0, e1, e2 = v0s_n[i], e1s_n[i], e2s_n[i]
            aabb_min, aabb_max = calculate_tri_aabb_numba(v0, e1, e2)
            aabbs_min_out[i] = aabb_min
            aabbs_max_out[i] = aabb_max
            centers_out[i] = (aabb_min + aabb_max) * 0.5 # 中心是 AABB 的中点

    precompute_bounds_centers_parallel(N, v0s, e1s, e2s,
                                       tri_aabbs_min, tri_aabbs_max, tri_centers)
    log_step('BVH 预计算', t0_precompute)

    # --- 初始化 BVH 节点数组和递归 ---
    # 具有 N 个叶节点的二叉树中最大可能的节点数为 2N-1
    max_nodes = max(1, 2 * N - 1)
    # 为所有潜在节点预分配扁平数组
    flat_nodes = np.zeros((max_nodes, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE)
    # 复制图元索引，因为递归构建将就地重新排序它们
    ordered_prim_indices = np.copy(prim_indices_in)
    # 跟踪使用的节点数 (从根节点 1 开始)
    nodes_used = np.array([1], dtype=INDEX_DTYPE) # 作为数组传递以供 Numba 修改

    t0_recursive = _now()
    # 从根节点 (索引 0) 开始递归构建
    recursive_build_numba(0, nodes_used, flat_nodes, ordered_prim_indices,
                          0, N, # 最初处理所有图元
                          tri_aabbs_min, tri_aabbs_max, tri_centers)
    log_step('BVH 递归构建', t0_recursive)

    # --- 完成 BVH ---
    actual_nodes_used = nodes_used[0]
    # 将节点数组修剪为实际使用的大小
    flat_nodes = flat_nodes[:actual_nodes_used]
    print(f"    BVH 构建完成: 使用了 {actual_nodes_used} 个节点。")

    # 打印根节点边界 (可选的调试信息)
    if actual_nodes_used > 0:
        root_min = flat_nodes[0, 0:3]
        root_max = flat_nodes[0, 3:6]
        print(f"    根节点 AABB Min: [{root_min[0]:.2f}, {root_min[1]:.2f}, {root_min[2]:.2f}], "
              f"Max: [{root_max[0]:.2f}, {root_max[1]:.2f}, {root_max[2]:.2f}]")

    # 返回扁平节点数组和重新排序的图元索引
    return flat_nodes, ordered_prim_indices


# --- 光线-三角形相交 (CPU) ---
@njit(fastmath=True)
def intersect_ray_triangle_cpu(orig, dir, v0, e1, e2):
    """Moller-Trumbore 光线-三角形相交测试 (CPU Numba 版本)。"""
    h = np.cross(dir, e2)
    a = np.dot(e1, h)

    # 检查光线是否与三角形平面平行或背面相交 (如果需要)
    if abs(a) < INTERSECTION_EPSILON:
        return INF # 如果没有相交，则返回无穷大

    f = GEOMETRY_DTYPE(1.0) / a
    s = orig - v0 # 从三角形顶点 0 到光线原点的向量
    u = f * np.dot(s, h) # 重心坐标 U

    # 检查 U 边界
    if u < 0.0 or u > 1.0:
        return INF

    q = np.cross(s, e1)
    v = f * np.dot(dir, q) # 重心坐标 V

    # 检查 V 边界和 U+V 边界
    if v < 0.0 or u + v > 1.0:
        return INF

    # 计算相交距离 t
    t = f * np.dot(e2, q)

    # 仅当是向前相交时才返回 t
    return t if t > INTERSECTION_EPSILON else INF

# --- 光线-AABB 相交 (CPU) ---
@njit(fastmath=True)
def intersect_ray_aabb_cpu(orig, dir_inv, tmin_global, node_aabb_min, node_aabb_max):
    """使用 Slab 测试的光线-AABB 相交测试 (CPU Numba 版本)。"""
    t_near = -INF
    t_far = INF

    for k in range(3): # 遍历 x, y, z 轴
        inv_d = dir_inv[k] # 预计算的 1.0 / dir[k]
        aabb_min_k = node_aabb_min[k]
        aabb_max_k = node_aabb_max[k]

        # 计算与平板平面的相交距离
        t1 = (aabb_min_k - orig[k]) * inv_d
        t2 = (aabb_max_k - orig[k]) * inv_d

        # 确保 t1 是较小的距离，t2 是较大的距离
        if t1 > t2:
            t1, t2 = t2, t1

        # 更新整体的近处和远处相交距离
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

        # 提前退出条件:
        # 1. 如果近处相交比远处相交更远 (未命中)
        # 2. 如果整个 AABB 都在光线原点之后 (t_far < epsilon)
        # 3. 如果 AABB 上的最近命中点 (t_near) 比目前找到的最近三角形命中点 (tmin_global) 更远
        if t_near >= t_far or t_far < INTERSECTION_EPSILON or t_near >= tmin_global:
            return False # 未命中

    return True # 命中

# --- 阴影光线追踪 (CPU) ---
@njit(fastmath=True)
def trace_shadow_ray_cpu(
    shadow_orig, shadow_dir, max_dist, # 要检查的最大距离 (对于方向光通常为 INF)
    flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered
):
    """使用 BVH 追踪阴影光线。如果被遮挡则返回 True，否则返回 False。"""
    num_nodes = flat_nodes.shape[0]
    num_tris_total = len(v0s_reordered)
    if num_nodes == 0:
        return False # 没有对象，没有阴影

    # 预计算 AABB 测试的反方向
    dir_inv = np.empty(3, dtype=GEOMETRY_DTYPE)
    for k in range(3):
        # 处理轴对齐光线的除零问题
        if abs(shadow_dir[k]) < INTERSECTION_EPSILON:
            dir_inv[k] = math.copysign(INF, shadow_dir[k])
        else:
            dir_inv[k] = GEOMETRY_DTYPE(1.0) / shadow_dir[k]

    # --- BVH 遍历堆栈 ---
    BVH_CPU_STACK_SIZE = 64 # Numba 的固定大小堆栈
    node_stack = np.empty(BVH_CPU_STACK_SIZE, dtype=INDEX_DTYPE)
    stack_ptr = 0
    node_stack[stack_ptr] = 0 # 从根节点开始遍历
    stack_ptr += 1

    while stack_ptr > 0: # 当堆栈不为空时
        stack_ptr -= 1
        node_idx = node_stack[stack_ptr]

        # 节点索引的基本边界检查
        if node_idx < 0 or node_idx >= num_nodes:
            continue

        node = flat_nodes[node_idx]
        node_aabb_min = node[0:3]
        node_aabb_max = node[3:6]

        # 检查阴影光线是否与节点的 AABB 相交
        # 我们只关心 *max_dist* 之前的相交
        aabb_hit = intersect_ray_aabb_cpu(shadow_orig, dir_inv, max_dist, node_aabb_min, node_aabb_max)
        if not aabb_hit:
            continue # 如果 AABB 未命中，则跳过此节点及其子节点

        # --- 处理节点 ---
        info_bits = node[7]
        info_val = float32_to_int32_bits(info_bits)

        if info_val < 0: # 叶节点
            prim_count = -info_val # 图元计数存储为负数
            prim_offset_bits = node[6]
            prim_offset = float32_to_int32_bits(prim_offset_bits)

            # 与叶节点中的所有三角形进行光线相交测试
            for p_local_idx in range(prim_count):
                p_idx = prim_offset + p_local_idx
                if p_idx < num_tris_total: # 边界检查
                    t = intersect_ray_triangle_cpu(shadow_orig, shadow_dir,
                                                   v0s_reordered[p_idx],
                                                   e1s_reordered[p_idx],
                                                   e2s_reordered[p_idx])
                    # 如果找到 *任何* 比 max_dist 更近的相交，
                    # 则该点被遮挡。提前退出。
                    if t < max_dist:
                        return True # 被遮挡

        else: # 内部节点
            left_child_idx_bits = node[6]
            left_child_idx = float32_to_int32_bits(left_child_idx_bits)
            right_child_idx = info_val # 右子节点索引直接存储

            # 将子节点压入堆栈 (顺序可能影响性能，但基本压栈即可)
            # 压栈前检查堆栈溢出
            if stack_ptr + 2 <= BVH_CPU_STACK_SIZE:
                # 如果右子节点有效，则压入
                if right_child_idx >= 0 and right_child_idx < num_nodes:
                    node_stack[stack_ptr] = right_child_idx
                    stack_ptr += 1
                # 如果左子节点有效，则压入
                if left_child_idx >= 0 and left_child_idx < num_nodes:
                    node_stack[stack_ptr] = left_child_idx
                    stack_ptr += 1
            # else: 堆栈溢出 (对于 BVH_CPU_STACK_SIZE=64 应该很少见)

    return False # 在光线段上未找到遮挡


# --- 光线追踪 (CPU - BVH - Phong 着色) ---
@njit(parallel=True, fastmath=True)
def raytrace_cpu_bvh(
    flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
    normals_reordered, # 面法线 (重新排序)
    labels_reordered, colors_reordered, # 语义标签和基础颜色 (重新排序)
    cam_o, cam_dir, right, up, # 相机参数
    screen_w, screen_h, W, H # 屏幕参数和分辨率
):
    """使用 BVH 在 CPU 上执行光线追踪，带有 Phong 着色和阴影。"""
    # 初始化输出数组
    rgb = np.zeros((H, W, 3), dtype=COLOR_DTYPE)
    depth = np.full((H, W), INF, dtype=DEPTH_DTYPE)
    sem = np.full((H, W), SKY_LABEL, dtype=LABEL_DTYPE) # 使用 SKY_LABEL 初始化语义图
    pts = np.full((H, W, 3), np.nan, dtype=POINT_DTYPE) # 相交点

    # 预计算光线生成的因子
    inv_W = GEOMETRY_DTYPE(1.0 / W)
    inv_H = GEOMETRY_DTYPE(1.0 / H)
    num_nodes = flat_nodes.shape[0]
    num_tris_total = len(v0s_reordered)

    # 处理空场景情况
    if num_nodes == 0:
        # 用天空颜色填充 rgb (稍后默认天空颜色会隐式完成)
        # 用天空标签填充 sem (已初始化)
        return rgb, depth, sem, pts

    # --- 光照常量 (在并行循环内部使用) ---
    sun_dir_cpu = SUN_DIRECTION
    ambient_light_cpu = AMBIENT_LIGHT
    specular_color_cpu = SPECULAR_COLOR
    specular_exponent_cpu = SPECULAR_EXPONENT
    ks_cpu = Ks
    shadow_bias_cpu = SHADOW_BIAS
    default_sky_color_cpu = DEFAULT_SKY_COLOR

    # --- 主循环 (按行并行) ---
    for i in prange(H): # Numba 并行循环
        # --- 每线程局部变量 ---
        # (与频繁访问共享全局数组相比，减少了竞争)
        node_stack = np.empty(BVH_CPU_STACK_SIZE, dtype=INDEX_DTYPE) # BVH 遍历堆栈
        d = np.empty(3, dtype=GEOMETRY_DTYPE) # 光线方向
        dir_inv = np.empty(3, dtype=GEOMETRY_DTYPE) # 光线反方向
        hit_normal = np.empty(3, dtype=GEOMETRY_DTYPE) # 命中点法线
        hit_point = np.empty(3, dtype=GEOMETRY_DTYPE) # 相交点
        shadow_ray_origin = np.empty(3, dtype=GEOMETRY_DTYPE) # 阴影光线原点
        view_dir = np.empty(3, dtype=GEOMETRY_DTYPE) # 从命中点到相机的方向
        reflect_dir = np.empty(3, dtype=GEOMETRY_DTYPE) # 反射光方向
        base_color_float = np.empty(3, dtype=GEOMETRY_DTYPE) # 基础颜色浮点数 [0,1]
        final_color_float = np.empty(3, dtype=GEOMETRY_DTYPE) # 最终计算颜色 [0,1]

        # --- 内循环 (按列) ---
        for j in range(W):
            # 1. 计算主光线方向
            # 将像素坐标 (i, j) 转换为归一化屏幕坐标 [-0.5, 0.5]
            u_norm = (GEOMETRY_DTYPE(j) + 0.5) * inv_W - 0.5
            v_norm = (GEOMETRY_DTYPE(i) + 0.5) * inv_H - 0.5 # 对典型图像坐标反转 v

            # 使用相机框架 (原点、方向、右、上) 计算方向
            d[0] = cam_dir[0] + u_norm * screen_w * right[0] - v_norm * screen_h * up[0]
            d[1] = cam_dir[1] + u_norm * screen_w * right[1] - v_norm * screen_h * up[1]
            d[2] = cam_dir[2] + u_norm * screen_w * right[2] - v_norm * screen_h * up[2]

            # 归一化方向向量
            norm_sq = d[0]**2 + d[1]**2 + d[2]**2
            if norm_sq < INTERSECTION_EPSILON**2: # 避免除零/无效光线
                # 如果光线无效，则分配天空属性 (应该很少见)
                rgb[i, j, 0] = default_sky_color_cpu[0]
                rgb[i, j, 1] = default_sky_color_cpu[1]
                rgb[i, j, 2] = default_sky_color_cpu[2]
                # depth[i,j] = INF (已经是默认值)
                # sem[i,j] = SKY_LABEL (已经是默认值)
                # pts[i,j] = NaN (已经是默认值)
                continue
            inv_norm = GEOMETRY_DTYPE(1.0) / math.sqrt(norm_sq)
            d *= inv_norm # d 现在是归一化的主光线方向

            # 预计算 AABB 测试的反方向
            for k in range(3):
                if abs(d[k]) < INTERSECTION_EPSILON:
                    dir_inv[k] = math.copysign(INF, d[k])
                else:
                    dir_inv[k] = GEOMETRY_DTYPE(1.0) / d[k]

            # 2. 初始化主光线遍历
            tmin = INF # 重置此光线的最近命中距离
            hit_prim_idx = -1 # 重置命中图元索引
            stack_ptr = 0
            node_stack[stack_ptr] = 0 # 从根节点开始
            stack_ptr += 1

            # 3. 主光线的 BVH 遍历循环
            while stack_ptr > 0:
                stack_ptr -= 1
                node_idx = node_stack[stack_ptr]

                if node_idx < 0 or node_idx >= num_nodes: continue # 无效节点索引检查

                node = flat_nodes[node_idx]
                node_aabb_min = node[0:3]
                node_aabb_max = node[3:6]

                # 检查 AABB 相交，如果节点比当前最近命中 (tmin) 更远，则剪枝
                aabb_hit = intersect_ray_aabb_cpu(cam_o, dir_inv, tmin, node_aabb_min, node_aabb_max)
                if not aabb_hit: continue # 如果 AABB 未命中或太远，则跳过节点

                info_bits = node[7]
                info_val = float32_to_int32_bits(info_bits)

                if info_val < 0: # 叶节点
                    prim_count = -info_val
                    prim_offset_bits = node[6]
                    prim_offset = float32_to_int32_bits(prim_offset_bits)

                    # 与叶节点中的三角形进行相交测试
                    for p_local_idx in range(prim_count):
                        p_idx = prim_offset + p_local_idx
                        if p_idx < num_tris_total: # 检查索引边界
                            t = intersect_ray_triangle_cpu(cam_o, d,
                                                           v0s_reordered[p_idx],
                                                           e1s_reordered[p_idx],
                                                           e2s_reordered[p_idx])
                            # 如果此命中比当前最小值更近
                            if t < tmin:
                                tmin = t # 更新最小距离
                                hit_prim_idx = p_idx # 存储命中图元的索引
                                # 存储命中三角形的法线 (无需使用 [:] 复制)
                                hit_normal = normals_reordered[p_idx]
                else: # 内部节点
                    left_child_idx_bits = node[6]
                    right_child_idx = info_val
                    left_child_idx = float32_to_int32_bits(left_child_idx_bits)

                    # 如果堆栈有空间，则将子节点压入堆栈
                    if stack_ptr + 2 <= BVH_CPU_STACK_SIZE:
                        if right_child_idx >= 0 and right_child_idx < num_nodes:
                            node_stack[stack_ptr] = right_child_idx
                            stack_ptr += 1
                        if left_child_idx >= 0 and left_child_idx < num_nodes:
                            node_stack[stack_ptr] = left_child_idx
                            stack_ptr += 1

            # 4. 处理像素的命中结果
            if hit_prim_idx >= 0: # 发生命中
                # 存储深度、语义标签和相交点
                depth[i, j] = tmin
                sem[i, j] = labels_reordered[hit_prim_idx]
                hit_point = cam_o + tmin * d # 计算相交点
                pts[i, j, 0] = hit_point[0]
                pts[i, j, 1] = hit_point[1]
                pts[i, j, 2] = hit_point[2]

                # --- 光照和阴影计算 ---
                # 获取基础颜色并转换为浮点数 [0, 1]
                color_uint8 = colors_reordered[hit_prim_idx]
                base_color_float[0] = GEOMETRY_DTYPE(color_uint8[0]) / 255.0
                base_color_float[1] = GEOMETRY_DTYPE(color_uint8[1]) / 255.0
                base_color_float[2] = GEOMETRY_DTYPE(color_uint8[2]) / 255.0

                # 确保法线指向相机 (远离表面)
                if np.dot(hit_normal, d) > 0.0:
                    hit_normal = -hit_normal # 如果法线指向光线方向，则翻转法线

                # --- 漫反射光照 (Lambertian) ---
                # 计算法线和光照方向之间的点积
                # 确保结果不为负
                light_dot_normal = max(0.0, np.dot(hit_normal, sun_dir_cpu))
                diffuse_intensity = SUN_INTENSITY * light_dot_normal

                # --- 阴影计算 ---
                # 沿法线稍微偏移原点以避免自相交
                shadow_ray_origin = hit_point + hit_normal * shadow_bias_cpu
                # 向太阳追踪阴影光线 (检查到无穷远)
                is_occluded = trace_shadow_ray_cpu(
                    shadow_ray_origin, sun_dir_cpu, INF,
                    flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered
                )
                shadow_factor = GEOMETRY_DTYPE(0.0) if is_occluded else GEOMETRY_DTYPE(1.0)

                # --- 镜面光照 (Phong) ---
                # 观察方向 (从命中点到相机)
                view_dir = cam_o - hit_point
                view_dir_norm = np.linalg.norm(view_dir)
                if view_dir_norm > INTERSECTION_EPSILON:
                    view_dir /= view_dir_norm
                else: # 如果相机在命中点，避免除零
                    view_dir = -d # 用负光线方向近似

                # 反射方向 R = 2 * (N . L) * N - L
                # L 是 *到* 光源的方向 = sun_dir_cpu
                reflect_dot = 2.0 * np.dot(hit_normal, sun_dir_cpu)
                reflect_dir[0] = reflect_dot * hit_normal[0] - sun_dir_cpu[0]
                reflect_dir[1] = reflect_dot * hit_normal[1] - sun_dir_cpu[1]
                reflect_dir[2] = reflect_dot * hit_normal[2] - sun_dir_cpu[2]
                # 如果 sun_dir 和 normal 已归一化，则无需归一化 reflect_dir

                # 镜面强度 = Ks * (max(0, V . R)) ^ exponent
                specular_dot_view = max(0.0, np.dot(view_dir, reflect_dir))
                specular_intensity = ks_cpu * (specular_dot_view ** specular_exponent_cpu)

                # --- 最终颜色计算 (环境光 + 漫反射*阴影 + 镜面反射*阴影) ---
                final_color_float[0] = base_color_float[0] * (ambient_light_cpu + diffuse_intensity * shadow_factor) + specular_color_cpu[0] * specular_intensity * shadow_factor
                final_color_float[1] = base_color_float[1] * (ambient_light_cpu + diffuse_intensity * shadow_factor) + specular_color_cpu[1] * specular_intensity * shadow_factor
                final_color_float[2] = base_color_float[2] * (ambient_light_cpu + diffuse_intensity * shadow_factor) + specular_color_cpu[2] * specular_intensity * shadow_factor

                # 将颜色值限制在 [0, 1] 并转换回 uint8 [0, 255]
                final_r = max(0.0, min(1.0, final_color_float[0]))
                final_g = max(0.0, min(1.0, final_color_float[1]))
                final_b = max(0.0, min(1.0, final_color_float[2]))
                rgb[i, j, 0] = int(final_r * 255.0)
                rgb[i, j, 1] = int(final_g * 255.0)
                rgb[i, j, 2] = int(final_b * 255.0)

            else: # 未命中 (天空)
                # 分配天空颜色
                rgb[i, j, 0] = default_sky_color_cpu[0]
                rgb[i, j, 1] = default_sky_color_cpu[1]
                rgb[i, j, 2] = default_sky_color_cpu[2]
                # depth 保持 INF, sem 保持 SKY_LABEL, pts 保持 NaN (已由初始化设置)

    return rgb, depth, sem, pts


# --- GPU 内核 (仅当 CUDA 可用时) ---
if _GPU_AVAILABLE:

    # --- 光线-三角形相交 (GPU 设备函数) ---
    @cuda.jit(device=True, inline=True)
    def ray_tri_intersect_gpu(orig, dir, v0, e1, e2):
        """Moller-Trumbore 光线-三角形相交测试 (GPU 设备函数)。"""
        eps_gpu = INTERSECTION_EPSILON
        inf_gpu = GEOMETRY_DTYPE(1e20) # GPU 上的大数表示无穷大

        # 计算行列式分量 (叉积)
        h0 = dir[1] * e2[2] - dir[2] * e2[1]
        h1 = dir[2] * e2[0] - dir[0] * e2[2]
        h2 = dir[0] * e2[1] - dir[1] * e2[0]
        a = e1[0] * h0 + e1[1] * h1 + e1[2] * h2 # 行列式

        if abs(a) < eps_gpu: return inf_gpu # 平行或接近平行

        f = GEOMETRY_DTYPE(1.0) / a
        s0 = orig[0] - v0[0]
        s1 = orig[1] - v0[1]
        s2 = orig[2] - v0[2]
        u = f * (s0 * h0 + s1 * h1 + s2 * h2) # 重心 U

        if u < 0.0 or u > 1.0: return inf_gpu # 检查 U 边界

        # 计算重心 V 分量 (叉积)
        q0 = s1 * e1[2] - s2 * e1[1]
        q1 = s2 * e1[0] - s0 * e1[2]
        q2 = s0 * e1[1] - s1 * e1[0]
        v = f * (dir[0] * q0 + dir[1] * q1 + dir[2] * q2) # 重心 V

        if v < 0.0 or u + v > 1.0: return inf_gpu # 检查 V 边界

        t = f * (e2[0] * q0 + e2[1] * q1 + e2[2] * q2) # 相交距离 t

        return t if t > eps_gpu else inf_gpu # 仅当向前相交时返回 t

    # --- 光线-AABB 相交 (GPU 设备函数) ---
    @cuda.jit(device=True, inline=True)
    def ray_aabb_intersect_gpu(orig, dir_inv, t_min_global, node_aabb_min, node_aabb_max):
        """光线-AABB 相交测试 (Slab 测试) (GPU 设备函数)。"""
        t_near = -INF # 使用模块级 INF
        t_far = INF  # 使用模块级 INF
        eps_aabb = INTERSECTION_EPSILON

        for k in range(3):
            inv_d = dir_inv[k]
            aabb_min_k = node_aabb_min[k]
            aabb_max_k = node_aabb_max[k]
            t1 = (aabb_min_k - orig[k]) * inv_d
            t2 = (aabb_max_k - orig[k]) * inv_d

            # 必要时交换
            if t1 > t2: t1, t2 = t2, t1

            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

            # 提前退出条件 (与 CPU 相同)
            if t_near >= t_far or t_far < eps_aabb or t_near >= t_min_global:
                return False # 未命中
        return True # 命中

    # --- 阴影光线追踪 (GPU 设备函数) ---
    @cuda.jit(device=True)
    def trace_shadow_ray_gpu_device(
        shadow_orig, shadow_dir, max_dist,
        flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered
    ):
        """在 GPU 上使用 BVH 追踪阴影光线。如果被遮挡则返回 True。"""
        num_nodes = flat_nodes.shape[0]
        num_tris_total_gpu = v0s_reordered.shape[0]
        if num_nodes == 0: return False

        # --- 用于遍历的每线程局部变量 ---
        node_stack = cuda.local.array(64, dtype=INDEX_DTYPE) # 每线程局部堆栈
        dir_inv = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        node_aabb_min_local = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        node_aabb_max_local = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        inf_gpu = GEOMETRY_DTYPE(1e20)
        eps_gpu = INTERSECTION_EPSILON

        # 预计算反方向
        for k in range(3):
            if abs(shadow_dir[k]) < eps_gpu:
                dir_inv[k] = math.copysign(inf_gpu, shadow_dir[k]) # 在 GPU 上使用 math.copysign
            else:
                dir_inv[k] = GEOMETRY_DTYPE(1.0) / shadow_dir[k]

        stack_ptr = 0
        node_stack[stack_ptr] = 0 # 从根节点开始
        stack_ptr += 1

        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = node_stack[stack_ptr]
            if node_idx < 0 or node_idx >= num_nodes: continue

            # 将 AABB 边界获取到局部内存中
            node_aabb_min_local[0] = flat_nodes[node_idx, 0]
            node_aabb_min_local[1] = flat_nodes[node_idx, 1]
            node_aabb_min_local[2] = flat_nodes[node_idx, 2]
            node_aabb_max_local[0] = flat_nodes[node_idx, 3]
            node_aabb_max_local[1] = flat_nodes[node_idx, 4]
            node_aabb_max_local[2] = flat_nodes[node_idx, 5]

            # 检查 AABB 相交
            if not ray_aabb_intersect_gpu(shadow_orig, dir_inv, max_dist, node_aabb_min_local, node_aabb_max_local):
                continue

            # --- 处理节点 ---
            # 在 GPU 设备数组上使用 .view(dtype) 进行位转换
            info_bits_f = flat_nodes[node_idx, 7]
            info_val = info_bits_f.view(INDEX_DTYPE) # 将 float 位转换为 int

            if info_val < 0: # 叶节点
                prim_count = -info_val
                prim_offset_bits_f = flat_nodes[node_idx, 6]
                prim_offset = prim_offset_bits_f.view(INDEX_DTYPE) # 将 float 位转换为 int

                # 与叶节点中的三角形进行相交测试
                for p_idx_offset in range(prim_count):
                    current_prim_idx = prim_offset + p_idx_offset
                    if current_prim_idx < num_tris_total_gpu: # 检查边界
                        # 直接获取三角形数据 (可能导致缓存未命中但更简单)
                        tri_v0 = v0s_reordered[current_prim_idx]
                        tri_e1 = e1s_reordered[current_prim_idx]
                        tri_e2 = e2s_reordered[current_prim_idx]
                        t = ray_tri_intersect_gpu(shadow_orig, shadow_dir, tri_v0, tri_e1, tri_e2)
                        # 如果 *任何* 命中比 max_dist 更近，则返回被遮挡
                        if t < max_dist:
                            return True # 被遮挡
            else: # 内部节点
                left_child_idx_bits_f = flat_nodes[node_idx, 6]
                left_child_idx = left_child_idx_bits_f.view(INDEX_DTYPE) # 转换
                right_child_idx = info_val # 已经是 int

                # 如果有可用空间，则将子节点压入堆栈
                if stack_ptr + 2 <= 64: # 检查局部堆栈大小
                    if right_child_idx >= 0 and right_child_idx < num_nodes:
                        node_stack[stack_ptr] = right_child_idx
                        stack_ptr += 1
                    if left_child_idx >= 0 and left_child_idx < num_nodes:
                        node_stack[stack_ptr] = left_child_idx
                        stack_ptr += 1
                # else: 堆栈溢出 (应该很少见)

        return False # 未被遮挡


    # --- 主光线追踪 CUDA 内核 (Phong 着色) ---
    @cuda.jit
    def raytrace_cuda_bvh_kernel(
        flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
        normals_reordered, # 面法线
        labels_reordered, colors_reordered, # 语义和颜色
        cam_o, cam_dir, right, up, # 相机框架 (作为 numpy 数组传递)
        sun_dir_gpu, # 太阳方向 (设备数组)
        specular_color_gpu, # 镜面颜色 (设备数组)
        scr_w, scr_h, W, H, # 屏幕参数和分辨率
        rgb, depth, sem, pts # 输出设备数组
    ):
        """用于 BVH、Phong 着色和阴影的光线追踪 CUDA 内核。"""
        i, j = cuda.grid(2) # 获取与像素 (行 i, 列 j) 对应的线程索引

        # 边界检查：确保线程在图像尺寸内
        if i >= H or j >= W: return

        # --- 常量和局部变量 ---
        inf_gpu = GEOMETRY_DTYPE(1e20)
        eps_gpu = INTERSECTION_EPSILON
        ambient_light_gpu = AMBIENT_LIGHT # 使用全局常量
        specular_exponent_gpu = SPECULAR_EXPONENT
        ks_gpu = Ks
        shadow_bias_gpu = SHADOW_BIAS
        sky_label_gpu = SKY_LABEL
        default_sky_r = DEFAULT_SKY_COLOR[0]
        default_sky_g = DEFAULT_SKY_COLOR[1]
        default_sky_b = DEFAULT_SKY_COLOR[2]

        # 线程内中间计算的局部数组
        d = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        dir_inv = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        node_aabb_min = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        node_aabb_max = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        hit_normal = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        hit_point = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        base_color_float = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        shadow_ray_origin = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        view_dir = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        reflect_dir = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        final_color_float = cuda.local.array(3, dtype=GEOMETRY_DTYPE)

        num_tris_total_gpu = v0s_reordered.shape[0]
        num_nodes = flat_nodes.shape[0]

        # 1. 计算主光线方向 (与 CPU 相同)
        u_norm = (GEOMETRY_DTYPE(j) + 0.5) / W - 0.5
        v_norm = (GEOMETRY_DTYPE(i) + 0.5) / H - 0.5
        d[0] = cam_dir[0] + u_norm * scr_w * right[0] - v_norm * scr_h * up[0]
        d[1] = cam_dir[1] + u_norm * scr_w * right[1] - v_norm * scr_h * up[1]
        d[2] = cam_dir[2] + u_norm * scr_w * right[2] - v_norm * scr_h * up[2]

        nrm_sq = d[0]**2 + d[1]**2 + d[2]**2
        if nrm_sq < eps_gpu**2:
             # 如果光线无效，则分配天空并返回
             rgb[i, j, 0] = default_sky_r
             rgb[i, j, 1] = default_sky_g
             rgb[i, j, 2] = default_sky_b
             depth[i, j] = inf_gpu
             sem[i, j] = sky_label_gpu
             # pts 保持 NaN (默认值)
             return

        inv_nrm = GEOMETRY_DTYPE(1.0) / math.sqrt(nrm_sq) # 在 GPU 上使用 math.sqrt
        d[0] *= inv_nrm
        d[1] *= inv_nrm
        d[2] *= inv_nrm

        # 预计算反方向
        for k in range(3):
            if abs(d[k]) < eps_gpu:
                dir_inv[k] = math.copysign(inf_gpu, d[k])
            else:
                dir_inv[k] = GEOMETRY_DTYPE(1.0) / d[k]

        # 2. 初始化主光线遍历
        tmin = inf_gpu
        hit_prim_idx = -1

        # 光线计算后处理空场景
        if num_nodes == 0:
            rgb[i, j, 0] = default_sky_r
            rgb[i, j, 1] = default_sky_g
            rgb[i, j, 2] = default_sky_b
            depth[i, j] = inf_gpu
            sem[i, j] = sky_label_gpu
            return

        # --- BVH 遍历堆栈 (局部) ---
        BVH_GPU_STACK_SIZE = 64 # 如果使用，必须与设备函数匹配
        node_stack = cuda.local.array(BVH_GPU_STACK_SIZE, dtype=INDEX_DTYPE)
        stack_ptr = 0
        node_stack[stack_ptr] = 0 # 从根节点开始
        stack_ptr += 1

        # 3. 主光线的 BVH 遍历循环
        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = node_stack[stack_ptr]
            if node_idx < 0 or node_idx >= num_nodes: continue

            # 将 AABB 边界获取到局部数组中
            node_aabb_min[0] = flat_nodes[node_idx, 0]
            node_aabb_min[1] = flat_nodes[node_idx, 1]
            node_aabb_min[2] = flat_nodes[node_idx, 2]
            node_aabb_max[0] = flat_nodes[node_idx, 3]
            node_aabb_max[1] = flat_nodes[node_idx, 4]
            node_aabb_max[2] = flat_nodes[node_idx, 5]

            # 检查 AABB 相交
            if not ray_aabb_intersect_gpu(cam_o, dir_inv, tmin, node_aabb_min, node_aabb_max):
                continue

            # 处理节点 (叶节点或内部节点)
            info_bits_f = flat_nodes[node_idx, 7]
            info_val = info_bits_f.view(INDEX_DTYPE) # 位转换

            if info_val < 0: # 叶节点
                prim_count = -info_val
                prim_offset_bits_f = flat_nodes[node_idx, 6]
                prim_offset = prim_offset_bits_f.view(INDEX_DTYPE) # 位转换

                for p_idx_offset in range(prim_count):
                    current_prim_idx = prim_offset + p_idx_offset
                    if current_prim_idx < num_tris_total_gpu: # 检查边界
                        # 获取三角形数据
                        tri_v0 = v0s_reordered[current_prim_idx]
                        tri_e1 = e1s_reordered[current_prim_idx]
                        tri_e2 = e2s_reordered[current_prim_idx]
                        t = ray_tri_intersect_gpu(cam_o, d, tri_v0, tri_e1, tri_e2)
                        if t < tmin:
                            tmin = t
                            hit_prim_idx = current_prim_idx
                            # 将法线存储在局部数组中
                            hit_normal[0] = normals_reordered[current_prim_idx, 0]
                            hit_normal[1] = normals_reordered[current_prim_idx, 1]
                            hit_normal[2] = normals_reordered[current_prim_idx, 2]
            else: # 内部节点
                left_child_idx_bits_f = flat_nodes[node_idx, 6]
                left_child_idx = left_child_idx_bits_f.view(INDEX_DTYPE) # 位转换
                right_child_idx = info_val # 已经是 int

                # 将子节点压入堆栈
                if stack_ptr + 2 <= BVH_GPU_STACK_SIZE:
                    if right_child_idx >= 0 and right_child_idx < num_nodes:
                        node_stack[stack_ptr] = right_child_idx
                        stack_ptr += 1
                    if left_child_idx >= 0 and left_child_idx < num_nodes:
                        node_stack[stack_ptr] = left_child_idx
                        stack_ptr += 1

        # 4. 处理像素的命中结果
        if hit_prim_idx >= 0: # 发生命中
            # --- 存储输出数据 ---
            depth[i, j] = tmin
            sem[i, j] = labels_reordered[hit_prim_idx]
            # 计算命中点并存储在 pts 数组中
            hit_point[0] = cam_o[0] + d[0] * tmin
            hit_point[1] = cam_o[1] + d[1] * tmin
            hit_point[2] = cam_o[2] + d[2] * tmin
            pts[i, j, 0] = hit_point[0]
            pts[i, j, 1] = hit_point[1]
            pts[i, j, 2] = hit_point[2]

            # --- 光照和阴影计算 ---
            # 获取基础颜色 uint8
            color_uint8 = colors_reordered[hit_prim_idx]
            # 转换为浮点数 [0, 1]
            base_color_float[0] = GEOMETRY_DTYPE(color_uint8[0]) / 255.0
            base_color_float[1] = GEOMETRY_DTYPE(color_uint8[1]) / 255.0
            base_color_float[2] = GEOMETRY_DTYPE(color_uint8[2]) / 255.0

            # 确保法线指向相机
            normal_dot_ray = hit_normal[0] * d[0] + hit_normal[1] * d[1] + hit_normal[2] * d[2]
            if normal_dot_ray > 0.0:
                hit_normal[0] = -hit_normal[0]
                hit_normal[1] = -hit_normal[1]
                hit_normal[2] = -hit_normal[2]

            # 漫反射光照
            light_dot_normal = max(0.0, hit_normal[0] * sun_dir_gpu[0] +
                                         hit_normal[1] * sun_dir_gpu[1] +
                                         hit_normal[2] * sun_dir_gpu[2])
            diffuse_intensity = SUN_INTENSITY * light_dot_normal # SUN_INTENSITY 是 1.0

            # 阴影计算
            shadow_ray_origin[0] = hit_point[0] + hit_normal[0] * shadow_bias_gpu
            shadow_ray_origin[1] = hit_point[1] + hit_normal[1] * shadow_bias_gpu
            shadow_ray_origin[2] = hit_point[2] + hit_normal[2] * shadow_bias_gpu

            # 调用阴影光线设备函数
            is_occluded = trace_shadow_ray_gpu_device(
                shadow_ray_origin, sun_dir_gpu, inf_gpu, # 检查到无穷远
                flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered
            )
            shadow_factor = GEOMETRY_DTYPE(0.0) if is_occluded else GEOMETRY_DTYPE(1.0)

            # 镜面光照 (Phong)
            # 观察方向 V = normalize(Eye - Hit)
            view_dir[0] = cam_o[0] - hit_point[0]
            view_dir[1] = cam_o[1] - hit_point[1]
            view_dir[2] = cam_o[2] - hit_point[2]
            view_dir_norm_sq = view_dir[0]**2 + view_dir[1]**2 + view_dir[2]**2
            if view_dir_norm_sq > eps_gpu**2:
                inv_view_norm = GEOMETRY_DTYPE(1.0) / math.sqrt(view_dir_norm_sq)
                view_dir[0] *= inv_view_norm
                view_dir[1] *= inv_view_norm
                view_dir[2] *= inv_view_norm
            # else: 如果范数接近零，view_dir 可以保持未归一化

            # 反射方向 R = 2 * (N . L) * N - L
            # L 是 *到* 光源的方向 = sun_dir_gpu
            reflect_dot = 2.0 * (hit_normal[0] * sun_dir_gpu[0] +
                                 hit_normal[1] * sun_dir_gpu[1] +
                                 hit_normal[2] * sun_dir_gpu[2])
            reflect_dir[0] = reflect_dot * hit_normal[0] - sun_dir_gpu[0]
            reflect_dir[1] = reflect_dot * hit_normal[1] - sun_dir_gpu[1]
            reflect_dir[2] = reflect_dot * hit_normal[2] - sun_dir_gpu[2]
            # 假设 N 和 L 已归一化，R 也将归一化

            # 镜面强度 = Ks * (max(0, V . R)) ^ exponent
            specular_dot_view = max(0.0, view_dir[0] * reflect_dir[0] +
                                         view_dir[1] * reflect_dir[1] +
                                         view_dir[2] * reflect_dir[2])
            specular_intensity = ks_gpu * (specular_dot_view ** specular_exponent_gpu)

            # 最终颜色 = 环境光 + 漫反射*阴影 + 镜面反射*阴影
            final_color_float[0] = base_color_float[0] * (ambient_light_gpu + diffuse_intensity * shadow_factor) + specular_color_gpu[0] * specular_intensity * shadow_factor
            final_color_float[1] = base_color_float[1] * (ambient_light_gpu + diffuse_intensity * shadow_factor) + specular_color_gpu[1] * specular_intensity * shadow_factor
            final_color_float[2] = base_color_float[2] * (ambient_light_gpu + diffuse_intensity * shadow_factor) + specular_color_gpu[2] * specular_intensity * shadow_factor

            # 限制并转换为 uint8 用于输出 RGB 数组
            # 如果需要，可以使用 cuda.atomic.max/min 进行限制，但简单的 max/min 通常有效
            final_r = max(0.0, min(1.0, final_color_float[0]))
            final_g = max(0.0, min(1.0, final_color_float[1]))
            final_b = max(0.0, min(1.0, final_color_float[2]))
            rgb[i, j, 0] = int(final_r * 255.0)
            rgb[i, j, 1] = int(final_g * 255.0)
            rgb[i, j, 2] = int(final_b * 255.0)

        else: # 未命中 (天空)
            rgb[i, j, 0] = default_sky_r
            rgb[i, j, 1] = default_sky_g
            rgb[i, j, 2] = default_sky_b
            depth[i, j] = inf_gpu
            sem[i, j] = sky_label_gpu # 分配天空标签
            # pts 保持 NaN

# --- OBJ 导出 (修改以支持图层和法线) ---
def save_combined_obj(filename, v0s, e1s, e2s, normals, labels, # 原始几何数据
                      cam_o, cam_dir, right, up, screen_w, screen_h,
                      pts, far):
    """将场景三角形 (按标签分组)、法线、相机视锥体
       和命中点保存到 OBJ 文件中。"""
    print(f"    准备保存带图层的 OBJ 文件: {filename}")
    num_triangles = len(v0s)
    if num_triangles == 0:
        print("    跳过 OBJ 保存 (场景中没有三角形)。")
        return

    # 创建从标签 ID 到三角形索引列表的映射
    tris_by_label = {}
    for i in range(num_triangles):
        label = labels[i]
        if label not in tris_by_label:
            tris_by_label[label] = []
        tris_by_label[label].append(i)

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'# Raytracer output: {datetime.now()}\n')
            f.write(f'# Generated with {num_triangles} triangles.\n')
            f.write(f'# Semantic Layers: {", ".join(LABEL_MAP.values())}\n\n')

            vertex_offset = 1 # OBJ 索引是 1-based
            normal_offset = 1

            # --- 按图层写入场景几何体 ---
            print("    按语义图层写入场景几何体...")
            # 遍历场景中找到的标签
            sorted_labels = sorted(tris_by_label.keys())

            all_vertices = []
            all_normals = []
            face_definitions = [] # 存储元组: (label, v1_idx, n1_idx, v2_idx, n2_idx, v3_idx, n3_idx)

            # 第一遍：收集所有唯一的顶点和法线
            vertex_map = {} # 映射 (x,y,z) 元组到索引
            normal_map = {} # 映射 (nx,ny,nz) 元组到索引

            current_v_idx = 1
            current_n_idx = 1

            for tri_idx in range(num_triangles):
                v0 = v0s[tri_idx]
                v1 = v0 + e1s[tri_idx]
                v2 = v0 + e2s[tri_idx]
                normal = normals[tri_idx] # 面法线

                # 处理法线
                normal_tuple = tuple(np.round(normal, 6)) # 四舍五入以进行稳健映射
                if normal_tuple not in normal_map:
                    normal_map[normal_tuple] = current_n_idx
                    all_normals.append(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    n_idx = current_n_idx
                    current_n_idx += 1
                else:
                    n_idx = normal_map[normal_tuple]

                # 处理顶点
                v_indices = []
                for v in [v0, v1, v2]:
                    v_tuple = tuple(np.round(v, 6)) # 四舍五入以进行稳健映射
                    if v_tuple not in vertex_map:
                        vertex_map[v_tuple] = current_v_idx
                        all_vertices.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        v_indices.append(current_v_idx)
                        current_v_idx += 1
                    else:
                        v_indices.append(vertex_map[v_tuple])

                # 存储带有顶点和法线索引的面定义
                label = labels[tri_idx]
                face_definitions.append((label, v_indices[0], n_idx, v_indices[1], n_idx, v_indices[2], n_idx))


            # 写入所有唯一的顶点
            f.write("# Vertices\n")
            f.writelines(all_vertices)
            print(f"    写入了 {len(all_vertices)} 个唯一顶点。")

            # 写入所有唯一的法线
            f.write("\n# Normals\n")
            f.writelines(all_normals)
            print(f"    写入了 {len(all_normals)} 个唯一法线。")

            # 按语义标签分组写入面
            f.write("\n# Faces by Semantic Layer\n")
            for label in sorted_labels:
                layer_name = LABEL_MAP.get(label, f"unknown_{label}")
                f.write(f"\ng {layer_name}\n") # 开始图层组
                f.write(f"usemtl {layer_name}\n") # 可选：引用材质
                f.write("s off\n") # 可选：禁用平滑以进行平面着色

                count = 0
                # 查找属于此标签的面
                for face_data in face_definitions:
                    face_label, v1, n1, v2, n2, v3, n3 = face_data
                    if face_label == label:
                        # 使用 v//vn 格式写入面 (顶点索引 // 法线索引)
                        f.write(f"f {v1}//{n1} {v2}//{n2} {v3}//{n3}\n")
                        count += 1
                print(f"    为图层 '{layer_name}' (标签 {label}) 写入了 {count} 个面")


            # --- 写入相机视锥体 ---
            f.write('\n\n# Camera Frustum\n')
            f.write('g camera_frustum\n')
            f.write(f"v {cam_o[0]:.6f} {cam_o[1]:.6f} {cam_o[2]:.6f}\n") # 相机原点
            cam_v_start_idx = current_v_idx # 获取相机原点顶点的索引
            current_v_idx += 1

            # 计算并写入 'far' 距离处的视锥体角点
            corner_indices = []
            for du_norm in [-0.5, 0.5]: # 左/右
                for dv_norm in [-0.5, 0.5]: # 下/上
                    # 计算到角点的方向
                    d_corner = cam_dir + (du_norm * screen_w * right) - (dv_norm * screen_h * up)
                    # 归一化方向
                    d_corner /= np.linalg.norm(d_corner)
                    # 计算角点
                    corner_pt = cam_o + d_corner * far
                    f.write(f"v {corner_pt[0]:.6f} {corner_pt[1]:.6f} {corner_pt[2]:.6f}\n")
                    corner_indices.append(current_v_idx)
                    current_v_idx += 1

            # 定义视锥体边缘的线
            bl, tl, br, tr = corner_indices # 左下, 左上, 右下, 右上
            f.write(f"l {cam_v_start_idx} {bl}\n") # 原点到角点
            f.write(f"l {cam_v_start_idx} {tl}\n")
            f.write(f"l {cam_v_start_idx} {br}\n")
            f.write(f"l {cam_v_start_idx} {tr}\n")
            f.write(f"l {bl} {tl}\n") # 远平面边缘
            f.write(f"l {tl} {tr}\n")
            f.write(f"l {tr} {br}\n")
            f.write(f"l {br} {bl}\n")

            # --- 写入相交点 ---
            f.write('\n\n# Intersection Points (Sampled)\n')
            f.write('g intersection_points\n')
            H_pts, W_pts = pts.shape[:2]
            # 调整步长以获得所需的点密度
            step = max(1, H_pts // 64, W_pts // 64)
            point_v_start_idx = current_v_idx
            num_pts_written = 0
            pts_indices = []

            # 将有效的相交点写入为顶点
            for i in range(0, H_pts, step):
                for j in range(0, W_pts, step):
                    p = pts[i, j]
                    # 检查点是否有效 (不是 NaN)
                    if not np.isnan(p[0]):
                        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                        pts_indices.append(current_v_idx)
                        current_v_idx += 1
                        num_pts_written += 1

            # 将点写入为 OBJ 'p' 元素 (点云)
            if num_pts_written > 0:
                 f.write(f"p {' '.join(map(str, pts_indices))}\n")

        print(f"    成功将带有图层、法线、视锥体和 {num_pts_written} 个采样点的 OBJ 保存到 {filename}")

    except IOError as e:
        print(f"    错误: 无法写入 OBJ 文件 {filename}: {e}")
    except Exception as e:
        print(f"    保存 OBJ 时发生未知错误: {e}")


# --- JSON 参数导出 ---
def save_parameters_json(filename, params):
    """将相关参数保存到 JSON 文件中。"""
    print(f"    将参数保存到 JSON: {filename}")
    try:
        # 将 numpy 数组转换为列表以便 JSON 序列化
        params_serializable = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                params_serializable[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                 params_serializable[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                 params_serializable[key] = int(value)
            elif isinstance(value, dict): # 递归处理嵌套字典
                nested_serializable = {}
                for nk, nv in value.items():
                    if isinstance(nv, np.ndarray):
                        nested_serializable[nk] = nv.tolist()
                    elif isinstance(nv, (np.float32, np.float64)):
                        nested_serializable[nk] = float(nv)
                    elif isinstance(nv, (np.int32, np.int64)):
                        nested_serializable[nk] = int(nv)
                    else:
                        nested_serializable[nk] = nv
                params_serializable[key] = nested_serializable
            else:
                params_serializable[key] = value

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(params_serializable, f, indent=4) # 使用缩进以提高可读性
        print(f"    成功将参数保存到 {filename}")
    except TypeError as e:
        print(f"    错误: 无法将参数序列化为 JSON: {e}")
        # 帮助调试：打印出无法序列化的类型
        problematic_keys = []
        for k, v in params.items():
            try:
                json.dumps({k: v})
            except TypeError:
                problematic_keys.append((k, type(v)))
        if problematic_keys:
             print(f"    可能有问题的键和类型: {problematic_keys}")
        else:
             # 如果直接转储失败，尝试逐个检查嵌套字典
             for k, v in params.items():
                 if isinstance(v, dict):
                     for nk, nv in v.items():
                         try:
                             json.dumps({nk: nv})
                         except TypeError:
                             problematic_keys.append((f"{k}.{nk}", type(nv)))
             if problematic_keys:
                 print(f"    可能有问题的嵌套键和类型: {problematic_keys}")

    except IOError as e:
        print(f"    错误: 无法写入 JSON 文件 {filename}: {e}")
    except Exception as e:
        print(f"    保存 JSON 时发生未知错误: {e}")


# --- 主执行流程 ---
def main():
    global _GPU_AVAILABLE # 允许在 GPU 失败时修改
    total_t0 = _now()

    # --- 输出目录设置 ---
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outd = os.path.join('output', ts)
    try:
        os.makedirs(outd, exist_ok=True)
        print(f"输出目录: {outd}")
    except OSError as e:
        print(f"错误: 无法创建输出目录 {outd}: {e}")
        return

    # --- [1/8] 场景生成 ---
    print('\n[1/8] 构建场景...')
    t0_scene = _now()
    # 构建场景返回原始几何数据
    v0s, e1s, e2s, normals, labels, colors, prim_indices = build_scene()
    num_triangles = len(v0s)
    log_step('场景构建', t0_scene)
    if num_triangles == 0:
        print("错误: 场景构建结果为 0 个三角形。正在退出。")
        return
    print(f"    场景包含 {num_triangles} 个三角形。")
    # 可选的目标三角形数量检查
    # if num_triangles < 200000: print(f"    注意: 三角形数量 ({num_triangles}) 低于目标。")

    # --- [2/8] 构建 BVH ---
    print('\n[2/8] 构建 BVH...')
    t0_bvh = _now()
    # 使用原始图元索引构建 BVH
    flat_nodes, ordered_prim_indices = build_bvh(v0s, e1s, e2s, prim_indices)
    log_step('BVH 构建', t0_bvh)

    # --- [3/8] 为光线追踪器重新排序几何数据 ---
    print('\n[3/8] 根据 BVH 重新排序几何数据...')
    t0_reorder = _now()
    # 根据 BVH 构建返回的顺序重新排序几何数组
    # 这提高了 BVH 遍历期间的内存局部性
    if len(ordered_prim_indices) != num_triangles:
        print(f"错误: BVH 排序索引计数 ({len(ordered_prim_indices)}) 与 "
              f"三角形计数 ({num_triangles}) 不匹配。")
        # 回退：使用原始顺序 (渲染效率较低)
        v0s_reordered = v0s
        e1s_reordered = e1s
        e2s_reordered = e2s
        normals_reordered = normals
        labels_reordered = labels
        colors_reordered = colors
        print("    警告: 使用原始几何数据顺序进行渲染。")
    else:
        try:
            # 应用排序排列
            v0s_reordered = v0s[ordered_prim_indices]
            e1s_reordered = e1s[ordered_prim_indices]
            e2s_reordered = e2s[ordered_prim_indices]
            normals_reordered = normals[ordered_prim_indices]
            labels_reordered = labels[ordered_prim_indices]
            colors_reordered = colors[ordered_prim_indices]
        except IndexError as e:
             print(f"错误: 几何数据重新排序期间发生索引错误: {e}")
             # 回退
             v0s_reordered = v0s
             e1s_reordered = e1s
             e2s_reordered = e2s
             normals_reordered = normals
             labels_reordered = labels
             colors_reordered = colors
             print("    警告: 使用原始几何数据顺序进行渲染。")
    log_step('几何数据重新排序', t0_reorder)

    # --- [4/8] 相机设置 ---
    print('\n[4/8] 设置相机...')
    t0_cam = _now()
    # 相机参数
    cam_o = np.array([-350.0, 25.0, -300.0], dtype=GEOMETRY_DTYPE) # 原点
    cam_t = np.array([50.0, 15.0, 100.0], dtype=GEOMETRY_DTYPE)  # 目标
    cam_up_vec = np.array([0.0, 1.0, 0.0], dtype=GEOMETRY_DTYPE) # 世界向上方向

    # 计算相机框架向量 (前、右、上)
    cam_dir = cam_t - cam_o # 前方向 (尚未归一化)
    norm_cam_dir = np.linalg.norm(cam_dir)
    if norm_cam_dir < INTERSECTION_EPSILON:
        print("错误: 相机原点和目标点太近。")
        return
    cam_dir /= norm_cam_dir # 归一化前方向 (Z 轴)

    # 计算右向量 (X 轴)
    right = np.cross(cam_dir, cam_up_vec)
    norm_right = np.linalg.norm(right)
    # 处理前方向与世界向上方向平行的情况
    if norm_right < INTERSECTION_EPSILON:
        print("警告: 相机前方向与世界向上向量平行。正在调整右向量。")
        # 如果直视上方/下方，则选择替代向上方向
        if abs(cam_dir[1]) > 1.0 - INTERSECTION_EPSILON:
             alt_up = np.array([0.0, 0.0, 1.0 if cam_dir[1] > 0 else -1.0], dtype=GEOMETRY_DTYPE)
             right = np.cross(alt_up, cam_dir) # 重新计算右向量
        else:
            # 如果相机水平但向上向量无效，则默认为世界 X 轴
            right = np.array([1.0, 0.0, 0.0], dtype=GEOMETRY_DTYPE)
        norm_right = np.linalg.norm(right)
        if norm_right < INTERSECTION_EPSILON:
            print("错误: 无法计算有效的右向量。")
            return
    right /= norm_right # 归一化右向量

    # 使用右向量和前向量的叉积计算上向量 (Y 轴)
    up = np.cross(right, cam_dir)
    # 如果 `right` 和 `cam_dir` 已归一化且正交，则无需归一化 `up`

    # 屏幕/图像参数
    W, H = 1920, 1080 # 分辨率
    fov_degrees = 60.0
    fov_radians = np.deg2rad(fov_degrees)
    aspect_ratio = float(W) / float(H)
    # 根据 FoV 计算世界单位中的屏幕高度和宽度
    # 为简单起见，假设屏幕平面沿 cam_dir 距离为 1 个单位
    screen_h = GEOMETRY_DTYPE(2.0 * np.tan(fov_radians / 2.0))
    screen_w = GEOMETRY_DTYPE(screen_h * aspect_ratio)
    log_step('相机设置', t0_cam)
    print(f"    分辨率: {W}x{H}, FoV: {fov_degrees} deg")
    print(f"    相机原点: [{cam_o[0]:.2f}, {cam_o[1]:.2f}, {cam_o[2]:.2f}]")
    print(f"    相机目标: [{cam_t[0]:.2f}, {cam_t[1]:.2f}, {cam_t[2]:.2f}]")

    # --- [5/8] 光线追踪 ---
    print('\n[5/8] 光线追踪...')
    t0_raytrace = _now()
    rgb, depth, sem_lbl, pts = None, None, None, None # 初始化输出变量
    use_gpu = _GPU_AVAILABLE and flat_nodes.shape[0] > 0 # 检查 GPU 是否可用且场景不为空

    if use_gpu:
        print("    尝试 GPU 执行 (CUDA + BVH)...")
        try:
            t_upload_start = _now()
            # --- 将数据上传到 GPU ---
            d_flat_nodes = cuda.to_device(flat_nodes)
            d_v0s = cuda.to_device(v0s_reordered)
            d_e1s = cuda.to_device(e1s_reordered)
            d_e2s = cuda.to_device(e2s_reordered)
            d_normals = cuda.to_device(normals_reordered)
            d_labels = cuda.to_device(labels_reordered)
            d_colors = cuda.to_device(colors_reordered)
            d_sun_dir = cuda.to_device(SUN_DIRECTION) # 上传光照信息
            d_specular_color = cuda.to_device(SPECULAR_COLOR)

            # 在 GPU 上分配输出数组
            d_rgb = cuda.device_array((H, W, 3), dtype=COLOR_DTYPE)
            d_depth = cuda.to_device(np.full((H, W), INF, dtype=DEPTH_DTYPE))
            # 在 GPU 上使用 SKY_LABEL 初始化语义图
            d_sem = cuda.to_device(np.full((H, W), SKY_LABEL, dtype=LABEL_DTYPE))
            d_pts = cuda.to_device(np.full((H, W, 3), np.nan, dtype=POINT_DTYPE))
            log_step('GPU 数据上传', t_upload_start)

            # --- 内核启动配置 ---
            threads_per_block = (16, 16) # 典型块大小
            # 计算覆盖整个图像的网格尺寸
            blocks_per_grid_x = (W + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid_y = (H + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)
            print(f"    启动 CUDA 内核: Grid={blocks_per_grid}, Block={threads_per_block}")

            # --- 执行内核 ---
            t_kernel_start = _now()
            raytrace_cuda_bvh_kernel[blocks_per_grid, threads_per_block](
                d_flat_nodes, d_v0s, d_e1s, d_e2s,
                d_normals,
                d_labels, d_colors,
                cam_o, cam_dir, right, up, # 传递 CPU numpy 数组 (由 Numba 隐式处理)
                d_sun_dir, d_specular_color, # 传递光照设备数组
                screen_w, screen_h, W, H,
                d_rgb, d_depth, d_sem, d_pts # 输出设备数组
            )
            cuda.synchronize() # 等待内核完成
            log_step('GPU BVH 内核执行', t_kernel_start)

            # --- 从 GPU 下载结果 ---
            t_download_start = _now()
            rgb = d_rgb.copy_to_host()
            depth = d_depth.copy_to_host()
            sem_lbl = d_sem.copy_to_host()
            pts = d_pts.copy_to_host()
            log_step('GPU 数据下载', t_download_start)
            print("    GPU 执行成功。")

        except cuda.cudadrv.driver.CudaAPIError as e:
            print(f"\n---!! CUDA API 错误: {e} !!---")
            print("---!! 回退到 CPU 执行。 !!---\n")
            _GPU_AVAILABLE = False # 如果失败，则禁用 GPU 以供将来运行
            use_gpu = False
        except AttributeError as e:
             # 捕获潜在的 .view 错误或其他 Numba/CUDA 问题
             if ".view" in str(e):
                 print(f"\n---!! GPU 错误: {e} !!--- \n---!! 可能是 Numba CUDA .view() 问题。回退到 CPU。 !!---\n")
             else:
                 print(f"\n---!! GPU 属性错误: {e} !!---\n---!! 回退到 CPU 执行。 !!---\n")
             _GPU_AVAILABLE = False
             use_gpu = False
        except Exception as e:
            print(f"\n---!! 未知 GPU 错误: {e} ({type(e).__name__}) !!---")
            print("---!! 回退到 CPU 执行。 !!---\n")
            _GPU_AVAILABLE = False
            use_gpu = False

    # --- CPU 执行回退 ---
    if not use_gpu:
        print("    在 CPU 上执行 (Numba JIT + BVH)...")
        t_cpu_start = _now()
        # 确保 CPU 在可能的情况下使用并行执行
        print("    注意: CPU 执行使用并行 prange。")
        rgb, depth, sem_lbl, pts = raytrace_cpu_bvh(
            flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
            normals_reordered,
            labels_reordered, colors_reordered,
            cam_o, cam_dir, right, up,
            screen_w, screen_h, W, H
        )
        log_step('CPU BVH 光线追踪执行', t_cpu_start)

    log_step('总光线追踪', t0_raytrace)

    # --- [6/8] 图像导出 ---
    print('\n[6/8] 保存输出图像...')
    t0_save_img = _now()

    # 保存 RGB 渲染视图
    if rgb is not None:
        fn_view = os.path.join(outd, f'view_{ts}.png')
        try:
            Image.fromarray(rgb).save(fn_view)
            print(f"    已保存视图: {fn_view}")
        except Exception as e:
            print(f"    保存视图图像时出错: {e}")
    else:
        print("    跳过视图保存 (无 RGB 数据)。")

    # 保存深度图
    if depth is not None:
        fn_depth = os.path.join(outd, f'depth_{ts}.png')
        try:
            # 归一化深度以进行可视化 (排除无限值)
            valid_depth = depth[np.isfinite(depth)]
            if len(valid_depth) > 0:
                dmin = np.min(valid_depth)
                # 使用百分位数作为最大值，以避免异常值主导范围
                dmax_vis = np.percentile(valid_depth, 99.5)
                # 进一步限制可视化最大值以获得更好的近处对比度
                dmax_vis = min(dmax_vis, 1000.0)
                print(f"    深度范围 (有限): {dmin:.2f} 到 {np.max(valid_depth):.2f} "
                      f"(可视化最大值: {dmax_vis:.2f})")

                # 如果范围很小，避免除零
                if dmax_vis <= dmin: dmax_vis = dmin + 1.0
                scale = (dmax_vis - dmin)
                scale = max(scale, INTERSECTION_EPSILON) # 确保 scale 为正

                # 归一化、裁剪并缩放到 [0, 254]，对无穷大使用 255
                depth_normalized = (depth - dmin) / scale
                depth_clipped = np.clip(depth_normalized * 254.0, 0, 254)
                # 将最初为无穷大的像素 (天空/未命中) 分配为 255
                dmap = np.where(np.isfinite(depth), depth_clipped, 255).astype(np.uint8)

                Image.fromarray(dmap, 'L').save(fn_depth) # 保存为灰度图
                print(f"    已保存深度图: {fn_depth}")
            else:
                print("    跳过深度图保存 (无有效的有限深度值)。")
                # 保存黑色图像作为占位符
                Image.new('L', (W, H), 0).save(fn_depth)
        except RuntimeWarning as e:
             # 捕获百分位数计算等期间的警告
             print(f"    保存深度图时出现运行时警告: {e}")
             try: Image.new('L', (W, H), 0).save(fn_depth) # 尝试保存黑色图像
             except: pass # 忽略回退保存期间的错误
        except Exception as e:
            print(f"    保存深度图像时出错: {e}")
    else:
        print("    跳过深度图保存 (无深度数据)。")

    # 保存语义图
    if sem_lbl is not None:
        fn_sem = os.path.join(outd, f'semantic_{ts}.png')
        try:
            # 为语义图创建空的 RGB 图像
            sem_img = np.zeros((H, W, 3), dtype=COLOR_DTYPE)
            # 直接使用 C 颜色映射
            unique_labels = np.unique(sem_lbl)
            # print(f"语义图中的唯一标签: {unique_labels}") # 调试行

            for label_id in unique_labels:
                 if label_id in C:
                     mask = (sem_lbl == label_id)
                     sem_img[mask] = C[label_id]
                 else:
                     # 可选：将未知标签着色为黑色或打印警告
                     # sem_img[sem_lbl == label_id] = PALETTE_BACKGROUND_COLOR
                     print(f"    警告: 在颜色映射 C 中未找到语义标签 {label_id}。")

            Image.fromarray(sem_img).save(fn_sem)
            print(f"    已保存语义图: {fn_sem}")
        except Exception as e:
            print(f"    保存语义图像时出错: {e}")
    else:
        print("    跳过语义图保存 (无语义数据)。")

    log_step('图像导出', t0_save_img)

    # --- [7/8] OBJ 导出 ---
    print('\n[7/8] 保存组合 OBJ 文件...')
    t0_obj = _now()
    fn_obj = os.path.join(outd, f'combined_{ts}.obj')
    # 确定用于视锥体可视化的远距离
    far_dist = 1000.0 # 默认远距离
    if depth is not None:
        valid_depth = depth[np.isfinite(depth)]
        if len(valid_depth) > 0:
            # 将远距离设置为略微超出最远命中点
            far_dist = max(far_dist, np.max(valid_depth) * 1.1)

    if pts is not None:
         # 使用原始几何数据调用修改后的保存函数以获取正确的标签
         save_combined_obj(fn_obj, v0s, e1s, e2s, normals, labels, # 原始数据
                           cam_o, cam_dir, right, up, screen_w, screen_h,
                           pts, far=far_dist)
    else:
        print("    跳过 OBJ 保存 (无相交点数据)。")
    log_step('OBJ 导出', t0_obj)

    # --- [8/8] JSON 参数导出 ---
    print('\n[8/8] 保存参数 JSON 文件...')
    t0_json = _now()
    fn_json = os.path.join(outd, f'parameters_{ts}.json')
    # 将参数收集到字典中
    parameters = {
        "timestamp": ts,
        "resolution": {"width": W, "height": H},
        "fov_degrees": fov_degrees,
        "camera": {
            "origin": cam_o,
            "target": cam_t,
            "direction": cam_dir,
            "world_up": cam_up_vec,
            "frame_right": right,
            "frame_up": up,
            "screen_width_world": screen_w,
            "screen_height_world": screen_h,
        },
        "lighting": {
            "sun_direction": SUN_DIRECTION,
            "sun_intensity": SUN_INTENSITY,
            "ambient_light": AMBIENT_LIGHT,
            "specular_color": SPECULAR_COLOR,
            "specular_exponent": SPECULAR_EXPONENT,
            "specular_coefficient_Ks": Ks,
        },
        "rendering": {
            "gpu_available_initial": _GPU_AVAILABLE, # 最初是否检测到 GPU？
            "gpu_used": use_gpu, # 渲染时是否实际使用了 GPU？
            "shadow_bias": SHADOW_BIAS,
            "intersection_epsilon": INTERSECTION_EPSILON,
        },
        "scene": {
            "total_triangles": num_triangles,
            "label_map": LABEL_MAP,
            "color_map_rgb": {str(k): v.tolist() for k, v in C.items()}, # 转换颜色和键以用于 JSON
        },
        "bvh": {
             "max_leaf_size": BVH_MAX_LEAF_SIZE,
             "node_count": flat_nodes.shape[0] if flat_nodes is not None else 0,
        }
        # 根据需要添加更多参数
    }
    save_parameters_json(fn_json, parameters)
    log_step('JSON 参数导出', t0_json)


    # --- 完成 ---
    print('\n[完成]')
    total_time = _now() - total_t0
    print(f"总执行时间: {total_time:.2f}s")
    print(f"输出已保存至: {outd}")


# --- 脚本入口点 ---
if __name__ == '__main__':
    main()
