# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import math # 需要 math 模块

import numpy as np
from PIL import Image
from numba import njit, prange

# -- GPU Availability Check --------------------------------------------------
try:
    from numba import cuda
    # Check if a CUDA device is available and context can be created
    if cuda.is_available():
        # Try to detect and get device name
        try:
            cuda.detect()  # This will print device info if successful
            device = cuda.get_current_device()
            print(f"CUDA 可用: True")
            print(f"使用 GPU: {device.name.decode()}")
            _GPU_AVAILABLE = True
        except Exception as e_detect:
            print(f"CUDA 检测或获取设备名时出错: {e_detect}")
            print("CUDA 可能仍可用，但信息不完整。")
            _GPU_AVAILABLE = True # Let's try anyway
    else:
        print("CUDA 不可用: 未检测到兼容设备或驱动程序.")
        _GPU_AVAILABLE = False

except ImportError:
    print("Numba CUDA 扩展未安装或导入失败.")
    _GPU_AVAILABLE = False
except Exception as e:
    print(f"CUDA 初始化时发生未知错误: {e}")
    _GPU_AVAILABLE = False


# --- 常量定义 ---
GEOMETRY_DTYPE = np.float32
INDEX_DTYPE = np.int32
COLOR_DTYPE = np.uint8
LABEL_DTYPE = np.int32
DEPTH_DTYPE = np.float32
POINT_DTYPE = np.float32
BVH_NODE_DTYPE = np.float32 # BVH 节点数据类型 (AABB用)
INF = GEOMETRY_DTYPE(np.inf)
EPSILON = GEOMETRY_DTYPE(1e-6) # Epsilon for general float comparisons
BVH_MAX_LEAF_SIZE = 4 # BVH 叶子节点最大包含的图元数量

# --- 全局颜色映射 ---
C = {
    0: np.array([150, 200, 150], dtype=COLOR_DTYPE), # 草地 (背景/默认)
    1: np.array([200, 200, 200], dtype=COLOR_DTYPE), # 建筑
    2: np.array([220, 180, 50], dtype=COLOR_DTYPE),  # 道路
    3: np.array([90, 140, 210], dtype=COLOR_DTYPE),  # 水面
    4: np.array([210, 80, 80], dtype=COLOR_DTYPE),   # 摩天楼/特殊建筑
    5: np.array([0, 140, 0], dtype=COLOR_DTYPE),     # 树木
    6: np.array([120, 120, 120], dtype=COLOR_DTYPE)  # 山脉
}
PALETTE_BACKGROUND_COLOR = (0, 0, 0) # 语义图像背景色 (黑色)
DEFAULT_SKY_COLOR = np.array([135, 206, 235], dtype=COLOR_DTYPE) # 淡蓝色天空

# --- 工具函数 ---
def _now() -> float:
    """高精度时间戳"""
    return time.perf_counter()

def log_step(title: str, t0: float) -> None:
    """阶段耗时打印"""
    print(f"    {title} 用时 {_now() - t0:.3f}s") # Increased precision

# --- 随机数生成器 (可复现) ---
RAND = np.random.default_rng(seed=0)

# --- 几何体生成函数 ---
# (保持不变)
def create_box(center, size):
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    verts = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ], dtype=GEOMETRY_DTYPE)
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
        (3, 2, 6), (3, 6, 7), (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2),
    ] # 12 triangles
    return verts, faces

def create_prism(center, height, radius, sides=5):
    cx, cy, cz = center
    half = height / 2
    verts = []
    for i in range(sides):
        th = 2 * np.pi * i / sides
        x, z = cx + radius * np.cos(th), cz + radius * np.sin(th)
        verts.append((x, cy - half, z))
        verts.append((x, cy + half, z))
    verts = np.asarray(verts, dtype=GEOMETRY_DTYPE)
    faces = []
    for i in range(sides):
        i0, i1 = 2 * i, (2 * (i + 1)) % (2 * sides)
        faces.append((i0, i1, i1 + 1))
        faces.append((i0, i1 + 1, i0 + 1)) # 2*sides triangles
    return verts, faces

def create_skyscraper(center, base, height, levels=8, taper=0.85):
    verts, faces = [], []
    cx, cy, cz = center
    seg_h = height / levels
    cur_w = base
    offset = 0 # Start from the base y provided in 'center'
    for lv in range(levels):
        seg_center = (cx, cy + offset + seg_h / 2, cz)
        v, f = create_box(seg_center, (cur_w, seg_h, cur_w))
        o = len(verts)
        verts.extend(v)
        faces.extend([(a + o, b + o, c + o) for a, b, c in f])
        offset += seg_h
        cur_w *= taper # 12 * levels triangles
    return np.asarray(verts, dtype=GEOMETRY_DTYPE), faces

def create_cylinder(center, height, radius, *, segments: int = 16):
    cx, cy, cz = center
    verts = []
    half_h = height / 2
    for i in range(segments):
        th = 2 * np.pi * i / segments
        x = cx + radius * np.cos(th)
        z = cz + radius * np.sin(th)
        verts.append((x, cy - half_h, z))
        verts.append((x, cy + half_h, z))
    verts = np.asarray(verts, dtype=GEOMETRY_DTYPE)
    faces = []
    for i in range(segments):
        i0 = 2 * i
        i1 = (i0 + 2) % (2 * segments)
        faces.append((i0, i1, i1 + 1))
        faces.append((i0, i1 + 1, i0 + 1)) # 2 * segments triangles
    return verts, faces

def create_cone_tree(base_center, height, radius, *, segments: int = 10):
    cx, cy, cz = base_center # Base y is included here
    trunk_h = height * 0.3
    trunk_r = radius * 0.2
    trunk_base_cy = cy + trunk_h / 2 # Center y for the trunk cylinder
    trunk_verts, trunk_faces = create_cylinder(
        center=(cx, trunk_base_cy, cz),
        height=trunk_h, radius=trunk_r, segments=max(6, segments // 2))

    crown_h = height * 0.7
    crown_base_y = cy + trunk_h # Y level where the cone base starts
    apex = np.array([cx, crown_base_y + crown_h, cz], dtype=GEOMETRY_DTYPE) # Cone top vertex
    circle = []
    for i in range(segments):
        th = 2 * np.pi * i / segments
        x, z = cx + radius * np.cos(th), cz + radius * np.sin(th)
        circle.append((x, crown_base_y, z)) # Vertices at the base of the cone
    circle = np.asarray(circle, dtype=GEOMETRY_DTYPE)

    cone_faces = []
    num_circle_verts = len(circle)
    for i in range(num_circle_verts):
        # Apex index is 0 within cone_verts
        # Current circle vertex index is i + 1
        # Next circle vertex index is (i + 1) % num_circle_verts + 1
        cone_faces.append((0, i + 1, (i + 1) % num_circle_verts + 1))

    cone_verts = np.vstack((apex.reshape(1, 3), circle))

    # Combine trunk and cone
    verts = np.vstack((trunk_verts, cone_verts))
    # Adjust cone face indices by the number of trunk vertices
    faces = trunk_faces + [(a + len(trunk_verts), b + len(trunk_verts), c + len(trunk_verts))
                           for a, b, c in cone_faces]
    return verts, faces

def create_mountain_range(x_start, x_end, z_pos, *, segs=200, depth=25, h_min=18, h_max=50):
    xs = np.linspace(x_start, x_end, segs + 1, dtype=GEOMETRY_DTYPE)
    num_noise_points = max(5, segs // 10)
    noise_xs = np.linspace(x_start, x_end, num_noise_points)
    noise_ys_raw = RAND.uniform(-1, 1, noise_xs.shape)
    # Use some smoothing (e.g., moving average) on noise_ys_raw if needed
    noise_ys = np.interp(xs, noise_xs, noise_ys_raw) # Linear interpolation
    base_heights = np.interp(noise_ys, (-1, 1), (h_min, h_max)) # Map noise to height range

    verts, faces = [], []
    zero_y = GEOMETRY_DTYPE(0) # Base of the mountain segment will be at y=0 before final adjustment
    z_pos_dtype = GEOMETRY_DTYPE(z_pos)
    depth_dtype = GEOMETRY_DTYPE(depth)
    for i in range(segs):
        x0, x1 = xs[i], xs[i + 1]
        h0, h1 = base_heights[i], base_heights[i + 1]
        h0_dtype, h1_dtype = GEOMETRY_DTYPE(h0), GEOMETRY_DTYPE(h1)
        # Define 8 vertices for the segment box
        v = [(x0, zero_y, z_pos_dtype), (x1, zero_y, z_pos_dtype), # 0, 1: bottom front
             (x0, h0_dtype, z_pos_dtype), (x1, h1_dtype, z_pos_dtype), # 2, 3: top front
             (x0, zero_y, z_pos_dtype + depth_dtype), (x1, zero_y, z_pos_dtype + depth_dtype), # 4, 5: bottom back
             (x0, h0_dtype, z_pos_dtype + depth_dtype), (x1, h1_dtype, z_pos_dtype + depth_dtype)] # 6, 7: top back
        idx0 = len(verts) # Starting index for this segment's vertices
        verts.extend(v)
        f = lambda a, b, c: (idx0 + a, idx0 + b, idx0 + c) # Helper to offset indices
        # Create faces for the mountain segment (10 triangles for a closed segment)
        faces.append(f(0, 1, 3)); faces.append(f(0, 3, 2)) # Front face
        faces.append(f(4, 6, 7)); faces.append(f(4, 7, 5)) # Back face
        faces.append(f(0, 4, 6)); faces.append(f(0, 6, 2)) # Left face
        faces.append(f(1, 3, 7)); faces.append(f(1, 7, 5)) # Right face
        faces.append(f(2, 6, 7)); faces.append(f(2, 7, 3)) # Top face
        # faces.append(f(0, 5, 4)); faces.append(f(0, 1, 5)) # Bottom face (optional)
    return np.asarray(verts, dtype=GEOMETRY_DTYPE), faces


# --- 场景构建 ---
def build_scene():
    """构建整个场景几何体"""
    tris_geom, tris_labs, tris_cols = [], [], []

    # --- 对象数量 ---
    num_buildings = 1000
    num_skyscrapers = 150
    num_prisms = 150
    num_trees = 3500
    mountain_segs1 = 200
    mountain_segs2 = 160
    print(f"    目标对象数量: 建筑={num_buildings}, 摩天楼={num_skyscrapers}, 棱柱={num_prisms}, 树木={num_trees}")

    # Helper to add objects
    def add_object(verts, faces, label, color_map):
        color = color_map[label]
        for face_indices in faces:
            if all(idx < len(verts) for idx in face_indices):
                if len(face_indices) == 3:
                    a, b, c = face_indices
                    tris_geom.append((verts[a], verts[b], verts[c]))
                    tris_labs.append(label)
                    tris_cols.append(color)
                # else: print(f"警告: 非三角形面索引 {face_indices}，已跳过。") # Optional warning
            # else: print(f"警告: 面索引 {face_indices} 超出顶点数组范围 {len(verts)}，已跳过。") # Optional warning

    # --- 地面和基准高度 ---
    ground_y = -0.1 # Lower ground slightly more
    base_y = ground_y + 0.01 # Base level for objects slightly above ground

    # 地面
    g_v, g_f = create_box((0, ground_y + 0.005, 0), (200, 0.01, 200)) # Wider ground
    add_object(g_v, g_f, 0, C) # Label 0: Grass

    # 湖泊 (ensure surface is above base_y)
    lake_y = base_y + 0.02
    lake_v, lake_f = create_box((40, lake_y, 50), (50, 0.01, 40)) # Thin lake surface
    add_object(lake_v, [lake_f[i] for i in [2, 3]], 3, C) # Label 3: Water (only top faces)

    # 道路网格 (slightly above base_y)
    road_y = base_y + 0.03
    road_w = 4.0
    road_h = 0.01 # Thin roads
    grid_coords = np.linspace(-80, 80, 13, dtype=GEOMETRY_DTYPE) # Wider grid, more roads
    for x in grid_coords:
        v, f = create_box((x, road_y, 0), (road_w, road_h, 200.0 + road_w))
        add_object(v, f, 2, C) # Label 2: Road
    for z in grid_coords:
        v, f = create_box((0, road_y, z), (200.0 + road_w, road_h, road_w))
        add_object(v, f, 2, C)

    # 放置区域边界
    min_coord, max_coord = -85, 85 # Slightly larger placement area
    lake_x_min, lake_x_max = 15, 65
    lake_z_min, lake_z_max = 30, 70

    # --- 放置物体 (确保物体基座在 base_y) ---

    # 建筑
    building_count = 0
    for _ in range(num_buildings * 3): # Try placing more initially
        if building_count >= num_buildings: break
        x = RAND.uniform(min_coord, max_coord)
        z = RAND.uniform(min_coord, max_coord)
        if lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max: continue # Avoid lake
        w = RAND.uniform(5, 12)
        h = RAND.uniform(8, 18)
        center_y = base_y + h / 2 # Center based on base_y
        v, f = create_box((x, center_y, z), (w, h, w))
        add_object(v, f, 1, C)
        building_count += 1

    # 摩天楼
    skyscraper_count = 0
    for _ in range(num_skyscrapers * 3):
        if skyscraper_count >= num_skyscrapers: break
        x = RAND.uniform(min_coord + 5, max_coord - 5)
        z = RAND.uniform(min_coord + 5, max_coord - 5)
        if lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max: continue
        base_w = RAND.uniform(7, 12)
        h = RAND.uniform(40, 60)
        levels = RAND.integers(6, 10)
        taper = RAND.uniform(0.8, 0.95)
        # Skyscraper base starts at base_y
        v, f = create_skyscraper((x, base_y, z), base_w, h, levels=levels, taper=taper)
        add_object(v, f, 4, C)
        skyscraper_count += 1

    # 棱柱建筑
    prism_count = 0
    for _ in range(num_prisms * 3):
        if prism_count >= num_prisms: break
        x = RAND.uniform(min_coord, max_coord)
        z = RAND.uniform(min_coord, max_coord)
        if lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max: continue
        r = RAND.uniform(4, 7)
        h = RAND.uniform(12, 22)
        sides = RAND.integers(4, 7)
        center_y = base_y + h / 2
        v, f = create_prism((x, center_y, z), h, r, sides=sides)
        add_object(v, f, 4, C) # Also label 4
        prism_count += 1

    # 树木
    tree_count = 0
    for _ in range(num_trees * 3):
        if tree_count >= num_trees: break
        x = RAND.uniform(min_coord - 10, max_coord + 10) # Wider range for trees
        z = RAND.uniform(min_coord - 10, max_coord + 10)
        on_lake = (lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max)
        on_road = False
        road_clearance = road_w / 1.5 # Wider clearance for trees
        for gx in grid_coords:
             if abs(x - gx) < road_clearance: on_road = True; break
        if not on_road:
            for gz in grid_coords:
                if abs(z - gz) < road_clearance: on_road = True; break
        if on_lake or on_road: continue

        h = RAND.uniform(6, 11)
        r = RAND.uniform(1.5, 2.5)
        segs = RAND.integers(8, 14)
        # Tree base starts at base_y
        v, f = create_cone_tree((x, base_y, z), h, r, segments=segs)
        add_object(v, f, 5, C)
        tree_count += 1

    # 山脉 (Ensure base is at base_y)
    m_v, m_f = create_mountain_range(x_start=-120, x_end=120, z_pos=95, segs=mountain_segs1, depth=35, h_min=25, h_max=65)
    m_v[:, 1] += base_y # Adjust height relative to base_y
    add_object(m_v, m_f, 6, C)

    m_v2, m_f2 = create_mountain_range(x_start=-120, x_end=120, z_pos=130, segs=mountain_segs2, depth=30, h_min=20, h_max=50)
    m_v2[:, 1] += base_y # Adjust height
    add_object(m_v2, m_f2, 6, C)

    # --- 数据转换 ---
    N = len(tris_geom)
    if N == 0:
        print("警告: 场景为空!")
        return (np.empty((0, 3), dtype=GEOMETRY_DTYPE), np.empty((0, 3), dtype=GEOMETRY_DTYPE),
                np.empty((0, 3), dtype=GEOMETRY_DTYPE), np.empty(0, dtype=LABEL_DTYPE),
                np.empty((0, 3), dtype=COLOR_DTYPE), np.empty(0, dtype=INDEX_DTYPE))

    print(f"    实际生成对象: 建筑={building_count}, 摩天楼={skyscraper_count}, 棱柱={prism_count}, 树木={tree_count}")
    print(f"    总计三角形数量: {N}")

    v0s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    e1s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    e2s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    labels = np.array(tris_labs, dtype=LABEL_DTYPE)
    colors = np.array(tris_cols, dtype=COLOR_DTYPE)

    for i, (v0, v1, v2) in enumerate(tris_geom):
        v0s[i] = v0
        e1s[i] = np.subtract(v1, v0, dtype=GEOMETRY_DTYPE)
        e2s[i] = np.subtract(v2, v0, dtype=GEOMETRY_DTYPE)

    prim_indices = np.arange(N, dtype=INDEX_DTYPE)

    return v0s, e1s, e2s, labels, colors, prim_indices


# --- BVH 相关 -------------------------------------------------------------

BVH_NODE_FIELDS = 8

@njit(fastmath=True)
def calculate_tri_aabb_numba(v0, e1, e2):
    """Numba-optimized AABB calculation for a single triangle"""
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
    """Calculate the bounding box enclosing a set of primitives"""
    num_tris = len(indices)
    if num_tris == 0:
        return (np.full(3, INF, dtype=BVH_NODE_DTYPE), np.full(3, -INF, dtype=BVH_NODE_DTYPE))

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

# Helper function for int32 to float32 bit-casting (Numba CPU)
@njit
def int32_to_float32_bits(val_int32):
    """Reinterprets the bits of an int32 as a float32."""
    int_array = np.array([val_int32], dtype=INDEX_DTYPE)
    return int_array.view(BVH_NODE_DTYPE)[0]

# Helper function for float32 to int32 bit-casting (Numba CPU)
@njit
def float32_to_int32_bits(val_float32):
    """Reinterprets the bits of a float32 as an int32."""
    float_array = np.array([val_float32], dtype=BVH_NODE_DTYPE)
    return float_array.view(INDEX_DTYPE)[0]


# --- BVH Build (Recursive Part - Numba JITted) ---
@njit
def recursive_build_numba(
    current_node_idx, nodes_used, flat_nodes,
    prim_indices, start_idx, end_idx,
    tri_aabbs_min, tri_aabbs_max, tri_centers
):
    """Recursive BVH build function, optimized with Numba."""
    num_prims = end_idx - start_idx
    node = flat_nodes[current_node_idx]

    # 1. Calculate bounds
    indices_slice = prim_indices[start_idx:end_idx]
    aabb_min, aabb_max = calculate_bounds(indices_slice, tri_aabbs_min, tri_aabbs_max)
    node[0:3] = aabb_min
    node[3:6] = aabb_max

    # 2. Leaf node check
    if num_prims <= BVH_MAX_LEAF_SIZE:
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx))
        # *** MODIFICATION: Store negative count for leaves ***
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims)) # Store negative count
        return

    # 3. Internal node: Choose split axis and position
    extent = aabb_max - aabb_min
    split_axis = np.argmax(extent)

    if extent[split_axis] < EPSILON: # Handle zero extent (make leaf)
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx))
        # *** MODIFICATION: Store negative count for leaves ***
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims)) # Store negative count
        return

    split_coord = (aabb_min[split_axis] + aabb_max[split_axis]) * 0.5

    # 4. Partition primitives
    mid_point = start_idx
    for i in range(start_idx, end_idx):
        prim_idx = prim_indices[i]
        if tri_centers[prim_idx, split_axis] < split_coord:
            prim_indices[i], prim_indices[mid_point] = prim_indices[mid_point], prim_indices[i]
            mid_point += 1

    # Handle ineffective split
    if mid_point == start_idx or mid_point == end_idx:
        mid_point = start_idx + num_prims // 2

    # 5. Allocate children nodes
    left_child_idx = nodes_used[0]
    nodes_used[0] += 1
    right_child_idx = nodes_used[0]
    nodes_used[0] += 1

    # Store children indices (non-negative)
    node[6] = int32_to_float32_bits(INDEX_DTYPE(left_child_idx))
    node[7] = int32_to_float32_bits(INDEX_DTYPE(right_child_idx)) # Store right child index (non-negative)

    # 6. Recursively build children
    recursive_build_numba(left_child_idx, nodes_used, flat_nodes, prim_indices,
                          start_idx, mid_point,
                          tri_aabbs_min, tri_aabbs_max, tri_centers)
    recursive_build_numba(right_child_idx, nodes_used, flat_nodes, prim_indices,
                          mid_point, end_idx,
                          tri_aabbs_min, tri_aabbs_max, tri_centers)


def build_bvh(v0s, e1s, e2s, prim_indices_in):
    """Builds the BVH using a top-down, middle-split approach."""
    N = len(prim_indices_in)
    if N == 0:
        return np.empty((0, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE), np.empty(0, dtype=INDEX_DTYPE)

    t0_precompute = _now()
    # 1. Precompute AABBs and centroids
    tri_aabbs_min = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_aabbs_max = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_centers = np.empty((N, 3), dtype=BVH_NODE_DTYPE)

    @njit(parallel=True)
    def precompute_bounds_centers_parallel(num_tris, v0s_n, e1s_n, e2s_n,
                                           aabbs_min_out, aabbs_max_out, centers_out):
        for i in prange(num_tris):
            v0, e1, e2 = v0s_n[i], e1s_n[i], e2s_n[i]
            aabb_min, aabb_max = calculate_tri_aabb_numba(v0, e1, e2)
            aabbs_min_out[i] = aabb_min
            aabbs_max_out[i] = aabb_max
            centers_out[i] = (aabb_min + aabb_max) * 0.5

    precompute_bounds_centers_parallel(N, v0s, e1s, e2s, tri_aabbs_min, tri_aabbs_max, tri_centers)
    log_step('BVH Precomputation', t0_precompute)

    # 2. Allocate node space
    max_nodes = max(1, 2 * N - 1) # Ensure at least 1 node if N=1
    flat_nodes = np.zeros((max_nodes, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE)

    # 3. Initialize recursive build
    ordered_prim_indices = np.copy(prim_indices_in)
    nodes_used = np.array([1], dtype=INDEX_DTYPE) # Counter passed as mutable array

    t0_recursive = _now()
    # Call the Numba-jitted recursive function
    recursive_build_numba(
        0, nodes_used, flat_nodes,
        ordered_prim_indices, 0, N,
        tri_aabbs_min, tri_aabbs_max, tri_centers
    )
    log_step('BVH Recursive Build', t0_recursive)

    # 4. Trim unused nodes
    actual_nodes_used = nodes_used[0]
    flat_nodes = flat_nodes[:actual_nodes_used]

    print(f"    BVH 构建完成: {actual_nodes_used} 个节点.")
    if actual_nodes_used > 0:
        root_min = flat_nodes[0, 0:3]
        root_max = flat_nodes[0, 3:6]
        print(f"    根节点 AABB Min: [{root_min[0]:.2f}, {root_min[1]:.2f}, {root_min[2]:.2f}], "
              f"Max: [{root_max[0]:.2f}, {root_max[1]:.2f}, {root_max[2]:.2f}]")
        # Debug root node type
        # root_info = float32_to_int32_bits(flat_nodes[0, 7])
        # print(f"    根节点 Info (node[7] as int): {root_info}")

    return flat_nodes, ordered_prim_indices


# --- 光线-三角形相交 (CPU) ---
@njit(fastmath=True)
def intersect_ray_triangle_cpu(orig, dir, v0, e1, e2):
    """Moller-Trumbore ray-triangle intersection test."""
    h = np.cross(dir, e2)
    a = np.dot(e1, h)

    if abs(a) < EPSILON: # Ray parallel to triangle plane
        return INF

    f = GEOMETRY_DTYPE(1.0) / a
    s = orig - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return INF

    q = np.cross(s, e1)
    v = f * np.dot(dir, q)

    if v < 0.0 or u + v > 1.0:
        return INF

    # Calculate t, the distance to intersection
    t = f * np.dot(e2, q)

    return t if t > EPSILON else INF # Return t only if intersection is in front

# --- 光线-AABB 相交 (CPU) ---
@njit(fastmath=True)
def intersect_ray_aabb_cpu(orig, dir_inv, tmin_global, node_aabb_min, node_aabb_max):
    """Ray-AABB intersection test (Slab Test) for CPU BVH traversal."""
    t_near = -INF
    t_far = INF
    for k in range(3):
        inv_d = dir_inv[k]
        aabb_min_k = node_aabb_min[k]
        aabb_max_k = node_aabb_max[k]

        t1 = (aabb_min_k - orig[k]) * inv_d
        t2 = (aabb_max_k - orig[k]) * inv_d

        if t1 > t2: t1, t2 = t2, t1 # Ensure t1 is near, t2 is far

        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

        # Early exit conditions:
        if t_near >= t_far or t_far < EPSILON or t_near >= tmin_global:
            return False # Miss

    return True # Hit


# --- 光线追踪 (CPU - BVH) ---
@njit(parallel=True, fastmath=True) # Re-enable parallel
# @njit(fastmath=True) # Use non-parallel for debugging
def raytrace_cpu_bvh(
    flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
    labels_reordered, colors_reordered,
    cam_o, cam_dir, right, up,
    screen_w, screen_h, W, H
):
    """Performs ray tracing on the CPU using the BVH."""
    rgb = np.zeros((H, W, 3), dtype=COLOR_DTYPE)
    depth = np.full((H, W), INF, dtype=DEPTH_DTYPE)
    sem = np.zeros((H, W), dtype=LABEL_DTYPE)
    pts = np.full((H, W, 3), np.nan, dtype=POINT_DTYPE)

    inv_W = GEOMETRY_DTYPE(1.0 / W)
    inv_H = GEOMETRY_DTYPE(1.0 / H)
    num_nodes = flat_nodes.shape[0]
    num_tris_total = len(v0s_reordered) # Get total number of triangles

    if num_nodes == 0:
        for i in prange(H): # Use prange with parallel=True
             for j in range(W): rgb[i, j] = DEFAULT_SKY_COLOR
        return rgb, depth, sem, pts

    BVH_CPU_STACK_SIZE = 64

    for i in prange(H): # Use prange with parallel=True
        # Per-thread locals
        node_stack = np.empty(BVH_CPU_STACK_SIZE, dtype=INDEX_DTYPE)
        d = np.empty(3, dtype=GEOMETRY_DTYPE)
        dir_inv = np.empty(3, dtype=GEOMETRY_DTYPE)

        for j in range(W):
            # 1. Calculate ray direction
            u = (GEOMETRY_DTYPE(j) + 0.5) * inv_W - 0.5
            v = (GEOMETRY_DTYPE(i) + 0.5) * inv_H - 0.5
            d[0] = cam_dir[0] + u * screen_w * right[0] - v * screen_h * up[0]
            d[1] = cam_dir[1] + u * screen_w * right[1] - v * screen_h * up[1]
            d[2] = cam_dir[2] + u * screen_w * right[2] - v * screen_h * up[2]

            norm_sq = d[0]**2 + d[1]**2 + d[2]**2
            if norm_sq < EPSILON**2: continue
            inv_norm = GEOMETRY_DTYPE(1.0) / math.sqrt(norm_sq)
            d *= inv_norm

            for k in range(3):
                if abs(d[k]) < EPSILON:
                    dir_inv[k] = math.copysign(INF, d[k])
                else:
                    dir_inv[k] = GEOMETRY_DTYPE(1.0) / d[k]

            # 2. Initialize traversal
            tmin = INF
            hit_prim_idx = -1
            stack_ptr = 0
            node_stack[stack_ptr] = 0
            stack_ptr += 1

            # 3. BVH Traversal Loop
            while stack_ptr > 0:
                stack_ptr -= 1
                node_idx = node_stack[stack_ptr]

                if node_idx < 0 or node_idx >= num_nodes: continue

                node = flat_nodes[node_idx]
                node_aabb_min = node[0:3]
                node_aabb_max = node[3:6]

                aabb_hit = intersect_ray_aabb_cpu(cam_o, dir_inv, tmin, node_aabb_min, node_aabb_max)

                if not aabb_hit: continue

                # *** MODIFICATION: Check sign of node[7] to determine node type ***
                info_bits = node[7]
                info_val = float32_to_int32_bits(info_bits)

                if info_val < 0: # Leaf Node (negative count stored)
                    prim_count = -info_val # Get positive count
                    prim_offset_bits = node[6]
                    prim_offset = float32_to_int32_bits(prim_offset_bits)

                    for p_local_idx in range(prim_count):
                        p_idx = prim_offset + p_local_idx
                        if p_idx < num_tris_total:
                            t = intersect_ray_triangle_cpu(cam_o, d,
                                                           v0s_reordered[p_idx],
                                                           e1s_reordered[p_idx],
                                                           e2s_reordered[p_idx])
                            if t < tmin:
                                tmin = t
                                hit_prim_idx = p_idx

                else: # Internal Node (info_val >= 0 is right_child_idx)
                    left_child_idx_bits = node[6]
                    left_child_idx = float32_to_int32_bits(left_child_idx_bits)
                    right_child_idx = info_val # Already have the right child index

                    if stack_ptr + 2 <= BVH_CPU_STACK_SIZE:
                        # Check validity before pushing
                        if left_child_idx >= 0 and left_child_idx < num_nodes:
                            node_stack[stack_ptr] = left_child_idx
                            stack_ptr += 1
                        if right_child_idx >= 0 and right_child_idx < num_nodes:
                            node_stack[stack_ptr] = right_child_idx
                            stack_ptr += 1
                    # else: # Handle stack overflow

            # 4. Process hit result
            if hit_prim_idx >= 0:
                depth[i, j] = tmin
                sem[i, j] = labels_reordered[hit_prim_idx]
                rgb[i, j] = colors_reordered[hit_prim_idx]
                hit_point = cam_o + tmin * d
                pts[i, j, 0] = hit_point[0]
                pts[i, j, 1] = hit_point[1]
                pts[i, j, 2] = hit_point[2]
            else:
                rgb[i, j] = DEFAULT_SKY_COLOR
                # depth, sem, pts remain INF, 0, NaN

    return rgb, depth, sem, pts


# --- GPU 内核 ---
if _GPU_AVAILABLE:

    # --- 光线-三角形相交 (GPU 设备函数) ---
    @cuda.jit(device=True, inline=True)
    def ray_tri_intersect_gpu(orig, dir, v0, e1, e2):
        """Moller-Trumbore ray-triangle intersection test for GPU."""
        eps_gpu = GEOMETRY_DTYPE(1e-6)
        inf_gpu = GEOMETRY_DTYPE(1e20)

        h0 = dir[1] * e2[2] - dir[2] * e2[1]
        h1 = dir[2] * e2[0] - dir[0] * e2[2]
        h2 = dir[0] * e2[1] - dir[1] * e2[0]
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


    # --- 光线-AABB 相交 (GPU 设备函数) ---
    @cuda.jit(device=True, inline=True)
    def ray_aabb_intersect_gpu(orig, dir_inv, t_min_global, node_aabb_min, node_aabb_max):
        """Ray-AABB intersection test (Slab Test) for GPU."""
        t_near = -INF; t_far = INF
        eps_aabb = GEOMETRY_DTYPE(1e-6)

        for k in range(3):
            inv_d = dir_inv[k]
            aabb_min_k = node_aabb_min[k]
            aabb_max_k = node_aabb_max[k]
            t1 = (aabb_min_k - orig[k]) * inv_d
            t2 = (aabb_max_k - orig[k]) * inv_d
            if t1 > t2: t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
            if t_near >= t_far or t_far < eps_aabb or t_near >= t_min_global:
                return False # Miss
        return True # Hit

    # --- BVH 光线追踪 CUDA 内核 ---
    @cuda.jit
    def raytrace_cuda_bvh_kernel(
        flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
        labels_reordered, colors_reordered,
        cam_o, cam_dir, right, up,
        scr_w, scr_h, W, H,
        rgb, depth, sem, pts
    ):
        """CUDA kernel for ray tracing with BVH traversal."""
        i, j = cuda.grid(2)
        if i >= H or j >= W: return

        # Constants & Local Arrays
        inf_gpu = GEOMETRY_DTYPE(1e20)
        eps_gpu = GEOMETRY_DTYPE(1e-6)
        d = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        dir_inv = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        node_aabb_min = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        node_aabb_max = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        num_tris_total_gpu = v0s_reordered.shape[0]

        # 1. Calculate ray direction
        u = (GEOMETRY_DTYPE(j) + 0.5) / W - 0.5
        v = (GEOMETRY_DTYPE(i) + 0.5) / H - 0.5
        d[0] = cam_dir[0] + u * scr_w * right[0] - v * scr_h * up[0]
        d[1] = cam_dir[1] + u * scr_w * right[1] - v * scr_h * up[1]
        d[2] = cam_dir[2] + u * scr_w * right[2] - v * scr_h * up[2]

        nrm_sq = d[0]**2 + d[1]**2 + d[2]**2
        if nrm_sq < eps_gpu**2:
             rgb[i, j, 0] = DEFAULT_SKY_COLOR[0]; rgb[i, j, 1] = DEFAULT_SKY_COLOR[1]; rgb[i, j, 2] = DEFAULT_SKY_COLOR[2]
             depth[i, j] = inf_gpu; sem[i, j] = 0
             return

        inv_nrm = GEOMETRY_DTYPE(1.0) / math.sqrt(nrm_sq)
        d[0] *= inv_nrm; d[1] *= inv_nrm; d[2] *= inv_nrm

        for k in range(3):
            if abs(d[k]) < eps_gpu: dir_inv[k] = math.copysign(inf_gpu, d[k])
            else: dir_inv[k] = GEOMETRY_DTYPE(1.0) / d[k]

        # 2. Initialize traversal
        tmin = inf_gpu
        hit_prim_idx = -1
        num_nodes = flat_nodes.shape[0]
        if num_nodes == 0:
            rgb[i, j, 0] = DEFAULT_SKY_COLOR[0]; rgb[i, j, 1] = DEFAULT_SKY_COLOR[1]; rgb[i, j, 2] = DEFAULT_SKY_COLOR[2]
            depth[i, j] = inf_gpu; sem[i, j] = 0
            return

        BVH_GPU_STACK_SIZE = 64
        node_stack = cuda.local.array(BVH_GPU_STACK_SIZE, dtype=INDEX_DTYPE)
        stack_ptr = 0
        node_stack[stack_ptr] = 0
        stack_ptr += 1

        # 3. BVH Traversal Loop
        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = node_stack[stack_ptr]

            if node_idx < 0 or node_idx >= num_nodes: continue

            node_aabb_min[0] = flat_nodes[node_idx, 0]; node_aabb_min[1] = flat_nodes[node_idx, 1]; node_aabb_min[2] = flat_nodes[node_idx, 2]
            node_aabb_max[0] = flat_nodes[node_idx, 3]; node_aabb_max[1] = flat_nodes[node_idx, 4]; node_aabb_max[2] = flat_nodes[node_idx, 5]

            if not ray_aabb_intersect_gpu(cam_o, dir_inv, tmin, node_aabb_min, node_aabb_max):
                continue

            # *** MODIFICATION: Check sign of node[7] using .view() ***
            info_bits_f = flat_nodes[node_idx, 7]
            info_val = info_bits_f.view(INDEX_DTYPE) # Cast bits to int

            if info_val < 0: # Leaf Node (negative count stored)
                prim_count = -info_val # Get positive count
                prim_offset_bits_f = flat_nodes[node_idx, 6]
                prim_offset = prim_offset_bits_f.view(INDEX_DTYPE) # Cast bits

                for p_idx_offset in range(prim_count):
                    current_prim_idx = prim_offset + p_idx_offset
                    if current_prim_idx < num_tris_total_gpu:
                        tri_v0 = v0s_reordered[current_prim_idx]
                        tri_e1 = e1s_reordered[current_prim_idx]
                        tri_e2 = e2s_reordered[current_prim_idx]
                        t = ray_tri_intersect_gpu(cam_o, d, tri_v0, tri_e1, tri_e2)
                        if t < tmin:
                            tmin = t
                            hit_prim_idx = current_prim_idx

            else: # Internal Node (info_val >= 0 is right_child_idx)
                left_child_idx_bits_f = flat_nodes[node_idx, 6]
                left_child_idx = left_child_idx_bits_f.view(INDEX_DTYPE) # Cast bits
                right_child_idx = info_val # Already have the right child index

                if stack_ptr + 2 <= BVH_GPU_STACK_SIZE:
                    if left_child_idx >= 0 and left_child_idx < num_nodes:
                        node_stack[stack_ptr] = left_child_idx
                        stack_ptr += 1
                    if right_child_idx >= 0 and right_child_idx < num_nodes:
                        node_stack[stack_ptr] = right_child_idx
                        stack_ptr += 1
                # else: # Handle stack overflow

        # 4. Process hit result
        if hit_prim_idx >= 0:
            depth[i, j] = tmin
            sem[i, j] = labels_reordered[hit_prim_idx]
            color = colors_reordered[hit_prim_idx]
            rgb[i, j, 0] = color[0]; rgb[i, j, 1] = color[1]; rgb[i, j, 2] = color[2]
            pts[i, j, 0] = cam_o[0] + d[0] * tmin
            pts[i, j, 1] = cam_o[1] + d[1] * tmin
            pts[i, j, 2] = cam_o[2] + d[2] * tmin
        else:
            # No hit
            rgb[i, j, 0] = DEFAULT_SKY_COLOR[0]; rgb[i, j, 1] = DEFAULT_SKY_COLOR[1]; rgb[i, j, 2] = DEFAULT_SKY_COLOR[2]
            depth[i, j] = inf_gpu
            sem[i, j] = 0
            # pts remain NaN

# --- OBJ 导出 ---
# (保持不变)
def save_combined_obj(filename, v0s, e1s, e2s,
                      cam_o, cam_dir, right, up, screen_w, screen_h,
                      pts, far):
    """Saves scene triangles, camera frustum, and hit points to an OBJ file."""
    print(f"    准备保存 OBJ 文件: {filename}")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'# Raytracer output: {datetime.now()}\n')
            f.write(f'# Generated with {len(v0s)} triangles.\n')

            # --- Scene Triangles ---
            f.write('\n# Scene Geometry (Triangles)\n')
            f.write('o scene_geometry\n')
            vidx = 1 # OBJ indices start from 1
            for i in range(v0s.shape[0]):
                v0 = v0s[i]; v1 = v0 + e1s[i]; v2 = v0 + e2s[i]
                f.write(f"v {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
                f.write(f"v {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"v {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                f.write(f"f {vidx}// {vidx+1}// {vidx+2}//\n")
                vidx += 3

            # --- Camera Frustum ---
            f.write('\n# Camera Frustum\n')
            f.write('o camera_frustum\n')
            f.write(f"v {cam_o[0]:.6f} {cam_o[1]:.6f} {cam_o[2]:.6f}\n")
            cam_v_start = vidx; vidx += 1
            corners = []
            for du in [-0.5, 0.5]:
                for dv in [-0.5, 0.5]:
                    d_corner = cam_dir + (du * screen_w * right) - (dv * screen_h * up)
                    d_corner /= np.linalg.norm(d_corner)
                    corner_pt = cam_o + d_corner * far
                    corners.append(corner_pt)
                    f.write(f"v {corner_pt[0]:.6f} {corner_pt[1]:.6f} {corner_pt[2]:.6f}\n")
                    vidx += 1
            bl, tl, br, tr = cam_v_start + 1, cam_v_start + 2, cam_v_start + 3, cam_v_start + 4
            f.write(f"l {cam_v_start} {bl}\n"); f.write(f"l {cam_v_start} {tl}\n")
            f.write(f"l {cam_v_start} {br}\n"); f.write(f"l {cam_v_start} {tr}\n")
            f.write(f"l {bl} {tl}\n"); f.write(f"l {tl} {tr}\n")
            f.write(f"l {tr} {br}\n"); f.write(f"l {br} {bl}\n")

            # --- Intersection Points ---
            f.write('\n# Intersection Points (Sampled)\n')
            f.write('o intersection_points\n')
            H_pts, W_pts = pts.shape[:2] # Use actual shape of pts array
            step = max(1, H_pts // 64, W_pts // 64)
            point_v_start = vidx
            num_pts_written = 0
            for i in range(0, H_pts, step):
                for j in range(0, W_pts, step):
                    p = pts[i, j]
                    if not np.isnan(p[0]):
                        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                        vidx += 1
                        num_pts_written += 1
            if num_pts_written > 0:
                 f.write(f"g hit_points\n")
                 f.write(f"p {' '.join(map(str, range(point_v_start, vidx)))}\n")

        print(f"    成功保存场景、视锥体和 {num_pts_written} 个采样点到 {filename}")
    except IOError as e:
        print(f"    错误: 无法写入 OBJ 文件 {filename}: {e}")
    except Exception as e:
        print(f"    保存 OBJ 时发生未知错误: {e}")


# --- 主流程 ---
def main():
    global _GPU_AVAILABLE
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

    # --- 场景生成 ---
    print('[1/7] 构建场景...')
    t0 = _now()
    v0s, e1s, e2s, labels, colors, prim_indices = build_scene()
    num_triangles = len(v0s)
    log_step('场景构建', t0)
    if num_triangles == 0:
        print("错误: 场景构建结果为 0 个三角形。正在退出。")
        return
    print(f"    场景包含 {num_triangles} 个三角形。")

    # --- 构建 BVH ---
    print('[2/7] 构建 BVH...')
    t0 = _now()
    flat_nodes, ordered_prim_indices = build_bvh(v0s, e1s, e2s, prim_indices)
    log_step('BVH 构建', t0)

    # --- Reorder Geometry Data Based on BVH ---
    print('[3/7] 根据 BVH 重新排序几何数据...')
    t0 = _now()
    if len(ordered_prim_indices) != num_triangles:
        print(f"错误: BVH 返回的索引数量 ({len(ordered_prim_indices)}) 与三角形数量 ({num_triangles}) 不匹配。")
        v0s_reordered = v0s; e1s_reordered = e1s; e2s_reordered = e2s
        labels_reordered = labels; colors_reordered = colors
        print("    警告: 使用原始几何数据顺序进行光线追踪。")
    else:
        try:
            v0s_reordered = v0s[ordered_prim_indices]
            e1s_reordered = e1s[ordered_prim_indices]
            e2s_reordered = e2s[ordered_prim_indices]
            labels_reordered = labels[ordered_prim_indices]
            colors_reordered = colors[ordered_prim_indices]
        except IndexError as e:
             print(f"错误: 使用 ordered_prim_indices 重新排序几何数据时发生索引错误: {e}")
             print("    警告: 使用原始几何数据顺序进行光线追踪。")
             v0s_reordered = v0s; e1s_reordered = e1s; e2s_reordered = e2s
             labels_reordered = labels; colors_reordered = colors

    log_step('几何数据重新排序', t0)

    # --- 相机设置 ---
    print('[4/7] 设置相机...')
    t0 = _now()
    # *** MODIFICATION: New Camera Position and Target ***
    # cam_o = np.array([-70, 40, -120], dtype=GEOMETRY_DTYPE) # Old position
    # cam_t = np.array([10, 5, 0], dtype=GEOMETRY_DTYPE)    # Old target
    cam_o = np.array([-50, 100, -180], dtype=GEOMETRY_DTYPE) # Higher, further back, less side angle
    cam_t = np.array([20, 10, 80], dtype=GEOMETRY_DTYPE)    # Target towards lake/mountains
    # *** END MODIFICATION ***

    cam_up_vec = np.array([0, 1, 0], dtype=GEOMETRY_DTYPE)

    cam_dir = cam_t - cam_o
    norm_cam_dir = np.linalg.norm(cam_dir)
    if norm_cam_dir < EPSILON: print("错误: 相机位置和目标点重合。"); return
    cam_dir /= norm_cam_dir

    right = np.cross(cam_dir, cam_up_vec)
    norm_right = np.linalg.norm(right)
    if norm_right < EPSILON:
        print("警告: 相机方向与向上向量平行。调整右向量。")
        if abs(cam_dir[1]) > 1.0 - EPSILON: right = np.cross(np.array([0, 0, 1.0], dtype=GEOMETRY_DTYPE), cam_dir)
        else: right = np.array([1, 0, 0], dtype=GEOMETRY_DTYPE)
        norm_right = np.linalg.norm(right)
    right /= norm_right
    up = np.cross(right, cam_dir)

    W, H = 1920, 1080
    fov_degrees = 65.0 # Slightly narrower FoV might be better for distant view
    fov_radians = np.deg2rad(fov_degrees)
    aspect_ratio = W / H
    screen_h = GEOMETRY_DTYPE(2.0 * np.tan(fov_radians / 2.0))
    screen_w = GEOMETRY_DTYPE(screen_h * aspect_ratio)

    log_step('相机设置', t0)
    print(f"    分辨率: {W}x{H}, FoV: {fov_degrees} deg")
    print(f"    相机位置: [{cam_o[0]:.2f}, {cam_o[1]:.2f}, {cam_o[2]:.2f}]")
    print(f"    相机目标: [{cam_t[0]:.2f}, {cam_t[1]:.2f}, {cam_t[2]:.2f}]")


    # --- 光线追踪 ---
    print('[5/7] 光线追踪...')
    t0_raytrace = _now()
    rgb, depth, sem_lbl, pts = None, None, None, None
    use_gpu = _GPU_AVAILABLE and flat_nodes.shape[0] > 0

    # --- Remove forced CPU for normal execution ---
    # print("---!! 强制使用 CPU 进行调试 !! ---")
    # use_gpu = False
    # ---

    if use_gpu:
        print("    尝试使用 GPU (CUDA + BVH)...")
        try:
            t_upload_start = _now()
            # --- Upload Data to GPU ---
            d_flat_nodes = cuda.to_device(flat_nodes)
            d_v0s    = cuda.to_device(v0s_reordered)
            d_e1s    = cuda.to_device(e1s_reordered)
            d_e2s    = cuda.to_device(e2s_reordered)
            d_labels = cuda.to_device(labels_reordered)
            d_colors = cuda.to_device(colors_reordered)
            d_cam_o   = cuda.to_device(cam_o)
            d_cam_dir = cuda.to_device(cam_dir)
            d_right   = cuda.to_device(right)
            d_up      = cuda.to_device(up)

            # Output buffers (initialize on GPU)
            d_rgb    = cuda.device_array((H, W, 3), dtype=COLOR_DTYPE)
            depth_init_host = np.full((H, W), INF, dtype=DEPTH_DTYPE)
            d_depth  = cuda.to_device(depth_init_host)
            sem_init_host = np.zeros((H, W), dtype=LABEL_DTYPE)
            d_sem    = cuda.to_device(sem_init_host)
            pts_init_host = np.full((H, W, 3), np.nan, dtype=POINT_DTYPE)
            d_pts    = cuda.to_device(pts_init_host)
            log_step('GPU 数据上传', t_upload_start)

            # --- Kernel Launch ---
            threads_per_block = (16, 16)
            blocks_per_grid_x = (W + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid_y = (H + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)
            print(f"    启动 CUDA 内核: Grid={blocks_per_grid}, Block={threads_per_block}")

            t_kernel_start = _now()
            raytrace_cuda_bvh_kernel[blocks_per_grid, threads_per_block](
                d_flat_nodes, d_v0s, d_e1s, d_e2s, d_labels, d_colors,
                d_cam_o, d_cam_dir, d_right, d_up,
                screen_w, screen_h, W, H,
                d_rgb, d_depth, d_sem, d_pts
            )
            cuda.synchronize()
            log_step('GPU BVH 内核执行', t_kernel_start)

            # --- Download Results ---
            t_download_start = _now()
            rgb     = d_rgb.copy_to_host()
            depth   = d_depth.copy_to_host()
            sem_lbl = d_sem.copy_to_host()
            pts     = d_pts.copy_to_host()
            log_step('GPU 数据下载', t_download_start)
            print("    GPU 执行成功。")

        except cuda.cudadrv.driver.CudaAPIError as e:
            print(f"\n---!! CUDA API 错误: {e} !!---")
            print("---!! 可能是显存不足或驱动问题。回退到 CPU 执行。 !!---\n")
            _GPU_AVAILABLE = False; use_gpu = False
        except AttributeError as e:
             if "'float' object has no attribute 'view'" in str(e) or \
                "'DeviceFunctionTemplate' object has no attribute 'view'" in str(e):
                 print(f"\n---!! GPU 错误: {e} !!---")
                 print("---!! Numba CUDA 的 .view() 用法可能存在问题或版本不兼容。回退到 CPU。 !!---\n")
             else:
                 print(f"\n---!! GPU 执行期间发生属性错误: {e} !!---")
                 print("---!! 回退到 CPU 执行。 !!---\n")
             _GPU_AVAILABLE = False; use_gpu = False
        except Exception as e:
            print(f"\n---!! GPU 执行期间发生未知错误: {e} !!---")
            print(f"---!! 错误类型: {type(e).__name__}")
            print("---!! 回退到 CPU 执行。 !!---\n")
            _GPU_AVAILABLE = False; use_gpu = False

    # --- CPU Execution ---
    if not use_gpu:
        print("    使用 CPU (Numba JIT + BVH)...")
        t_cpu_start = _now()
        # Ensure parallel execution is enabled for performance
        print("    注意: CPU 使用并行计算。")
        rgb, depth, sem_lbl, pts = raytrace_cpu_bvh(
            flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
            labels_reordered, colors_reordered,
            cam_o, cam_dir, right, up,
            screen_w, screen_h, W, H
        )
        log_step('CPU BVH 光线追踪执行', t_cpu_start)

    log_step('总光线追踪', t0_raytrace)

    # --- 图像导出 ---
    print('[6/7] 保存输出图像...')
    t0_save = _now()
    if rgb is not None:
        fn_view = os.path.join(outd, f'view_{ts}.png')
        try:
            Image.fromarray(rgb).save(fn_view)
            print(f"    已保存视图: {fn_view}")
        except Exception as e: print(f"    错误: 保存视图图像失败: {e}")
    else: print("    跳过视图保存 (无 RGB 数据).")

    if depth is not None:
        fn_depth = os.path.join(outd, f'depth_{ts}.png')
        try:
            valid_depth = depth[np.isfinite(depth)]
            if len(valid_depth) > 0:
                dmin = np.min(valid_depth)
                dmax_vis = np.percentile(valid_depth, 99.5)
                dmax_vis = min(dmax_vis, 600.0)
                print(f"    深度范围 (有限值): {dmin:.2f} 到 {np.max(valid_depth):.2f} (可视化上限: {dmax_vis:.2f})")
                if dmax_vis <= dmin: dmax_vis = dmin + 1.0

                scale = (dmax_vis - dmin)
                if scale < EPSILON: scale = EPSILON
                depth_normalized = (depth - dmin) / scale

                depth_clipped = np.clip(depth_normalized * 254, 0, 254)
                dmap = np.where(np.isfinite(depth), depth_clipped, 255).astype(np.uint8)

                Image.fromarray(dmap, 'L').save(fn_depth)
                print(f"    已保存深度图: {fn_depth}")
            else:
                print("    跳过深度图保存 (无有效深度值).")
                Image.new('L', (W, H), 0).save(fn_depth) # Save black image
        except RuntimeWarning as e:
             print(f"    保存深度图时发生运行时警告: {e}")
             print(f"    这通常发生在所有深度值都是 INF 时。")
             try: Image.new('L', (W, H), 0).save(fn_depth)
             except: pass
        except Exception as e: print(f"    错误: 保存深度图像失败: {e}")
    else: print("    跳过深度图保存 (无深度数据).")

    if sem_lbl is not None:
        fn_sem = os.path.join(outd, f'semantic_{ts}.png')
        try:
            palette = {label: color_array.tolist() for label, color_array in C.items()}
            palette[0] = PALETTE_BACKGROUND_COLOR
            sem_img = np.zeros((H, W, 3), dtype=COLOR_DTYPE)
            unique_labels = np.unique(sem_lbl)
            for label_id in unique_labels:
                 if label_id in palette:
                     mask = (sem_lbl == label_id)
                     sem_img[mask] = palette[label_id]
            Image.fromarray(sem_img).save(fn_sem)
            print(f"    已保存语义图: {fn_sem}")
        except Exception as e: print(f"    错误: 保存语义图像失败: {e}")
    else: print("    跳过语义图保存 (无语义数据).")

    log_step('图像导出', t0_save)

    # --- OBJ 导出 ---
    print('[7/7] 保存组合 OBJ 文件...')
    t0_obj = _now()
    fn_obj = os.path.join(outd, f'combined_{ts}.obj')
    far_dist = 350.0
    if depth is not None:
        valid_depth = depth[np.isfinite(depth)]
        if len(valid_depth) > 0: far_dist = max(far_dist, np.max(valid_depth) * 1.1)

    if pts is not None:
         save_combined_obj(fn_obj, v0s, e1s, e2s,
                           cam_o, cam_dir, right, up, screen_w, screen_h,
                           pts, far=far_dist)
    else: print("    跳过 OBJ 保存 (无交点数据).")
    log_step('OBJ 导出', t0_obj)

    # --- 完成 ---
    print('\n[完成]')
    total_time = _now() - total_t0
    print(f"总执行时间: {total_time:.2f}s")
    print(f"输出已保存至: {outd}")


if __name__ == '__main__':
    main()
