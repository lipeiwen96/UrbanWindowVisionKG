# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import math # 需要 math 模块
import random # 需要 random 模块 for choice

import numpy as np
from PIL import Image
from numba import njit, prange

# -- GPU Availability Check --------------------------------------------------
# (保持不变 - Keep unchanged from your original code)
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
# (保持不变 - Keep unchanged from your original code)
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
# (保持不变 - Keep unchanged from your original code)
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
# (保持不变 - Keep unchanged from your original code)
def _now() -> float:
    """高精度时间戳"""
    return time.perf_counter()

def log_step(title: str, t0: float) -> None:
    """阶段耗时打印"""
    print(f"    {title} 用时 {_now() - t0:.3f}s") # Increased precision

# --- 随机数生成器 (可复现) ---
# (保持不变 - Keep unchanged from your original code)
RAND = np.random.default_rng(seed=0)

# --- 几何体生成函数 ---
# (保持不变 - Keep unchanged from your original code)
def create_box(center, size):
    """创建长方体"""
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    verts = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ], dtype=GEOMETRY_DTYPE)
    faces = [
        (0, 1, 2), (0, 2, 3), # Bottom
        (4, 6, 5), (4, 7, 6), # Top
        (0, 4, 5), (0, 5, 1), # Front
        (3, 2, 6), (3, 6, 7), # Back
        (0, 3, 7), (0, 7, 4), # Left
        (1, 5, 6), (1, 6, 2), # Right
    ] # 12 triangles
    return verts, faces

def create_prism(center, height, radius, sides=5):
    """创建棱柱体"""
    cx, cy, cz = center
    half_h = height / 2
    verts = []
    # Create vertices for top and bottom faces
    for i in range(sides):
        angle = 2 * np.pi * i / sides
        x = cx + radius * np.cos(angle)
        z = cz + radius * np.sin(angle)
        verts.append((x, cy - half_h, z)) # Bottom vertex
        verts.append((x, cy + half_h, z)) # Top vertex
    verts = np.asarray(verts, dtype=GEOMETRY_DTYPE)

    faces = []
    # Create side faces (quads made of two triangles)
    for i in range(sides):
        i0_bottom = 2 * i
        i1_bottom = (2 * (i + 1)) % (2 * sides)
        i0_top = i0_bottom + 1
        i1_top = i1_bottom + 1

        # Triangle 1 of the side quad
        faces.append((i0_bottom, i1_bottom, i1_top))
        # Triangle 2 of the side quad
        faces.append((i0_bottom, i1_top, i0_top))

    # Optional: Add top and bottom cap faces if needed (increases poly count)
    # Requires a center vertex for each cap or triangulation logic
    # Example (simple fan triangulation for caps):
    # Add center points (could average vertices, but placing at cx, cz is simpler)
    # bottom_center_idx = len(verts)
    # verts = np.vstack([verts, np.array([[cx, cy - half_h, cz]], dtype=GEOMETRY_DTYPE)])
    # top_center_idx = len(verts)
    # verts = np.vstack([verts, np.array([[cx, cy + half_h, cz]], dtype=GEOMETRY_DTYPE)])
    # for i in range(sides):
    #     i0_bottom = 2 * i
    #     i1_bottom = (2 * (i + 1)) % (2 * sides)
    #     i0_top = i0_bottom + 1
    #     i1_top = i1_bottom + 1
    #     faces.append((i0_bottom, i1_bottom, bottom_center_idx)) # Bottom cap triangle
    #     faces.append((i0_top, top_center_idx, i1_top))      # Top cap triangle (note order for winding)


    return verts, faces # Returns 2 * sides triangles without caps

def create_skyscraper(center, base, height, levels=8, taper=0.85):
    """创建分层、逐渐变细的摩天楼"""
    verts, faces = [], []
    cx, cy, cz = center # Base y-coordinate is included in center
    current_y = cy      # Start building from the provided base y
    current_width = base
    segment_height = height / levels

    for level in range(levels):
        # Calculate center y for this segment
        segment_center_y = current_y + segment_height / 2
        segment_center = (cx, segment_center_y, cz)
        segment_size = (current_width, segment_height, current_width)

        # Create the box for this level
        v_seg, f_seg = create_box(segment_center, segment_size)

        # Add vertices and adjust face indices
        vertex_offset = len(verts)
        verts.extend(v_seg)
        faces.extend([(a + vertex_offset, b + vertex_offset, c + vertex_offset) for a, b, c in f_seg])

        # Update for next level
        current_y += segment_height
        current_width *= taper

    return np.asarray(verts, dtype=GEOMETRY_DTYPE), faces # 12 * levels triangles

def create_cylinder(center, height, radius, *, segments: int = 16):
    """创建圆柱体"""
    cx, cy, cz = center
    verts = []
    half_h = height / 2
    # Create vertices for top and bottom circles
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = cx + radius * np.cos(angle)
        z = cz + radius * np.sin(angle)
        verts.append((x, cy - half_h, z)) # Bottom vertex
        verts.append((x, cy + half_h, z)) # Top vertex
    verts = np.asarray(verts, dtype=GEOMETRY_DTYPE)

    faces = []
    # Create side faces (quads made of two triangles)
    for i in range(segments):
        i0_bottom = 2 * i
        i1_bottom = (i0_bottom + 2) % (2 * segments) # Next bottom vertex index
        i0_top = i0_bottom + 1
        i1_top = i1_bottom + 1                 # Next top vertex index

        # Triangle 1 of the side quad
        faces.append((i0_bottom, i1_bottom, i1_top))
        # Triangle 2 of the side quad
        faces.append((i0_bottom, i1_top, i0_top))

    # Optional: Add top and bottom cap faces (similar to prism)

    return verts, faces # 2 * segments triangles without caps

def create_cone_tree(base_center, height, radius, *, segments: int = 10):
    """创建简化的锥形树（圆柱树干+圆锥树冠）"""
    cx, cy, cz = base_center # Base y is included here
    trunk_height = height * 0.3
    trunk_radius = radius * 0.15 # Thinner trunk
    crown_height = height * 0.7
    crown_radius = radius

    # --- Trunk (Cylinder) ---
    trunk_center_y = cy + trunk_height / 2
    trunk_segments = max(6, segments // 2) # Fewer segments for trunk
    trunk_verts, trunk_faces = create_cylinder(
        center=(cx, trunk_center_y, cz),
        height=trunk_height, radius=trunk_radius, segments=trunk_segments)
    num_trunk_verts = len(trunk_verts)

    # --- Crown (Cone) ---
    crown_base_y = cy + trunk_height # Y level where the cone base starts
    apex_y = crown_base_y + crown_height
    apex_vertex = np.array([cx, apex_y, cz], dtype=GEOMETRY_DTYPE)

    # Create base vertices for the cone
    cone_base_verts = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = cx + crown_radius * np.cos(angle)
        z = cz + crown_radius * np.sin(angle)
        cone_base_verts.append((x, crown_base_y, z))
    cone_base_verts = np.asarray(cone_base_verts, dtype=GEOMETRY_DTYPE)

    # Combine apex and base vertices for the cone part
    cone_verts = np.vstack((apex_vertex.reshape(1, 3), cone_base_verts))
    num_cone_verts = len(cone_verts) # apex + base vertices

    # Create cone faces (sides connecting apex to base)
    cone_faces = []
    apex_idx_local = 0 # Apex is the first vertex in cone_verts
    for i in range(segments):
        base_idx_local_1 = i + 1 # Base vertices start at index 1 in cone_verts
        base_idx_local_2 = (i + 1) % segments + 1 # Next base vertex index (wraps around)
        cone_faces.append((apex_idx_local, base_idx_local_1, base_idx_local_2))

    # Optional: Create cone base face (connecting base vertices)
    # Requires triangulation, e.g., fan from the first base vertex
    # base_center_idx_local = 1 # Use first base vertex as center for fan
    # for i in range(1, segments - 1):
    #     idx1 = base_center_idx_local + i
    #     idx2 = base_center_idx_local + i + 1
    #     cone_faces.append((base_center_idx_local, idx2, idx1)) # Winding order matters


    # --- Combine Trunk and Cone ---
    # Combine vertex arrays
    all_verts = np.vstack((trunk_verts, cone_verts))

    # Adjust cone face indices by the number of trunk vertices and add them
    adjusted_cone_faces = []
    for face in cone_faces:
        adjusted_face = (face[0] + num_trunk_verts,
                         face[1] + num_trunk_verts,
                         face[2] + num_trunk_verts)
        adjusted_cone_faces.append(adjusted_face)

    all_faces = trunk_faces + adjusted_cone_faces

    # Total triangles: (2 * trunk_segments) + segments (+ segments-2 if base cap)
    return all_verts, all_faces

def create_mountain_range(x_start, x_end, z_pos, *, segs=200, depth=25, h_min=18, h_max=50):
    """创建基于噪声的山脉段"""
    xs = np.linspace(x_start, x_end, segs + 1, dtype=GEOMETRY_DTYPE)
    num_noise_points = max(5, segs // 10) # Control noise frequency
    noise_xs = np.linspace(x_start, x_end, num_noise_points)
    # Generate random noise values between -1 and 1
    noise_ys_raw = RAND.uniform(-1, 1, noise_xs.shape)
    # Interpolate noise values to match the number of segments
    noise_ys = np.interp(xs, noise_xs, noise_ys_raw) # Linear interpolation
    # Map interpolated noise to the desired height range
    base_heights = np.interp(noise_ys, (-1, 1), (h_min, h_max))

    verts = []
    faces = []
    zero_y = GEOMETRY_DTYPE(0) # Base of the mountain segment will be at y=0 before final adjustment
    z_pos_dtype = GEOMETRY_DTYPE(z_pos)
    depth_dtype = GEOMETRY_DTYPE(depth)

    # Create segments along the x-axis
    for i in range(segs):
        x0, x1 = xs[i], xs[i + 1]
        h0, h1 = base_heights[i], base_heights[i + 1]
        h0_dtype, h1_dtype = GEOMETRY_DTYPE(h0), GEOMETRY_DTYPE(h1)

        # Define 8 vertices for the prism-like segment
        # Order: bottom-front-left, bottom-front-right, top-front-left, top-front-right,
        #        bottom-back-left,  bottom-back-right,  top-back-left,  top-back-right
        v_seg = [
            (x0, zero_y,   z_pos_dtype),           # 0
            (x1, zero_y,   z_pos_dtype),           # 1
            (x0, h0_dtype, z_pos_dtype),           # 2
            (x1, h1_dtype, z_pos_dtype),           # 3
            (x0, zero_y,   z_pos_dtype + depth_dtype), # 4
            (x1, zero_y,   z_pos_dtype + depth_dtype), # 5
            (x0, h0_dtype, z_pos_dtype + depth_dtype), # 6
            (x1, h1_dtype, z_pos_dtype + depth_dtype)  # 7
        ]
        idx0 = len(verts) # Starting index for this segment's vertices
        verts.extend(v_seg)

        # Helper to create faces with offset indices
        f = lambda a, b, c: (idx0 + a, idx0 + b, idx0 + c)

        # Create faces for the mountain segment (10 triangles for a closed segment)
        faces.append(f(0, 1, 3)) # Front face triangle 1
        faces.append(f(0, 3, 2)) # Front face triangle 2
        faces.append(f(4, 6, 7)) # Back face triangle 1
        faces.append(f(4, 7, 5)) # Back face triangle 2
        faces.append(f(0, 4, 6)) # Left face triangle 1
        faces.append(f(0, 6, 2)) # Left face triangle 2
        faces.append(f(1, 3, 7)) # Right face triangle 1
        faces.append(f(1, 7, 5)) # Right face triangle 2
        faces.append(f(2, 6, 7)) # Top face triangle 1
        faces.append(f(2, 7, 3)) # Top face triangle 2
        # faces.append(f(0, 5, 4)) # Bottom face triangle 1 (optional)
        # faces.append(f(0, 1, 5)) # Bottom face triangle 2 (optional)

    return np.asarray(verts, dtype=GEOMETRY_DTYPE), faces # 10 * segs triangles


# --- 场景构建 (MODIFIED) ---
def build_scene():
    """构建整个场景几何体 - 修改版，增加规模和建筑集群"""
    tris_geom = []
    tris_labs = []
    tris_cols = []

    # --- 目标对象数量 (大幅增加以达到 >200k 面) ---
    # Box = 12 tris, Skyscraper ~100-150 tris, Prism ~20-30 tris, Tree ~40-60 tris, Mountain = 10*segs
    # Estimate: 5000*12 + 500*120 + 500*25 + 10000*45 + 2*(10*250) = 60k + 60k + 12.5k + 450k + 5k = ~587.5k
    target_num_buildings = 5000
    target_num_skyscrapers = 500
    target_num_prisms = 500
    target_num_trees = 10000
    mountain_segs1 = 250 # More detail
    mountain_segs2 = 200

    print(f"    目标对象数量: 建筑={target_num_buildings}, 摩天楼={target_num_skyscrapers}, "
          f"棱柱={target_num_prisms}, 树木={target_num_trees}")

    # Helper to add objects (保持不变)
    def add_object(verts, faces, label, color_map):
        """将对象的顶点和面添加到全局列表"""
        color = color_map[label]
        for face_indices in faces:
            # Basic check for vertex index validity
            if all(idx < len(verts) for idx in face_indices):
                if len(face_indices) == 3:
                    a, b, c = face_indices
                    # Append vertices directly if needed, or just indices if verts are global
                    # Here we append the actual vertex coordinates per triangle
                    tris_geom.append((verts[a], verts[b], verts[c]))
                    tris_labs.append(label)
                    tris_cols.append(color)
                # else: print(f"警告: 非三角形面索引 {face_indices}，已跳过。")
            # else: print(f"警告: 面索引 {face_indices} 超出顶点数组范围 {len(verts)}，已跳过。")

    # --- 扩大世界范围 ---
    world_radius = 400.0 # Significantly larger world
    ground_size = (world_radius * 2.2, 0.01, world_radius * 2.2) # Ground slightly larger
    min_coord = -world_radius
    max_coord = world_radius

    # --- 地面和基准高度 (保持不变) ---
    ground_y = -0.1
    base_y = ground_y + 0.01 # Objects sit slightly above ground

    # 地面 (扩大)
    print("    生成地面...")
    ground_center = (0, ground_y + 0.005, 0)
    g_v, g_f = create_box(ground_center, ground_size)
    add_object(g_v, g_f, 0, C) # Label 0: Grass

    # 湖泊 (移动并可能调整大小以适应新布局)
    print("    生成湖泊...")
    lake_y = base_y + 0.02
    lake_center = (150, lake_y, 200) # Move lake further out
    lake_size = (120, 0.01, 90)     # Make it larger
    lake_v, lake_f = create_box(lake_center, lake_size)
    # Only add the top faces of the lake box
    add_object(lake_v, [lake_f[i] for i in [2, 3]], 3, C) # Label 3: Water
    # Define lake bounds for placement checks
    lake_x_min = lake_center[0] - lake_size[0] / 2
    lake_x_max = lake_center[0] + lake_size[0] / 2
    lake_z_min = lake_center[2] - lake_size[2] / 2
    lake_z_max = lake_center[2] + lake_size[2] / 2

    # 道路网格 (扩大)
    print("    生成道路网格...")
    road_y = base_y + 0.03
    road_w = 5.0 # Slightly wider roads
    road_h = 0.01
    num_roads = 15 # More roads for larger area
    # Generate coordinates spanning the larger world radius
    grid_coords = np.linspace(min_coord * 0.9, max_coord * 0.9, num_roads, dtype=GEOMETRY_DTYPE)
    road_len = world_radius * 2.0 + road_w # Ensure roads cross the area

    for x in grid_coords:
        # Create vertical roads (along Z axis)
        road_center_z = 0
        v, f = create_box((x, road_y, road_center_z), (road_w, road_h, road_len))
        add_object(v, f, 2, C) # Label 2: Road
    for z in grid_coords:
        # Create horizontal roads (along X axis)
        road_center_x = 0
        v, f = create_box((road_center_x, road_y, z), (road_len, road_h, road_w))
        add_object(v, f, 2, C)

    # --- 定义建筑集群 ---
    # List of tuples: (center_x, center_z, radius)
    clusters = [
        (-200, -150, 80), # Cluster 1: Bottom-left-ish
        (  50, -250, 90), # Cluster 2: Bottom-center-ish
        ( 250,  -50, 70), # Cluster 3: Mid-right
        ( 100,  180, 100), # Cluster 4: Top-center-ish (nearer lake)
        (-180,  220, 60)  # Cluster 5: Top-left-ish
    ]
    num_clusters = len(clusters)
    print(f"    定义了 {num_clusters} 个建筑集群。")

    # Helper function to get random point in a cluster
    def get_random_point_in_cluster(cluster_info, bias_center=False):
        """获取集群内随机点, 可选偏向中心"""
        center_x, center_z, radius = cluster_info
        angle = RAND.uniform(0, 2 * np.pi)
        # Use sqrt for more uniform distribution within the circle area
        # Bias towards center if requested (e.g., for skyscrapers)
        max_dist_factor = 0.6 if bias_center else 1.0
        distance = np.sqrt(RAND.uniform(0, max_dist_factor)) * radius
        x = center_x + distance * np.cos(angle)
        z = center_z + distance * np.sin(angle)
        return x, z

    # --- 放置物体 (集群化) ---
    building_count = 0
    skyscraper_count = 0
    prism_count = 0
    tree_count = 0

    # 放置普通建筑
    print("    放置普通建筑...")
    attempts = 0
    max_attempts_building = target_num_buildings * 5 # Allow more attempts
    while building_count < target_num_buildings and attempts < max_attempts_building:
        attempts += 1
        # Randomly select a cluster for each building
        chosen_cluster_info = random.choice(clusters)
        x, z = get_random_point_in_cluster(chosen_cluster_info, bias_center=False)

        # 检查是否在湖内或太靠近道路
        is_in_lake = (lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max)
        if is_in_lake:
            continue

        is_on_road = False
        road_clearance = road_w # Buildings can be closer than trees
        for gx in grid_coords:
            if abs(x - gx) < road_clearance + RAND.uniform(1, 3): # Add random buffer
                is_on_road = True
                break
        if not is_on_road:
            for gz in grid_coords:
                if abs(z - gz) < road_clearance + RAND.uniform(1, 3):
                    is_on_road = True
                    break
        if is_on_road:
            continue

        # 放置建筑
        w = RAND.uniform(6, 15) # Slightly larger variation
        h = RAND.uniform(10, 25)
        center_y = base_y + h / 2
        v, f = create_box((x, center_y, z), (w, h, w))
        add_object(v, f, 1, C)
        building_count += 1

    if attempts >= max_attempts_building:
        print(f"    警告: 达到最大尝试次数，只放置了 {building_count}/{target_num_buildings} 个普通建筑。")


    # 放置摩天楼 (主要在集群中心附近)
    print("    放置摩天楼...")
    attempts = 0
    max_attempts_skyscraper = target_num_skyscrapers * 5
    while skyscraper_count < target_num_skyscrapers and attempts < max_attempts_skyscraper:
        attempts += 1
        chosen_cluster_info = random.choice(clusters)
        # Get point biased towards the cluster center
        x, z = get_random_point_in_cluster(chosen_cluster_info, bias_center=True)

        # 检查碰撞 (湖/路)
        is_in_lake = (lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max)
        if is_in_lake:
            continue
        is_on_road = False
        road_clearance = road_w * 1.2 # Slightly larger clearance
        for gx in grid_coords:
            if abs(x - gx) < road_clearance: is_on_road = True; break
        if not is_on_road:
            for gz in grid_coords:
                if abs(z - gz) < road_clearance: is_on_road = True; break
        if is_on_road:
            continue

        # 放置摩天楼
        base_w = RAND.uniform(8, 14)
        h = RAND.uniform(50, 90) # Taller skyscrapers
        levels = RAND.integers(8, 15)
        taper = RAND.uniform(0.85, 0.98)
        # Skyscraper base starts at base_y
        v, f = create_skyscraper((x, base_y, z), base_w, h, levels=levels, taper=taper)
        add_object(v, f, 4, C)
        skyscraper_count += 1

    if attempts >= max_attempts_skyscraper:
         print(f"    警告: 达到最大尝试次数，只放置了 {skyscraper_count}/{target_num_skyscrapers} 个摩天楼。")


    # 放置棱柱建筑 (分布在集群中)
    print("    放置棱柱建筑...")
    attempts = 0
    max_attempts_prism = target_num_prisms * 5
    while prism_count < target_num_prisms and attempts < max_attempts_prism:
        attempts += 1
        chosen_cluster_info = random.choice(clusters)
        x, z = get_random_point_in_cluster(chosen_cluster_info, bias_center=False)

        # 检查碰撞 (湖/路)
        is_in_lake = (lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max)
        if is_in_lake:
             continue
        is_on_road = False
        road_clearance = road_w * 1.1
        for gx in grid_coords:
            if abs(x - gx) < road_clearance: is_on_road = True; break
        if not is_on_road:
            for gz in grid_coords:
                if abs(z - gz) < road_clearance: is_on_road = True; break
        if is_on_road:
            continue

        # 放置棱柱
        r = RAND.uniform(5, 9)
        h = RAND.uniform(15, 35)
        sides = RAND.integers(5, 9) # More sides possible
        center_y = base_y + h / 2
        v, f = create_prism((x, center_y, z), h, r, sides=sides)
        add_object(v, f, 4, C) # Also label 4 (special building)
        prism_count += 1

    if attempts >= max_attempts_prism:
        print(f"    警告: 达到最大尝试次数，只放置了 {prism_count}/{target_num_prisms} 个棱柱建筑。")


    # 放置树木 (更广泛分布，但避开道路/湖泊/建筑密集区)
    print("    放置树木...")
    # Define areas to avoid placing trees (e.g., cluster centers)
    avoid_zones = clusters # Simple avoidance based on cluster radius
    tree_placement_radius = world_radius * 1.1 # Allow trees slightly outside main area
    attempts = 0
    max_attempts_tree = target_num_trees * 3 # Trees are denser, allow fewer retries per tree

    while tree_count < target_num_trees and attempts < max_attempts_tree:
        attempts += 1
        x = RAND.uniform(-tree_placement_radius, tree_placement_radius)
        z = RAND.uniform(-tree_placement_radius, tree_placement_radius)

        # 检查碰撞 (湖/路)
        is_in_lake = (lake_x_min < x < lake_x_max and lake_z_min < z < lake_z_max)
        if is_in_lake:
            continue

        is_on_road = False
        road_clearance_tree = road_w / 1.5 # Trees need less clearance than buildings maybe
        for gx in grid_coords:
             if abs(x - gx) < road_clearance_tree: is_on_road = True; break
        if not is_on_road:
            for gz in grid_coords:
                if abs(z - gz) < road_clearance_tree: is_on_road = True; break
        if is_on_road:
            continue

        # Check if tree is too close to a cluster center (optional, to thin out trees in city centers)
        # is_in_dense_zone = False
        # for cx, cz, cr in avoid_zones:
        #     if (x - cx)**2 + (z - cz)**2 < (cr * 0.5)**2: # Avoid inner 50% radius of clusters
        #         is_in_dense_zone = True
        #         break
        # if is_in_dense_zone:
        #     continue


        # 放置树木
        h = RAND.uniform(7, 14)
        r = RAND.uniform(1.8, 3.0)
        segs = RAND.integers(9, 15)
        # Tree base starts at base_y
        v, f = create_cone_tree((x, base_y, z), h, r, segments=segs)
        add_object(v, f, 5, C)
        tree_count += 1

    if attempts >= max_attempts_tree:
        print(f"    警告: 达到最大尝试次数，只放置了 {tree_count}/{target_num_trees} 个树木。")


    # 山脉 (放置在远景)
    print("    生成山脉...")
    mountain_z1 = max_coord + 80  # Push mountains further back relative to world radius
    mountain_z2 = mountain_z1 + 60
    mountain_width = world_radius * 1.5 # Wider mountains to span the view
    mountain_depth1 = 50
    mountain_depth2 = 40
    mountain_h_min1, mountain_h_max1 = 30, 85
    mountain_h_min2, mountain_h_max2 = 25, 60


    m_v1, m_f1 = create_mountain_range(
        x_start=-mountain_width, x_end=mountain_width, z_pos=mountain_z1,
        segs=mountain_segs1, depth=mountain_depth1,
        h_min=mountain_h_min1, h_max=mountain_h_max1)
    # Adjust height relative to base_y AFTER creation
    m_v1[:, 1] += base_y
    add_object(m_v1, m_f1, 6, C) # Label 6: Mountain

    m_v2, m_f2 = create_mountain_range(
        x_start=-mountain_width, x_end=mountain_width, z_pos=mountain_z2,
        segs=mountain_segs2, depth=mountain_depth2,
        h_min=mountain_h_min2, h_max=mountain_h_max2)
    # Adjust height relative to base_y AFTER creation
    m_v2[:, 1] += base_y
    add_object(m_v2, m_f2, 6, C)

    # --- 数据转换 (优化) ---
    N = len(tris_geom) # Total number of triangles generated
    if N == 0:
        print("警告: 场景为空!")
        # Return empty arrays with correct shapes and dtypes
        return (np.empty((0, 3), dtype=GEOMETRY_DTYPE), np.empty((0, 3), dtype=GEOMETRY_DTYPE),
                np.empty((0, 3), dtype=GEOMETRY_DTYPE), np.empty(0, dtype=LABEL_DTYPE),
                np.empty((0, 3), dtype=COLOR_DTYPE), np.empty(0, dtype=INDEX_DTYPE))

    print(f"    实际生成对象: 建筑={building_count}, 摩天楼={skyscraper_count}, "
          f"棱柱={prism_count}, 树木={tree_count}")
    print(f"    总计三角形数量: {N}") # Check if this meets the target

    # Pre-allocate NumPy arrays for efficiency
    v0s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    e1s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    e2s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)

    # Populate geometry arrays directly from tris_geom
    # tris_geom contains tuples of (v0, v1, v2) where each v is a coordinate tuple/list
    for i, (v0, v1, v2) in enumerate(tris_geom):
        v0_np = np.array(v0, dtype=GEOMETRY_DTYPE) # Ensure numpy array for subtraction
        v1_np = np.array(v1, dtype=GEOMETRY_DTYPE)
        v2_np = np.array(v2, dtype=GEOMETRY_DTYPE)
        v0s[i] = v0_np
        e1s[i] = np.subtract(v1_np, v0_np) # Numba prefers NumPy arrays
        e2s[i] = np.subtract(v2_np, v0_np)

    # Convert labels and colors lists to NumPy arrays
    labels = np.array(tris_labs, dtype=LABEL_DTYPE)
    colors = np.array(tris_cols, dtype=COLOR_DTYPE)

    # Primitive indices are just a range from 0 to N-1
    prim_indices = np.arange(N, dtype=INDEX_DTYPE)

    return v0s, e1s, e2s, labels, colors, prim_indices


# --- BVH 相关 ---
# (保持不变 - Keep unchanged from your original code)
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
        # Return valid AABB with min > max
        return (np.full(3, INF, dtype=BVH_NODE_DTYPE), np.full(3, -INF, dtype=BVH_NODE_DTYPE))

    # Initialize with the AABB of the first primitive
    first_idx = indices[0]
    # Ensure copy to avoid modifying the original AABB array
    global_min = tri_aabbs_min[first_idx].copy()
    global_max = tri_aabbs_max[first_idx].copy()

    # Iterate through the rest of the primitives
    for i in range(1, num_tris):
        idx = indices[i]
        current_min = tri_aabbs_min[idx]
        current_max = tri_aabbs_max[idx]
        # Expand the global AABB
        for k in range(3): # Iterate over x, y, z axes
            global_min[k] = min(global_min[k], current_min[k])
            global_max[k] = max(global_max[k], current_max[k])

    return global_min, global_max

# Helper function for int32 to float32 bit-casting (Numba CPU)
@njit
def int32_to_float32_bits(val_int32):
    """Reinterprets the bits of an int32 as a float32."""
    # Create a 1-element array of the int32 value
    int_array = np.array([val_int32], dtype=INDEX_DTYPE)
    # View the memory of the int32 array as a float32 array and return the single element
    return int_array.view(BVH_NODE_DTYPE)[0]

# Helper function for float32 to int32 bit-casting (Numba CPU)
@njit
def float32_to_int32_bits(val_float32):
    """Reinterprets the bits of a float32 as an int32."""
    # Create a 1-element array of the float32 value
    float_array = np.array([val_float32], dtype=BVH_NODE_DTYPE)
    # View the memory of the float32 array as an int32 array and return the single element
    return float_array.view(INDEX_DTYPE)[0]


# --- BVH Build (Recursive Part - Numba JITted) ---
@njit
def recursive_build_numba(
    current_node_idx,    # Index of the node being processed
    nodes_used,          # Mutable counter for total nodes used (passed as array)
    flat_nodes,          # The flat array storing all BVH nodes
    prim_indices,        # Array of primitive indices (reordered during build)
    start_idx,           # Start index in prim_indices for this node
    end_idx,             # End index (exclusive) in prim_indices for this node
    tri_aabbs_min,       # Precomputed min coords of all triangle AABBs
    tri_aabbs_max,       # Precomputed max coords of all triangle AABBs
    tri_centers          # Precomputed centers of all triangle AABBs
):
    """Recursive BVH build function, optimized with Numba."""
    num_prims = end_idx - start_idx
    # Get the current node's data from the flat array
    node = flat_nodes[current_node_idx]

    # 1. Calculate bounds for the primitives in this node
    indices_slice = prim_indices[start_idx:end_idx]
    aabb_min, aabb_max = calculate_bounds(indices_slice, tri_aabbs_min, tri_aabbs_max)
    # Store AABB in the node (first 6 floats)
    node[0:3] = aabb_min
    node[3:6] = aabb_max

    # 2. Leaf node check: If primitive count is below threshold, make it a leaf
    if num_prims <= BVH_MAX_LEAF_SIZE:
        # Store the starting index of primitives in prim_indices array
        # Use bit-casting to store the int32 index as a float32
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx))
        # Store the *negative* count of primitives to mark it as a leaf node
        # Use bit-casting to store the int32 count as a float32
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims))
        return # Stop recursion for this branch

    # 3. Internal node: Determine split axis and position
    # Calculate the extent (size) of the node's AABB along each axis
    extent = aabb_max - aabb_min
    # Choose the axis with the largest extent for splitting
    split_axis = np.argmax(extent)

    # Handle degenerate case: If the largest extent is near zero, make it a leaf
    # This prevents infinite recursion if all primitives are at the same point/plane
    if extent[split_axis] < EPSILON:
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx))
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims)) # Mark as leaf
        return

    # Calculate the split coordinate (midpoint of the chosen axis extent)
    split_coord = (aabb_min[split_axis] + aabb_max[split_axis]) * 0.5

    # 4. Partition primitives based on their centroid along the split axis
    # Use a simple partitioning scheme (similar to C++ std::partition)
    mid_point = start_idx
    for i in range(start_idx, end_idx):
        prim_idx = prim_indices[i]
        # If primitive center is less than split coordinate, swap it towards the start
        if tri_centers[prim_idx, split_axis] < split_coord:
            # Swap elements in the prim_indices array
            prim_indices[i], prim_indices[mid_point] = prim_indices[mid_point], prim_indices[i]
            mid_point += 1

    # Handle ineffective split: If all primitives fall on one side,
    # or if the split coordinate was degenerate, split in the middle.
    if mid_point == start_idx or mid_point == end_idx:
        mid_point = start_idx + num_prims // 2

    # 5. Allocate children nodes
    # Get the index for the left child (current value of nodes_used)
    left_child_idx = nodes_used[0]
    nodes_used[0] += 1 # Increment the counter
    # Get the index for the right child
    right_child_idx = nodes_used[0]
    nodes_used[0] += 1 # Increment the counter

    # Store children indices in the current node (node[6] and node[7])
    # For internal nodes, these are non-negative indices.
    node[6] = int32_to_float32_bits(INDEX_DTYPE(left_child_idx))
    node[7] = int32_to_float32_bits(INDEX_DTYPE(right_child_idx))

    # 6. Recursively build children
    # Build left child with primitives from start_idx to mid_point
    recursive_build_numba(left_child_idx, nodes_used, flat_nodes, prim_indices,
                          start_idx, mid_point,
                          tri_aabbs_min, tri_aabbs_max, tri_centers)
    # Build right child with primitives from mid_point to end_idx
    recursive_build_numba(right_child_idx, nodes_used, flat_nodes, prim_indices,
                          mid_point, end_idx,
                          tri_aabbs_min, tri_aabbs_max, tri_centers)


def build_bvh(v0s, e1s, e2s, prim_indices_in):
    """Builds the BVH using a top-down, middle-split approach."""
    N = len(prim_indices_in) # Number of primitives (triangles)
    if N == 0:
        # Return empty BVH if no primitives
        return np.empty((0, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE), np.empty(0, dtype=INDEX_DTYPE)

    t0_precompute = _now()
    # 1. Precompute AABBs and centroids for all primitives
    tri_aabbs_min = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_aabbs_max = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_centers = np.empty((N, 3), dtype=BVH_NODE_DTYPE)

    # Use Numba's parallel prange for faster precomputation
    @njit(parallel=True, fastmath=True)
    def precompute_bounds_centers_parallel(num_tris, v0s_n, e1s_n, e2s_n,
                                           aabbs_min_out, aabbs_max_out, centers_out):
        """Parallel computation of AABBs and centers."""
        for i in prange(num_tris): # Parallel loop
            v0, e1, e2 = v0s_n[i], e1s_n[i], e2s_n[i]
            # Calculate AABB for the i-th triangle
            aabb_min, aabb_max = calculate_tri_aabb_numba(v0, e1, e2)
            aabbs_min_out[i] = aabb_min
            aabbs_max_out[i] = aabb_max
            # Calculate centroid (center of AABB)
            centers_out[i] = (aabb_min + aabb_max) * 0.5

    # Run the parallel precomputation
    precompute_bounds_centers_parallel(N, v0s, e1s, e2s, tri_aabbs_min, tri_aabbs_max, tri_centers)
    log_step('BVH Precomputation', t0_precompute)

    # 2. Allocate node space
    # Maximum possible nodes in a binary tree with N leaves is 2N-1
    max_nodes = max(1, 2 * N - 1) # Ensure at least 1 node if N=1
    # Allocate the flat array for nodes, initialized to zeros
    flat_nodes = np.zeros((max_nodes, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE)

    # 3. Initialize recursive build
    # Create a copy of the input primitive indices, as it will be reordered
    ordered_prim_indices = np.copy(prim_indices_in)
    # Use a 1-element array to pass the node counter by reference to Numba function
    nodes_used = np.array([1], dtype=INDEX_DTYPE) # Start counter at 1 (node 0 is root)

    t0_recursive = _now()
    # Call the Numba-jitted recursive build function starting at the root (node 0)
    recursive_build_numba(
        0,                      # Start with root node index 0
        nodes_used,             # Pass the node counter array
        flat_nodes,             # Pass the node storage array
        ordered_prim_indices,   # Pass the reorderable primitive indices
        0,                      # Start index for primitives (0)
        N,                      # End index for primitives (N)
        tri_aabbs_min,          # Pass precomputed min AABBs
        tri_aabbs_max,          # Pass precomputed max AABBs
        tri_centers             # Pass precomputed centroids
    )
    log_step('BVH Recursive Build', t0_recursive)

    # 4. Trim unused node space
    actual_nodes_used = nodes_used[0] # Get the final count from the counter array
    flat_nodes = flat_nodes[:actual_nodes_used] # Slice the array to keep only used nodes

    print(f"    BVH 构建完成: {actual_nodes_used} 个节点.")
    # Optional: Print root node bounds for verification
    if actual_nodes_used > 0:
        root_min = flat_nodes[0, 0:3]
        root_max = flat_nodes[0, 3:6]
        print(f"    根节点 AABB Min: [{root_min[0]:.2f}, {root_min[1]:.2f}, {root_min[2]:.2f}], "
              f"Max: [{root_max[0]:.2f}, {root_max[1]:.2f}, {root_max[2]:.2f}]")
        # Debug root node type
        # root_info_bits = flat_nodes[0, 7]
        # root_info_int = float32_to_int32_bits(root_info_bits)
        # print(f"    根节点 Info (node[7] as int): {root_info_int}") # Should be non-negative if N > leaf_size

    # Return the flat BVH node array and the reordered primitive indices
    return flat_nodes, ordered_prim_indices


# --- 光线-三角形相交 (CPU) ---
# (保持不变 - Keep unchanged from your original code)
@njit(fastmath=True)
def intersect_ray_triangle_cpu(orig, dir, v0, e1, e2):
    """Moller-Trumbore ray-triangle intersection test (CPU version)."""
    # Calculate determinant
    h = np.cross(dir, e2)
    a = np.dot(e1, h)

    # Check if ray is parallel to the triangle plane (or nearly so)
    if abs(a) < EPSILON:
        return INF # Return infinity (no intersection)

    # Calculate inverse determinant
    f = GEOMETRY_DTYPE(1.0) / a

    # Calculate vector from ray origin to triangle vertex v0
    s = orig - v0

    # Calculate u barycentric coordinate
    u = f * np.dot(s, h)
    # Check if u is outside the valid range [0, 1]
    if u < 0.0 or u > 1.0:
        return INF

    # Calculate v barycentric coordinate
    q = np.cross(s, e1)
    v = f * np.dot(dir, q)
    # Check if v is outside the valid range [0, 1] and if u+v is within the triangle
    if v < 0.0 or u + v > 1.0:
        return INF

    # Calculate t, the distance along the ray to the intersection point
    t = f * np.dot(e2, q)

    # Return t only if the intersection is in front of the ray origin (t > epsilon)
    # Otherwise, return infinity (intersection behind origin is not counted)
    return t if t > EPSILON else INF

# --- 光线-AABB 相交 (CPU) ---
# (保持不变 - Keep unchanged from your original code)
@njit(fastmath=True)
def intersect_ray_aabb_cpu(orig, dir_inv, tmin_global, node_aabb_min, node_aabb_max):
    """Ray-AABB intersection test using Slab Test (CPU version)."""
    # Initialize near and far intersection distances
    t_near = -INF
    t_far = INF

    # Iterate over the three axes (x, y, z)
    for k in range(3):
        # Get precomputed inverse direction component for this axis
        inv_d = dir_inv[k]
        # Get AABB min and max for this axis
        aabb_min_k = node_aabb_min[k]
        aabb_max_k = node_aabb_max[k]

        # Calculate intersection distances with the two slab planes for this axis
        t1 = (aabb_min_k - orig[k]) * inv_d
        t2 = (aabb_max_k - orig[k]) * inv_d

        # Ensure t1 is the intersection with the near plane and t2 with the far plane
        if t1 > t2:
            t1, t2 = t2, t1 # Swap if necessary

        # Update the overall near and far intersection distances
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

        # Early exit conditions:
        # 1. If near intersection is further than far intersection (missed the box)
        # 2. If the intersection interval is entirely behind the ray origin (t_far < epsilon)
        # 3. If the near intersection is already further than the closest hit found so far (t_near >= tmin_global)
        if t_near >= t_far or t_far < EPSILON or t_near >= tmin_global:
            return False # Miss

    # If the loop completes without exiting, the ray intersects the AABB within the relevant range
    return True # Hit


# --- 光线追踪 (CPU - BVH) ---
# (保持不变 - Keep unchanged from your original code)
@njit(parallel=True, fastmath=True) # Enable parallel execution for performance
# @njit(fastmath=True) # Use non-parallel version for easier debugging if needed
def raytrace_cpu_bvh(
    flat_nodes,          # Flat array of BVH nodes
    v0s_reordered,       # Triangle vertex data (reordered by BVH build)
    e1s_reordered,       # Triangle edge1 data (reordered)
    e2s_reordered,       # Triangle edge2 data (reordered)
    labels_reordered,    # Primitive labels (reordered)
    colors_reordered,    # Primitive colors (reordered)
    cam_o,               # Camera origin
    cam_dir,             # Camera view direction (normalized)
    right,               # Camera right vector (normalized)
    up,                  # Camera up vector (normalized)
    screen_w,            # Width of the virtual screen plane
    screen_h,            # Height of the virtual screen plane
    W,                   # Image width in pixels
    H                    # Image height in pixels
):
    """Performs ray tracing on the CPU using the BVH."""
    # Initialize output arrays
    rgb = np.zeros((H, W, 3), dtype=COLOR_DTYPE)         # Color image
    depth = np.full((H, W), INF, dtype=DEPTH_DTYPE)      # Depth map
    sem = np.zeros((H, W), dtype=LABEL_DTYPE)            # Semantic map
    pts = np.full((H, W, 3), np.nan, dtype=POINT_DTYPE) # Intersection points

    # Precompute inverse image dimensions for normalization
    inv_W = GEOMETRY_DTYPE(1.0 / W)
    inv_H = GEOMETRY_DTYPE(1.0 / H)
    num_nodes = flat_nodes.shape[0]
    num_tris_total = len(v0s_reordered) # Total number of triangles

    # Handle empty BVH case
    if num_nodes == 0:
        # Fill image with sky color if scene is empty
        for i in prange(H): # Use prange for parallel loop
             for j in range(W):
                 rgb[i, j] = DEFAULT_SKY_COLOR
        return rgb, depth, sem, pts

    # BVH traversal stack size (per thread) - adjust if needed based on BVH depth
    BVH_CPU_STACK_SIZE = 64

    # Parallel loop over image rows (Numba handles thread distribution)
    for i in prange(H):
        # --- Per-thread local variables ---
        # Allocate stack for BVH traversal for this thread
        node_stack = np.empty(BVH_CPU_STACK_SIZE, dtype=INDEX_DTYPE)
        # Allocate temporary arrays for ray direction and its inverse
        d = np.empty(3, dtype=GEOMETRY_DTYPE)       # Ray direction
        dir_inv = np.empty(3, dtype=GEOMETRY_DTYPE) # Inverse ray direction

        # Loop over image columns
        for j in range(W):
            # 1. Calculate ray direction for this pixel
            # Map pixel coordinates (i, j) to screen plane coordinates (u, v)
            # Add 0.5 to sample pixel centers
            u_norm = (GEOMETRY_DTYPE(j) + 0.5) * inv_W - 0.5 # Range [-0.5, 0.5]
            v_norm = (GEOMETRY_DTYPE(i) + 0.5) * inv_H - 0.5 # Range [-0.5, 0.5]

            # Calculate direction vector using camera frame and screen dimensions
            d[0] = cam_dir[0] + u_norm * screen_w * right[0] - v_norm * screen_h * up[0]
            d[1] = cam_dir[1] + u_norm * screen_w * right[1] - v_norm * screen_h * up[1]
            d[2] = cam_dir[2] + u_norm * screen_w * right[2] - v_norm * screen_h * up[2]

            # Normalize the direction vector
            norm_sq = d[0]**2 + d[1]**2 + d[2]**2
            # Avoid division by zero if direction is zero vector
            if norm_sq < EPSILON**2:
                # Handle degenerate ray (e.g., set pixel to sky color and continue)
                rgb[i, j] = DEFAULT_SKY_COLOR
                continue # Skip to next pixel
            inv_norm = GEOMETRY_DTYPE(1.0) / math.sqrt(norm_sq)
            d *= inv_norm # d is now normalized

            # Precompute inverse direction components for AABB intersection
            for k in range(3):
                if abs(d[k]) < EPSILON:
                    # Handle division by zero for axis-aligned rays
                    dir_inv[k] = math.copysign(INF, d[k]) # Assign signed infinity
                else:
                    dir_inv[k] = GEOMETRY_DTYPE(1.0) / d[k]

            # 2. Initialize traversal state for this ray
            tmin = INF           # Minimum intersection distance found so far
            hit_prim_idx = -1    # Index of the hit primitive (-1 if no hit)
            stack_ptr = 0        # Pointer for the node stack
            node_stack[stack_ptr] = 0 # Start traversal at the root node (index 0)
            stack_ptr += 1

            # 3. BVH Traversal Loop
            while stack_ptr > 0:
                # Pop a node index from the stack
                stack_ptr -= 1
                node_idx = node_stack[stack_ptr]

                # Basic validity check (should not happen with correct build/traversal)
                if node_idx < 0 or node_idx >= num_nodes:
                    continue

                # Get the node data
                node = flat_nodes[node_idx]
                node_aabb_min = node[0:3]
                node_aabb_max = node[3:6]

                # Check if the ray intersects the node's AABB
                # Pass current tmin to prune branches further than the closest hit
                aabb_hit = intersect_ray_aabb_cpu(cam_o, dir_inv, tmin, node_aabb_min, node_aabb_max)

                if not aabb_hit:
                    continue # Skip this node and its children if AABB is missed

                # --- Node Type Check ---
                # Get the info stored in node[7] (bit-cast to int)
                info_bits = node[7]
                info_val = float32_to_int32_bits(info_bits)

                if info_val < 0: # Leaf Node (identified by negative primitive count)
                    prim_count = -info_val # Get the positive count of primitives
                    # Get the starting offset of primitives in the reordered array
                    prim_offset_bits = node[6]
                    prim_offset = float32_to_int32_bits(prim_offset_bits)

                    # Intersect ray with all primitives in this leaf node
                    for p_local_idx in range(prim_count):
                        p_global_idx = prim_offset + p_local_idx
                        # Check bounds just in case (should be unnecessary if BVH is correct)
                        if p_global_idx < num_tris_total:
                            # Perform ray-triangle intersection test
                            t = intersect_ray_triangle_cpu(cam_o, d,
                                                           v0s_reordered[p_global_idx],
                                                           e1s_reordered[p_global_idx],
                                                           e2s_reordered[p_global_idx])
                            # If this intersection is closer than the current minimum, update
                            if t < tmin:
                                tmin = t
                                hit_prim_idx = p_global_idx # Store index of the hit primitive

                else: # Internal Node (info_val >= 0 is the right child index)
                    # Get the left child index (stored in node[6])
                    left_child_idx_bits = node[6]
                    left_child_idx = float32_to_int32_bits(left_child_idx_bits)
                    # Right child index is already available in info_val
                    right_child_idx = info_val

                    # Push children onto the stack (order can matter for performance, but simple push is fine)
                    # Check for stack overflow before pushing
                    if stack_ptr + 2 <= BVH_CPU_STACK_SIZE:
                        # Push right child first, then left (so left is processed first - LIFO)
                        # Check validity of child indices before pushing
                        if right_child_idx >= 0 and right_child_idx < num_nodes:
                            node_stack[stack_ptr] = right_child_idx
                            stack_ptr += 1
                        if left_child_idx >= 0 and left_child_idx < num_nodes:
                            node_stack[stack_ptr] = left_child_idx
                            stack_ptr += 1
                    # else: # Handle stack overflow (e.g., print warning or stop)
                    #     print("BVH stack overflow!") # Should not happen with reasonable stack size

            # 4. Process hit result for the pixel
            if hit_prim_idx >= 0: # If a primitive was hit
                depth[i, j] = tmin # Store intersection distance
                sem[i, j] = labels_reordered[hit_prim_idx] # Store label of hit primitive
                rgb[i, j] = colors_reordered[hit_prim_idx] # Store color of hit primitive
                # Calculate intersection point
                hit_point = cam_o + tmin * d
                pts[i, j, 0] = hit_point[0]
                pts[i, j, 1] = hit_point[1]
                pts[i, j, 2] = hit_point[2]
            else: # No hit (ray went to infinity or hit nothing)
                rgb[i, j] = DEFAULT_SKY_COLOR # Set pixel to sky color
                # depth remains INF, sem remains 0, pts remains NaN

    return rgb, depth, sem, pts


# --- GPU 内核 ---
# (保持不变 - Keep unchanged from your original code, including the _GPU_AVAILABLE check)
if _GPU_AVAILABLE:

    # --- 光线-三角形相交 (GPU 设备函数) ---
    @cuda.jit(device=True, inline=True)
    def ray_tri_intersect_gpu(orig, dir, v0, e1, e2):
        """Moller-Trumbore ray-triangle intersection test for GPU."""
        # Define constants within the device function scope
        eps_gpu = GEOMETRY_DTYPE(1e-6)
        inf_gpu = GEOMETRY_DTYPE(1e20) # Use a large number for infinity on GPU

        # Calculate determinant part 1: h = cross(dir, e2)
        h0 = dir[1] * e2[2] - dir[2] * e2[1]
        h1 = dir[2] * e2[0] - dir[0] * e2[2]
        h2 = dir[0] * e2[1] - dir[1] * e2[0]
        # Calculate determinant part 2: a = dot(e1, h)
        a = e1[0] * h0 + e1[1] * h1 + e1[2] * h2

        # Check for parallel ray or backface culling (if a < eps)
        if abs(a) < eps_gpu:
             return inf_gpu
        # Calculate inverse determinant
        f = GEOMETRY_DTYPE(1.0) / a
        # Calculate vector s = orig - v0
        s0 = orig[0] - v0[0]
        s1 = orig[1] - v0[1]
        s2 = orig[2] - v0[2]
        # Calculate u coordinate: u = f * dot(s, h)
        u = f * (s0 * h0 + s1 * h1 + s2 * h2)

        # Check u bounds
        if u < 0.0 or u > 1.0:
            return inf_gpu
        # Calculate vector q = cross(s, e1)
        q0 = s1 * e1[2] - s2 * e1[1]
        q1 = s2 * e1[0] - s0 * e1[2]
        q2 = s0 * e1[1] - s1 * e1[0]
        # Calculate v coordinate: v = f * dot(dir, q)
        v = f * (dir[0] * q0 + dir[1] * q1 + dir[2] * q2)

        # Check v bounds and u+v bounds
        if v < 0.0 or u + v > 1.0:
            return inf_gpu
        # Calculate t: t = f * dot(e2, q)
        t = f * (e2[0] * q0 + e2[1] * q1 + e2[2] * q2)

        # Return t only if it's a valid forward intersection
        return t if t > eps_gpu else inf_gpu


    # --- 光线-AABB 相交 (GPU 设备函数) ---
    @cuda.jit(device=True, inline=True)
    def ray_aabb_intersect_gpu(orig, dir_inv, t_min_global, node_aabb_min, node_aabb_max):
        """Ray-AABB intersection test (Slab Test) for GPU."""
        t_near = -INF # Initialize near intersection distance
        t_far = INF  # Initialize far intersection distance
        eps_aabb = GEOMETRY_DTYPE(1e-6) # Epsilon for AABB intersection

        # Loop over x, y, z axes
        for k in range(3):
            inv_d = dir_inv[k]      # Inverse ray direction component
            aabb_min_k = node_aabb_min[k] # AABB min for this axis
            aabb_max_k = node_aabb_max[k] # AABB max for this axis

            # Calculate intersection distances with slab planes
            t1 = (aabb_min_k - orig[k]) * inv_d
            t2 = (aabb_max_k - orig[k]) * inv_d

            # Ensure t1 is the near intersection, t2 is the far
            if t1 > t2:
                t1, t2 = t2, t1

            # Update overall near and far distances
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

            # Early exit conditions (same as CPU version)
            if t_near >= t_far or t_far < eps_aabb or t_near >= t_min_global:
                return False # Miss

        return True # Hit

    # --- BVH 光线追踪 CUDA 内核 ---
    @cuda.jit
    def raytrace_cuda_bvh_kernel(
        flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
        labels_reordered, colors_reordered,
        cam_o, cam_dir, right, up,
        scr_w, scr_h, W, H, # Screen params and image dimensions
        rgb, depth, sem, pts # Output arrays (device pointers)
    ):
        """CUDA kernel for ray tracing with BVH traversal."""
        # Get thread indices for the 2D grid
        i, j = cuda.grid(2) # i = row (y), j = column (x)

        # Check if the thread indices are within the image bounds
        if i >= H or j >= W:
            return # Thread is outside the image, do nothing

        # Constants & Local Arrays (allocated in thread-local memory)
        inf_gpu = GEOMETRY_DTYPE(1e20)
        eps_gpu = GEOMETRY_DTYPE(1e-6)
        # Local arrays for calculations within the thread
        d = cuda.local.array(3, dtype=GEOMETRY_DTYPE)       # Ray direction
        dir_inv = cuda.local.array(3, dtype=GEOMETRY_DTYPE) # Inverse ray direction
        node_aabb_min = cuda.local.array(3, dtype=BVH_NODE_DTYPE) # Temp storage for node AABB min
        node_aabb_max = cuda.local.array(3, dtype=BVH_NODE_DTYPE) # Temp storage for node AABB max
        # Get total triangle count (needed for bounds checking in leaf nodes)
        num_tris_total_gpu = v0s_reordered.shape[0]

        # 1. Calculate ray direction (same logic as CPU version)
        u_norm = (GEOMETRY_DTYPE(j) + 0.5) / W - 0.5
        v_norm = (GEOMETRY_DTYPE(i) + 0.5) / H - 0.5
        d[0] = cam_dir[0] + u_norm * scr_w * right[0] - v_norm * scr_h * up[0]
        d[1] = cam_dir[1] + u_norm * scr_w * right[1] - v_norm * scr_h * up[1]
        d[2] = cam_dir[2] + u_norm * scr_w * right[2] - v_norm * scr_h * up[2]

        # Normalize direction
        nrm_sq = d[0]**2 + d[1]**2 + d[2]**2
        if nrm_sq < eps_gpu**2:
             # Handle degenerate ray: Set pixel to sky color and exit thread
             rgb[i, j, 0] = DEFAULT_SKY_COLOR[0]
             rgb[i, j, 1] = DEFAULT_SKY_COLOR[1]
             rgb[i, j, 2] = DEFAULT_SKY_COLOR[2]
             depth[i, j] = inf_gpu
             sem[i, j] = 0
             # pts remains NaN (implicitly)
             return

        inv_nrm = GEOMETRY_DTYPE(1.0) / math.sqrt(nrm_sq) # Use math.sqrt on GPU
        d[0] *= inv_nrm
        d[1] *= inv_nrm
        d[2] *= inv_nrm

        # Calculate inverse direction
        for k in range(3):
            if abs(d[k]) < eps_gpu:
                 dir_inv[k] = math.copysign(inf_gpu, d[k]) # Use math.copysign on GPU
            else:
                 dir_inv[k] = GEOMETRY_DTYPE(1.0) / d[k]

        # 2. Initialize traversal state
        tmin = inf_gpu
        hit_prim_idx = -1
        num_nodes = flat_nodes.shape[0]

        # Handle empty BVH
        if num_nodes == 0:
            rgb[i, j, 0] = DEFAULT_SKY_COLOR[0]
            rgb[i, j, 1] = DEFAULT_SKY_COLOR[1]
            rgb[i, j, 2] = DEFAULT_SKY_COLOR[2]
            depth[i, j] = inf_gpu
            sem[i, j] = 0
            return

        # Allocate local stack for BVH traversal
        BVH_GPU_STACK_SIZE = 64 # Must match CPU stack size if logic is identical
        node_stack = cuda.local.array(BVH_GPU_STACK_SIZE, dtype=INDEX_DTYPE)
        stack_ptr = 0
        node_stack[stack_ptr] = 0 # Start at root node
        stack_ptr += 1

        # 3. BVH Traversal Loop
        while stack_ptr > 0:
            # Pop node index
            stack_ptr -= 1
            node_idx = node_stack[stack_ptr]

            # Basic validity check
            if node_idx < 0 or node_idx >= num_nodes:
                continue

            # Fetch node AABB data into local arrays
            # Accessing global memory (flat_nodes) here
            node_aabb_min[0] = flat_nodes[node_idx, 0]
            node_aabb_min[1] = flat_nodes[node_idx, 1]
            node_aabb_min[2] = flat_nodes[node_idx, 2]
            node_aabb_max[0] = flat_nodes[node_idx, 3]
            node_aabb_max[1] = flat_nodes[node_idx, 4]
            node_aabb_max[2] = flat_nodes[node_idx, 5]

            # Check AABB intersection
            if not ray_aabb_intersect_gpu(cam_o, dir_inv, tmin, node_aabb_min, node_aabb_max):
                continue

            # --- Node Type Check using bit casting ---
            # Access the float value storing the info
            info_bits_f = flat_nodes[node_idx, 7]
            # Reinterpret the bits of the float as an int32
            # Numba CUDA requires using .view() on the float value itself
            info_val = info_bits_f.view(INDEX_DTYPE)

            if info_val < 0: # Leaf Node (negative count stored)
                prim_count = -info_val # Get positive count
                # Get the primitive offset (stored in node[6])
                prim_offset_bits_f = flat_nodes[node_idx, 6]
                prim_offset = prim_offset_bits_f.view(INDEX_DTYPE) # Cast bits

                # Intersect with primitives in the leaf
                for p_idx_offset in range(prim_count):
                    current_prim_idx = prim_offset + p_idx_offset
                    # Bounds check against total number of triangles
                    if current_prim_idx < num_tris_total_gpu:
                        # Fetch triangle data (accessing global memory)
                        # It's often faster to load into local arrays if reused, but direct access here
                        tri_v0 = v0s_reordered[current_prim_idx]
                        tri_e1 = e1s_reordered[current_prim_idx]
                        tri_e2 = e2s_reordered[current_prim_idx]
                        # Perform intersection test
                        t = ray_tri_intersect_gpu(cam_o, d, tri_v0, tri_e1, tri_e2)
                        # Update closest hit if this one is nearer
                        if t < tmin:
                            tmin = t
                            hit_prim_idx = current_prim_idx

            else: # Internal Node (info_val >= 0 is right_child_idx)
                # Get left child index (stored in node[6])
                left_child_idx_bits_f = flat_nodes[node_idx, 6]
                left_child_idx = left_child_idx_bits_f.view(INDEX_DTYPE) # Cast bits
                # Right child index is already info_val
                right_child_idx = info_val

                # Push children onto stack (check stack bounds first)
                if stack_ptr + 2 <= BVH_GPU_STACK_SIZE:
                    # Push right child then left child (LIFO order)
                    if right_child_idx >= 0 and right_child_idx < num_nodes:
                        node_stack[stack_ptr] = right_child_idx
                        stack_ptr += 1
                    if left_child_idx >= 0 and left_child_idx < num_nodes:
                        node_stack[stack_ptr] = left_child_idx
                        stack_ptr += 1
                # else: # Handle stack overflow (optional: could signal an error)
                #     pass

        # 4. Process hit result for the pixel (write to global memory output arrays)
        if hit_prim_idx >= 0: # Hit occurred
            depth[i, j] = tmin
            sem[i, j] = labels_reordered[hit_prim_idx]
            # Fetch color data for the hit primitive
            color = colors_reordered[hit_prim_idx]
            # Write color components to the output RGB array
            rgb[i, j, 0] = color[0]
            rgb[i, j, 1] = color[1]
            rgb[i, j, 2] = color[2]
            # Calculate and write intersection point
            pts[i, j, 0] = cam_o[0] + d[0] * tmin
            pts[i, j, 1] = cam_o[1] + d[1] * tmin
            pts[i, j, 2] = cam_o[2] + d[2] * tmin
        else: # No hit
            # Write sky color
            rgb[i, j, 0] = DEFAULT_SKY_COLOR[0]
            rgb[i, j, 1] = DEFAULT_SKY_COLOR[1]
            rgb[i, j, 2] = DEFAULT_SKY_COLOR[2]
            # Write default values for other outputs
            depth[i, j] = inf_gpu
            sem[i, j] = 0
            # pts[i, j] remains NaN (default initialization)

# --- OBJ 导出 ---
# (保持不变 - Keep unchanged from your original code)
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
            # Iterate through all triangles
            for i in range(v0s.shape[0]):
                # Calculate vertex coordinates
                v0 = v0s[i]
                v1 = v0 + e1s[i]
                v2 = v0 + e2s[i]
                # Write vertex coordinates
                f.write(f"v {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
                f.write(f"v {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
                f.write(f"v {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
                # Write face definition using the indices of the written vertices
                # Using // to indicate no texture or normal indices
                f.write(f"f {vidx}// {vidx+1}// {vidx+2}//\n")
                vidx += 3 # Increment vertex index by 3 for the next triangle

            # --- Camera Frustum ---
            f.write('\n# Camera Frustum\n')
            f.write('o camera_frustum\n')
            # Write camera origin vertex
            f.write(f"v {cam_o[0]:.6f} {cam_o[1]:.6f} {cam_o[2]:.6f}\n")
            cam_v_start = vidx # Store index of the camera origin vertex
            vidx += 1
            # Calculate and write the four corner points of the far plane
            corners = []
            # Iterate through screen corners (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)
            for du_norm in [-0.5, 0.5]:
                for dv_norm in [-0.5, 0.5]: # Note: OBJ typically uses Y-up, so dv sign might need flip depending on convention
                    # Calculate direction to the corner
                    d_corner = cam_dir + (du_norm * screen_w * right) - (dv_norm * screen_h * up)
                    # Normalize the corner direction
                    d_corner /= np.linalg.norm(d_corner)
                    # Calculate the point on the far plane
                    corner_pt = cam_o + d_corner * far
                    corners.append(corner_pt)
                    # Write the corner vertex
                    f.write(f"v {corner_pt[0]:.6f} {corner_pt[1]:.6f} {corner_pt[2]:.6f}\n")
                    vidx += 1
            # Define lines connecting camera origin to corners and corners to form the far plane rectangle
            # Indices of the corner vertices (relative to cam_v_start)
            bl, tl, br, tr = cam_v_start + 1, cam_v_start + 2, cam_v_start + 3, cam_v_start + 4 # Adjust based on loop order
            # Lines from origin to corners
            f.write(f"l {cam_v_start} {bl}\n")
            f.write(f"l {cam_v_start} {tl}\n")
            f.write(f"l {cam_v_start} {br}\n")
            f.write(f"l {cam_v_start} {tr}\n")
            # Lines forming the far plane rectangle
            f.write(f"l {bl} {tl}\n")
            f.write(f"l {tl} {tr}\n")
            f.write(f"l {tr} {br}\n")
            f.write(f"l {br} {bl}\n")

            # --- Intersection Points ---
            f.write('\n# Intersection Points (Sampled)\n')
            f.write('o intersection_points\n')
            H_pts, W_pts = pts.shape[:2] # Get dimensions of the points array
            # Sample points to avoid huge OBJ files (e.g., every 64th pixel)
            step = max(1, H_pts // 64, W_pts // 64)
            point_v_start = vidx # Store starting index for point vertices
            num_pts_written = 0
            # Iterate through sampled pixels
            for i in range(0, H_pts, step):
                for j in range(0, W_pts, step):
                    p = pts[i, j]
                    # Write vertex only if it's not NaN (i.e., a valid hit occurred)
                    if not np.isnan(p[0]):
                        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                        vidx += 1
                        num_pts_written += 1
            # If points were written, create a point group
            if num_pts_written > 0:
                 f.write(f"g hit_points\n") # Group name
                 # 'p' command lists the indices of vertices to be rendered as points
                 f.write(f"p {' '.join(map(str, range(point_v_start, vidx)))}\n")

        print(f"    成功保存场景、视锥体和 {num_pts_written} 个采样点到 {filename}")
    except IOError as e:
        print(f"    错误: 无法写入 OBJ 文件 {filename}: {e}")
    except Exception as e:
        print(f"    保存 OBJ 时发生未知错误: {e}")


# --- 主流程 (MODIFIED Camera) ---
def main():
    global _GPU_AVAILABLE # Allow modifying the global GPU flag if errors occur
    total_t0 = _now() # Start total timer

    # --- 输出目录设置 ---
    # (保持不变)
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
    t0_scene = _now()
    # Calls the modified build_scene function with clustering and larger scale
    v0s, e1s, e2s, labels, colors, prim_indices = build_scene()
    num_triangles = len(v0s) # Get actual number of triangles generated
    log_step('场景构建', t0_scene)
    if num_triangles == 0:
        print("错误: 场景构建结果为 0 个三角形。正在退出。")
        return
    # Check final triangle count against target
    print(f"    场景包含 {num_triangles} 个三角形。")
    if num_triangles < 200000:
        print(f"    注意: 三角形数量 ({num_triangles}) 低于 200,000 的目标。可能需要调整对象数量或放置逻辑。")


    # --- 构建 BVH ---
    print('[2/7] 构建 BVH...')
    t0_bvh = _now()
    # Build BVH using the generated geometry and original indices
    flat_nodes, ordered_prim_indices = build_bvh(v0s, e1s, e2s, prim_indices)
    log_step('BVH 构建', t0_bvh)

    # --- Reorder Geometry Data Based on BVH ---
    print('[3/7] 根据 BVH 重新排序几何数据...')
    t0_reorder = _now()
    # (逻辑保持不变 - Logic unchanged, but applied to potentially larger arrays)
    # Check if the reordered indices match the number of triangles
    if len(ordered_prim_indices) != num_triangles:
        print(f"错误: BVH 返回的索引数量 ({len(ordered_prim_indices)}) 与三角形数量 ({num_triangles}) 不匹配。")
        # Fallback to using the original, unordered geometry data
        v0s_reordered = v0s
        e1s_reordered = e1s
        e2s_reordered = e2s
        labels_reordered = labels
        colors_reordered = colors
        print("    警告: 使用原始几何数据顺序进行光线追踪。")
    else:
        # Apply the reordering using the indices from the BVH build
        try:
            v0s_reordered = v0s[ordered_prim_indices]
            e1s_reordered = e1s[ordered_prim_indices]
            e2s_reordered = e2s[ordered_prim_indices]
            labels_reordered = labels[ordered_prim_indices]
            colors_reordered = colors[ordered_prim_indices]
        except IndexError as e:
             # Handle potential errors during reordering
             print(f"错误: 使用 ordered_prim_indices 重新排序几何数据时发生索引错误: {e}")
             print("    警告: 使用原始几何数据顺序进行光线追踪。")
             # Fallback to original order
             v0s_reordered = v0s
             e1s_reordered = e1s
             e2s_reordered = e2s
             labels_reordered = labels
             colors_reordered = colors

    log_step('几何数据重新排序', t0_reorder)

    # --- 相机设置 (MODIFIED for Level View) ---
    print('[4/7] 设置相机...')
    t0_cam = _now()

    # *** NEW CAMERA POSITION AND TARGET FOR LEVEL VIEW ***
    # Place camera somewhat low (like a window height), and further back,
    # looking towards the clusters which are now further out.
    # Adjust Y for height, X/Z for position relative to clusters.
    cam_o = np.array([-350, 25, -300], dtype=GEOMETRY_DTYPE) # Example: Back-left, relatively low
    # Target a point within the general area of the clusters, keeping Y similar for level view.
    # Aim towards the center/first few clusters area (e.g., around x=0, z=0 or slightly positive z).
    cam_t = np.array([50, 15, 100], dtype=GEOMETRY_DTYPE) # Example: Aiming towards central/further clusters
    # *** END MODIFICATION ***

    # World Up vector (usually Y-axis)
    cam_up_vec = np.array([0, 1, 0], dtype=GEOMETRY_DTYPE)

    # Calculate camera frame vectors (Forward, Right, Up)
    # Forward vector (view direction)
    cam_dir = cam_t - cam_o
    norm_cam_dir = np.linalg.norm(cam_dir)
    if norm_cam_dir < EPSILON:
        print("错误: 相机位置和目标点重合。")
        return
    cam_dir /= norm_cam_dir # Normalized forward vector

    # Calculate Right vector using cross product with world up
    right = np.cross(cam_dir, cam_up_vec)
    norm_right = np.linalg.norm(right)
    # Handle edge case: If camera looks straight up or down, cross product is zero.
    if norm_right < EPSILON:
        print("警告: 相机方向与向上向量平行。调整右向量。")
        # If looking straight up/down, use cross product with world X or Z axis
        if abs(cam_dir[1]) > 1.0 - EPSILON: # Check if looking almost vertically
             # Use cross product with a non-parallel vector like world Z
             alt_vec = np.array([0, 0, 1.0 if cam_dir[1] > 0 else -1.0], dtype=GEOMETRY_DTYPE)
             right = np.cross(alt_vec, cam_dir)
        else:
             # Default to world X if something else went wrong (should be rare)
             right = np.array([1, 0, 0], dtype=GEOMETRY_DTYPE)
        norm_right = np.linalg.norm(right) # Recalculate norm
        # Handle potential zero vector after adjustment (highly unlikely)
        if norm_right < EPSILON:
             print("错误: 无法计算有效的右向量。")
             return
    right /= norm_right # Normalize right vector

    # Calculate actual camera Up vector using cross product of right and forward
    up = np.cross(right, cam_dir)
    # No need to normalize 'up' if 'right' and 'cam_dir' are already normalized and orthogonal.

    # Image dimensions
    W, H = 1920, 1080
    # Field of View (adjust for desired perspective)
    fov_degrees = 60.0 # Slightly narrower FoV might suit the level view better
    fov_radians = np.deg2rad(fov_degrees)
    aspect_ratio = W / H
    # Calculate virtual screen plane dimensions based on FoV
    # screen_h = 2.0 * distance_to_screen * tan(fov_radians / 2.0)
    # Assuming distance_to_screen = 1 for simplicity
    screen_h = GEOMETRY_DTYPE(2.0 * np.tan(fov_radians / 2.0))
    screen_w = GEOMETRY_DTYPE(screen_h * aspect_ratio)

    log_step('相机设置', t0_cam)
    print(f"    分辨率: {W}x{H}, FoV: {fov_degrees} deg")
    print(f"    相机位置: [{cam_o[0]:.2f}, {cam_o[1]:.2f}, {cam_o[2]:.2f}]")
    print(f"    相机目标: [{cam_t[0]:.2f}, {cam_t[1]:.2f}, {cam_t[2]:.2f}]")
    print(f"    相机方向: [{cam_dir[0]:.2f}, {cam_dir[1]:.2f}, {cam_dir[2]:.2f}]")


    # --- 光线追踪 ---
    print('[5/7] 光线追踪...')
    t0_raytrace = _now()
    rgb, depth, sem_lbl, pts = None, None, None, None # Initialize output variables
    # Determine whether to use GPU or CPU
    # Use GPU if available and the BVH was successfully built (num_nodes > 0)
    use_gpu = _GPU_AVAILABLE and flat_nodes.shape[0] > 0

    # --- Optional: Force CPU for debugging ---
    # print("---!! 强制使用 CPU 进行调试 !! ---")
    # use_gpu = False
    # ---

    if use_gpu:
        print("    尝试使用 GPU (CUDA + BVH)...")
        try:
            t_upload_start = _now()
            # --- Upload Data to GPU ---
            # Upload BVH nodes and reordered geometry/attribute data
            d_flat_nodes = cuda.to_device(flat_nodes)
            d_v0s    = cuda.to_device(v0s_reordered)
            d_e1s    = cuda.to_device(e1s_reordered)
            d_e2s    = cuda.to_device(e2s_reordered)
            d_labels = cuda.to_device(labels_reordered)
            d_colors = cuda.to_device(colors_reordered)
            # Upload camera parameters (as device arrays or passed directly if small)
            # Passing NumPy arrays directly often works for small, read-only data
            # d_cam_o   = cuda.to_device(cam_o) # Can often pass cam_o directly
            # d_cam_dir = cuda.to_device(cam_dir)
            # d_right   = cuda.to_device(right)
            # d_up      = cuda.to_device(up)

            # Output buffers (allocate on GPU)
            d_rgb    = cuda.device_array((H, W, 3), dtype=COLOR_DTYPE)
            # Initialize depth/sem/pts on host then copy, or use cuda.pinned_array for faster H2D/D2H
            depth_init_host = np.full((H, W), INF, dtype=DEPTH_DTYPE)
            d_depth  = cuda.to_device(depth_init_host)
            sem_init_host = np.zeros((H, W), dtype=LABEL_DTYPE)
            d_sem    = cuda.to_device(sem_init_host)
            pts_init_host = np.full((H, W, 3), np.nan, dtype=POINT_DTYPE)
            d_pts    = cuda.to_device(pts_init_host)
            log_step('GPU 数据上传', t_upload_start)

            # --- Kernel Launch Configuration ---
            threads_per_block = (16, 16) # Typical block size for 2D problems
            # Calculate grid dimensions needed to cover all pixels
            blocks_per_grid_x = (W + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid_y = (H + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x) # Grid dimensions (y, x)
            print(f"    启动 CUDA 内核: Grid={blocks_per_grid}, Block={threads_per_block}")

            t_kernel_start = _now()
            # Launch the kernel
            raytrace_cuda_bvh_kernel[blocks_per_grid, threads_per_block](
                d_flat_nodes, d_v0s, d_e1s, d_e2s, d_labels, d_colors,
                cam_o, cam_dir, right, up, # Pass camera params directly
                screen_w, screen_h, W, H,  # Pass scalar parameters
                d_rgb, d_depth, d_sem, d_pts # Pass output device arrays
            )
            cuda.synchronize() # Wait for kernel to finish execution
            log_step('GPU BVH 内核执行', t_kernel_start)

            # --- Download Results from GPU to Host ---
            t_download_start = _now()
            rgb     = d_rgb.copy_to_host()
            depth   = d_depth.copy_to_host()
            sem_lbl = d_sem.copy_to_host()
            pts     = d_pts.copy_to_host()
            log_step('GPU 数据下载', t_download_start)
            print("    GPU 执行成功。")

        # --- Error Handling for GPU Execution ---
        except cuda.cudadrv.driver.CudaAPIError as e:
            # Handle CUDA driver/API errors (e.g., out of memory)
            print(f"\n---!! CUDA API 错误: {e} !!---")
            print("---!! 可能是显存不足或驱动问题。回退到 CPU 执行。 !!---\n")
            _GPU_AVAILABLE = False # Disable GPU for this run
            use_gpu = False        # Set flag to trigger CPU execution
        except AttributeError as e:
            # Handle specific errors like missing '.view()' if Numba version is incompatible
             if "'float' object has no attribute 'view'" in str(e) or \
                "'DeviceFunctionTemplate' object has no attribute 'view'" in str(e):
                 print(f"\n---!! GPU 属性错误: {e} !!---")
                 print("---!! Numba CUDA 的 .view() 用法可能存在问题或版本不兼容。回退到 CPU。 !!---\n")
             else:
                 # Handle other attribute errors
                 print(f"\n---!! GPU 执行期间发生属性错误: {e} !!---")
                 print("---!! 回退到 CPU 执行。 !!---\n")
             _GPU_AVAILABLE = False
             use_gpu = False
        except Exception as e:
            # Handle any other unexpected errors during GPU execution
            print(f"\n---!! GPU 执行期间发生未知错误: {e} !!---")
            print(f"---!! 错误类型: {type(e).__name__}")
            print("---!! 回退到 CPU 执行。 !!---\n")
            _GPU_AVAILABLE = False
            use_gpu = False

    # --- CPU Execution Fallback or Default ---
    if not use_gpu:
        print("    使用 CPU (Numba JIT + BVH)...")
        t_cpu_start = _now()
        # Ensure parallel execution is enabled for performance on CPU
        print("    注意: CPU 使用并行计算 (Numba @njit(parallel=True))。")
        # Call the CPU ray tracing function
        rgb, depth, sem_lbl, pts = raytrace_cpu_bvh(
            flat_nodes, v0s_reordered, e1s_reordered, e2s_reordered,
            labels_reordered, colors_reordered,
            cam_o, cam_dir, right, up,
            screen_w, screen_h, W, H
        )
        log_step('CPU BVH 光线追踪执行', t_cpu_start)

    log_step('总光线追踪', t0_raytrace) # Log total ray tracing time (GPU or CPU)

    # --- 图像导出 ---
    print('[6/7] 保存输出图像...')
    t0_save = _now()

    # Save RGB view image
    if rgb is not None:
        fn_view = os.path.join(outd, f'view_{ts}.png')
        try:
            Image.fromarray(rgb).save(fn_view)
            print(f"    已保存视图: {fn_view}")
        except Exception as e:
            print(f"    错误: 保存视图图像失败: {e}")
    else:
        print("    跳过视图保存 (无 RGB 数据).")

    # Save Depth map image
    if depth is not None:
        fn_depth = os.path.join(outd, f'depth_{ts}.png')
        try:
            # Normalize depth map for visualization
            valid_depth = depth[np.isfinite(depth)] # Filter out infinite values
            if len(valid_depth) > 0:
                dmin = np.min(valid_depth)
                # Use percentile for max visualization range to handle outliers
                dmax_vis = np.percentile(valid_depth, 99.5)
                # Clamp max visualization distance if needed
                dmax_vis = min(dmax_vis, 1000.0) # Adjust max visualization distance as needed
                print(f"    深度范围 (有限值): {dmin:.2f} 到 {np.max(valid_depth):.2f} (可视化上限: {dmax_vis:.2f})")

                # Avoid division by zero if min and max are too close
                if dmax_vis <= dmin:
                    dmax_vis = dmin + 1.0

                scale = (dmax_vis - dmin)
                if scale < EPSILON:
                    scale = EPSILON # Prevent division by zero

                # Normalize depth values to [0, 1] based on visualization range
                depth_normalized = (depth - dmin) / scale
                # Clip values to [0, 1] and scale to [0, 254] for grayscale image
                # Use 255 for infinite depth (sky)
                depth_clipped = np.clip(depth_normalized * 254, 0, 254)
                # Create grayscale image: finite values are mapped, infinite values become 255 (white)
                dmap = np.where(np.isfinite(depth), depth_clipped, 255).astype(np.uint8)

                Image.fromarray(dmap, 'L').save(fn_depth) # Save as grayscale ('L')
                print(f"    已保存深度图: {fn_depth}")
            else:
                # Handle case where all depth values are infinite
                print("    跳过深度图保存 (无有效深度值).")
                # Save a black image as placeholder
                Image.new('L', (W, H), 0).save(fn_depth)
        except RuntimeWarning as e:
             # Catch warnings often occurring with all-infinite depth
             print(f"    保存深度图时发生运行时警告: {e}")
             print(f"    这通常发生在所有深度值都是 INF 时。")
             try:
                 # Attempt to save a black image
                 Image.new('L', (W, H), 0).save(fn_depth)
             except Exception as save_e:
                 print(f"    尝试保存黑色深度图也失败: {save_e}")
        except Exception as e:
            print(f"    错误: 保存深度图像失败: {e}")
    else:
        print("    跳过深度图保存 (无深度数据).")

    # Save Semantic map image
    if sem_lbl is not None:
        fn_sem = os.path.join(outd, f'semantic_{ts}.png')
        try:
            # Create a color palette mapping labels to colors
            # Use black for background label 0
            palette = {label: color_array.tolist() for label, color_array in C.items()}
            palette[0] = PALETTE_BACKGROUND_COLOR # Override label 0 for background

            # Create an empty RGB image
            sem_img = np.zeros((H, W, 3), dtype=COLOR_DTYPE)
            # Get unique labels present in the semantic map
            unique_labels = np.unique(sem_lbl)
            # Apply colors based on the palette
            for label_id in unique_labels:
                 if label_id in palette: # Check if label exists in palette
                     mask = (sem_lbl == label_id) # Create mask for pixels with this label
                     sem_img[mask] = palette[label_id] # Apply color to masked pixels
                 # else: print(f"警告: 语义标签 {label_id} 未在调色板中找到。")

            Image.fromarray(sem_img).save(fn_sem)
            print(f"    已保存语义图: {fn_sem}")
        except Exception as e:
            print(f"    错误: 保存语义图像失败: {e}")
    else:
        print("    跳过语义图保存 (无语义数据).")

    log_step('图像导出', t0_save)

    # --- OBJ 导出 ---
    print('[7/7] 保存组合 OBJ 文件...')
    t0_obj = _now()
    fn_obj = os.path.join(outd, f'combined_{ts}.obj')
    # Determine far distance for camera frustum visualization
    far_dist = 500.0 # Default far distance
    if depth is not None:
        valid_depth = depth[np.isfinite(depth)]
        if len(valid_depth) > 0:
            # Set far distance slightly beyond the maximum finite depth found
            far_dist = max(far_dist, np.max(valid_depth) * 1.1)

    # Save OBJ if intersection points were generated
    if pts is not None:
         save_combined_obj(fn_obj, v0s, e1s, e2s, # Original geometry data
                           cam_o, cam_dir, right, up, screen_w, screen_h, # Camera params
                           pts, far=far_dist) # Intersection points and far distance
    else:
        print("    跳过 OBJ 保存 (无交点数据).")
    log_step('OBJ 导出', t0_obj)

    # --- 完成 ---
    print('\n[完成]')
    total_time = _now() - total_t0
    print(f"总执行时间: {total_time:.2f}s")
    print(f"输出已保存至: {outd}")


# --- Script Entry Point ---
if __name__ == '__main__':
    main()
