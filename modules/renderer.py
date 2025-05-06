# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import math
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Sequence # 添加了 Sequence
import numpy as np
from PIL import Image
from numba import njit, prange, cuda


# -- GPU 可用性检查 ---
_GPU_AVAILABLE = False
try:
    if cuda.is_available():
        try:
            cuda.detect()
            device = cuda.get_current_device()
            print(f"CUDA 可用: True")
            print(f"使用 GPU: {device.name.decode('utf-8', errors='replace')}") # 使用 utf-8 解码
            _GPU_AVAILABLE = True
        except Exception as e_detect:
            print(f"检测 CUDA 设备或获取名称时出错: {e_detect}")
            print("CUDA 可能仍然可用，但信息不完整。")
            _GPU_AVAILABLE = True # 假设仍然可用，但在渲染时会再次检查
    else:
        print("CUDA 不可用: 未检测到兼容设备或驱动程序。渲染将无法进行。")
except ImportError:
    print("Numba CUDA 扩展未安装或导入失败。渲染将无法进行。")
except Exception as e:
    print(f"CUDA 初始化期间发生未知错误: {e}。渲染可能无法进行。")


# --- 常量 ---
GEOMETRY_DTYPE = np.float32
INDEX_DTYPE = np.int32
COLOR_DTYPE = np.uint8
LABEL_DTYPE = np.int32
DEPTH_DTYPE = np.float32
POINT_DTYPE = np.float32
BVH_NODE_DTYPE = np.float32
INF = GEOMETRY_DTYPE(np.inf)
GPU_INF = GEOMETRY_DTYPE(1e20) # GPU上的无穷大表示 (避免溢出)

INTERSECTION_EPSILON = GEOMETRY_DTYPE(1e-5)
# SHADOW_BIAS 不再需要

# --- 光照默认值已移除 ---
# 默认值现在应在主脚本的 render_params 中设置

# --- BVH 常量 ---
BVH_MAX_LEAF_SIZE = 4
BVH_NODE_FIELDS = 8 # AABB min (3), AABB max (3), Info1 (1), Info2 (1)

# --- 语义 ---
SKY_LABEL = -1 # 在语义图中代表天空
# DEFAULT_SKY_COLOR 现在从 render_params 读取

# --- Gamma ---
GAMMA_OUT = 2.2               # 输出到 sRGB 的 Gamma
INV_GAMMA_OUT = 1.0 / GAMMA_OUT


# --- 工具函数 ---
def _now() -> float:
    """获取当前高精度时间。"""
    return time.perf_counter()


def log_step(title: str, t0: float) -> None:
    """打印步骤耗时。"""
    print(f"    {title} 耗时 {_now() - t0:.3f}s")


@njit # Numba JIT 加速
def int32_to_float32_bits(val_int32):
    """将 int32 按位转换为 float32。"""
    int_array = np.array([val_int32], dtype=INDEX_DTYPE)
    return int_array.view(BVH_NODE_DTYPE)[0]


@njit # Numba JIT 加速
def float32_to_int32_bits(val_float32):
    """将 float32 按位转换为 int32。"""
    float_array = np.array([val_float32], dtype=BVH_NODE_DTYPE)
    return float_array.view(INDEX_DTYPE)[0]


# --- BVH 构建函数 (Numba JIT 在 CPU 上运行 - GPU 需要) ---
# 这些保持不变，因为 BVH 构建在 CPU 上进行，然后传输到 GPU
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
        # 返回一个无效的包围盒
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


@njit
def recursive_build_numba(current_node_idx, nodes_used, flat_nodes, prim_indices,
                          start_idx, end_idx,
                          tri_aabbs_min, tri_aabbs_max, tri_centers):
    """递归构建 BVH 结构 (Numba)。"""
    num_prims = end_idx - start_idx
    node = flat_nodes[current_node_idx]

    # 获取当前节点覆盖的图元索引切片
    indices_slice = prim_indices[start_idx:end_idx]

    # 计算当前节点包围盒
    aabb_min, aabb_max = calculate_bounds(indices_slice, tri_aabbs_min, tri_aabbs_max)
    node[0:3] = aabb_min
    node[3:6] = aabb_max

    # 如果图元数量小于等于叶子节点阈值，则创建叶子节点
    if num_prims <= BVH_MAX_LEAF_SIZE:
        # Info1: 存储第一个图元在排序后数组中的偏移量
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx))
        # Info2: 存储负的图元数量，表示这是一个叶子节点
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims))
        return

    # --- 否则，创建内部节点并递归 ---
    # 计算包围盒范围并选择最长的轴进行分割
    extent = aabb_max - aabb_min
    split_axis = np.argmax(extent)

    # 如果包围盒在最长轴上非常小（接近退化），也创建叶子节点
    if extent[split_axis] < INTERSECTION_EPSILON:
        node[6] = int32_to_float32_bits(INDEX_DTYPE(start_idx))
        node[7] = int32_to_float32_bits(INDEX_DTYPE(-num_prims))
        return

    # 计算分割点（包围盒中心）
    split_coord = (aabb_min[split_axis] + aabb_max[split_axis]) * 0.5

    # 使用类似于快速排序的划分方法，将图元按中心点位置分配到左右子集
    mid_point = start_idx
    for i in range(start_idx, end_idx):
        prim_idx = prim_indices[i]
        if tri_centers[prim_idx, split_axis] < split_coord:
            # 交换元素，将中心点在分割点左侧的图元移到前面
            prim_indices[i], prim_indices[mid_point] = prim_indices[mid_point], prim_indices[i]
            mid_point += 1

    # 处理划分不均匀的情况（所有图元都在一侧）
    if mid_point == start_idx or mid_point == end_idx:
        mid_point = start_idx + num_prims // 2 # 强制居中分割

    # 分配子节点索引
    left_child_idx = nodes_used[0]
    nodes_used[0] += 1
    right_child_idx = nodes_used[0]
    nodes_used[0] += 1

    # Info1: 存储左子节点索引
    node[6] = int32_to_float32_bits(INDEX_DTYPE(left_child_idx))
    # Info2: 存储右子节点索引 (正数表示内部节点)
    node[7] = int32_to_float32_bits(INDEX_DTYPE(right_child_idx)) # 注意这里是右子节点索引

    # 递归构建左右子树
    recursive_build_numba(left_child_idx, nodes_used, flat_nodes, prim_indices,
                          start_idx, mid_point, # 左子树范围
                          tri_aabbs_min, tri_aabbs_max, tri_centers)
    recursive_build_numba(right_child_idx, nodes_used, flat_nodes, prim_indices,
                          mid_point, end_idx,   # 右子树范围
                          tri_aabbs_min, tri_aabbs_max, tri_centers)


def build_bvh(v0s, e1s, e2s, prim_indices_in):
    """构建 BVH 加速结构 (在 CPU 上使用 Numba)。"""
    N = len(prim_indices_in)
    if N == 0:
        print("    场景为空，跳过 BVH 构建。")
        return np.empty((0, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE), np.empty(0, dtype=INDEX_DTYPE)

    print("    开始 BVH 构建...")
    t0_total_bvh = _now()

    # --- 1. 预计算每个三角形的 AABB 和中心点 ---
    t0_precompute = _now()
    tri_aabbs_min = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_aabbs_max = np.empty((N, 3), dtype=BVH_NODE_DTYPE)
    tri_centers = np.empty((N, 3), dtype=BVH_NODE_DTYPE)

    @njit(parallel=True, fastmath=True)
    def precompute_bounds_centers_parallel(num_tris, v0s_n, e1s_n, e2s_n,
                                           aabbs_min_out, aabbs_max_out, centers_out):
        """用于预计算的并行 Numba 函数。"""
        for i in prange(num_tris): # 使用 prange 进行并行计算
            v0, e1, e2 = v0s_n[i], e1s_n[i], e2s_n[i]
            aabb_min, aabb_max = calculate_tri_aabb_numba(v0, e1, e2)
            aabbs_min_out[i] = aabb_min
            aabbs_max_out[i] = aabb_max
            centers_out[i] = (aabb_min + aabb_max) * 0.5

    # 调用并行预计算函数
    precompute_bounds_centers_parallel(N, v0s, e1s, e2s,
                                       tri_aabbs_min, tri_aabbs_max, tri_centers)
    log_step('BVH 预计算 (AABB 和中心点)', t0_precompute)

    # --- 2. 准备递归构建所需的数据结构 ---
    # 预分配足够大的节点数组 (最坏情况是 2N-1 个节点)
    max_nodes = max(1, 2 * N - 1)
    flat_nodes = np.zeros((max_nodes, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE)
    # 复制一份图元索引，递归函数会对其进行原地排序
    ordered_prim_indices = np.copy(prim_indices_in)
    # 用于跟踪下一个可用的节点索引，从 1 开始 (0 是根节点)
    nodes_used = np.array([1], dtype=INDEX_DTYPE)

    # --- 3. 执行递归构建 ---
    t0_recursive = _now()
    recursive_build_numba(0, # 从根节点 (索引 0) 开始
                          nodes_used, flat_nodes, ordered_prim_indices,
                          0, N, # 初始范围包含所有图元
                          tri_aabbs_min, tri_aabbs_max, tri_centers)
    log_step('BVH 递归构建', t0_recursive)

    # --- 4. 清理和返回结果 ---
    # 裁剪节点数组到实际使用的数量
    actual_nodes_used = nodes_used[0]
    flat_nodes = flat_nodes[:actual_nodes_used]
    print(f"    BVH 构建完成: 使用了 {actual_nodes_used} 个节点。")

    # 打印根节点信息 (用于调试)
    if actual_nodes_used > 0:
        root_min = flat_nodes[0, 0:3]
        root_max = flat_nodes[0, 3:6]
        print(f"    根节点 AABB Min: [{root_min[0]:.2f}, {root_min[1]:.2f}, {root_min[2]:.2f}], "
              f"Max: [{root_max[0]:.2f}, {root_max[1]:.2f}, {root_max[2]:.2f}]")

    log_step('BVH 构建总耗时', t0_total_bvh)
    return flat_nodes, ordered_prim_indices


# --- GPU 光线求交函数 ---
if _GPU_AVAILABLE: # 仅在 GPU 可用时定义
    @cuda.jit(device=True, inline=True)
    def ray_tri_intersect_gpu(orig, dir, v0, e1, e2):
        """Moller-Trumbore 光线-三角形相交测试 (GPU 设备函数)。
        返回交点距离 t，如果未命中则返回 GPU_INF。
        """
        eps_gpu = INTERSECTION_EPSILON # 使用全局定义的 Epsilon
        inf_gpu = GPU_INF # 使用定义的 GPU 无穷大

        # 计算行列式因子 a
        h0 = dir[1] * e2[2] - dir[2] * e2[1]
        h1 = dir[2] * e2[0] - dir[0] * e2[2]
        h2 = dir[0] * e2[1] - dir[1] * e2[0]
        a = e1[0] * h0 + e1[1] * h1 + e1[2] * h2

        # 检查光线是否与三角形平面平行或接近平行 (a 接近 0)
        # 或三角形是否退化 (面积接近 0)
        if abs(a) < eps_gpu:
            return inf_gpu # 未命中

        # 计算 1/a
        f = GEOMETRY_DTYPE(1.0) / a

        # 计算第一个重心坐标 u
        s0 = orig[0] - v0[0]
        s1 = orig[1] - v0[1]
        s2 = orig[2] - v0[2]
        u = f * (s0 * h0 + s1 * h1 + s2 * h2)

        # 检查 u 是否在 [0, 1] 范围内
        if u < 0.0 or u > 1.0:
            return inf_gpu # 未命中

        # 计算第二个重心坐标 v
        q0 = s1 * e1[2] - s2 * e1[1]
        q1 = s2 * e1[0] - s0 * e1[2]
        q2 = s0 * e1[1] - s1 * e1[0]
        v = f * (dir[0] * q0 + dir[1] * q1 + dir[2] * q2)

        # 检查 v 和 u+v 是否在 [0, 1] 范围内
        if v < 0.0 or u + v > 1.0:
            return inf_gpu # 未命中

        # 计算交点距离 t
        t = f * (e2[0] * q0 + e2[1] * q1 + e2[2] * q2)

        # 只有当 t 大于一个小的正值（避免起点处的自相交）时才认为是有效命中
        # 确保交点在光线方向上，而不是起点后面
        return t if t > eps_gpu else inf_gpu

    @cuda.jit(device=True, inline=True)
    def ray_aabb_intersect_gpu(orig, dir_inv, t_min_global, node_aabb_min, node_aabb_max):
        """光线-AABB 相交测试 (Slab test) (GPU 设备函数)。
        检查光线是否与 AABB 相交，并且相交区间与 [epsilon, t_min_global] 有重叠。
        返回 True 表示可能命中（需要进一步检查子节点或图元），False 表示肯定未命中。
        """
        # 注意：这里的 t_near/t_far 是相对于光线起点的距离
        t_near = -INF # 初始化为负无穷
        t_far = INF   # 初始化为正无穷
        eps_aabb = INTERSECTION_EPSILON # 使用与三角形求交相同的 epsilon

        # 对 x, y, z 三个轴进行 slab 测试
        for k in range(3):
            inv_d = dir_inv[k] # 预计算的光线方向倒数
            aabb_min_k = node_aabb_min[k]
            aabb_max_k = node_aabb_max[k]

            # 计算光线进入和离开当前 slab 的 t 值
            t1 = (aabb_min_k - orig[k]) * inv_d
            t2 = (aabb_max_k - orig[k]) * inv_d

            # 确保 t1 <= t2
            if t1 > t2: t1, t2 = t2, t1

            # 更新全局的 t_near 和 t_far
            # t_near 是所有 slab 进入时间的最大值
            # t_far 是所有 slab 离开时间的最小值
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

            # --- 剪枝条件 ---
            # 1. t_near >= t_far:
            #    光线进入最后一个 slab 的时间晚于或等于离开第一个 slab 的时间，
            #    意味着所有 slab 的交集为空，光线不可能穿过 AABB。
            # 2. t_far < eps_aabb:
            #    整个相交区间都在光线起点之前或非常接近起点，
            #    我们只关心起点之后的交点。
            # 3. t_near >= t_min_global:
            #    AABB 的最近可能交点 (t_near) 比当前已知的最近物体交点 (t_min_global) 还要远，
            #    这个 AABB 及其内部不可能包含更近的命中。
            if t_near >= t_far or t_far < eps_aabb or t_near >= t_min_global:
                return False # 未命中或被剪枝

        # 如果通过了所有轴的测试，则光线与 AABB 相交，并且可能包含更近的命中
        return True # 命中


# --- GPU 光线追踪内核 (已修改，移除阴影和高光，合并着色) ---
if _GPU_AVAILABLE:
    @cuda.jit
    def raytrace_cuda_bvh_kernel(
            # --- 场景数据 (设备指针) ---
            d_flat_nodes, d_v0s, d_e1s, d_e2s,
            d_normals, d_labels, d_colors,
            num_tris_total_gpu, # 显式传递三角形总数
            # --- 相机参数 (Host 值) ---
            cam_o, cam_dir, right, up,
            # --- 渲染参数 (Host 值 / 设备指针) ---
            d_sun_dir,          # 动态上传的设备数组 (vec3) - 归一化
            d_sky_color,        # 动态上传的设备数组 (vec3 uint8)
            scr_w, scr_h, W, H,   # Host floats/ints
            ambient_light_gpu,    # Host float
            sun_intensity_gpu,    # Host float
            # --- 雾效参数 ---
            fog_enabled_gpu,      # Host bool
            d_fog_color_float,  # 动态上传的设备数组 (vec3 float [0,1])
            fog_density_gpu,      # Host float
            # --- 输出缓冲 (设备指针) ---
            rgb, depth, sem, pts
    ):
        """GPU 光线追踪内核 (BVH, 统一光照和雾效)。"""
        i, j = cuda.grid(2) # 像素坐标 (行 i, 列 j)
        if i >= H or j >= W: return # 边界检查

        # --- 常量和本地变量 ---
        inf_gpu = GPU_INF
        eps_gpu = INTERSECTION_EPSILON
        sky_label_gpu = SKY_LABEL

        # 每个线程的本地数组，用于计算
        ray_dir = cuda.local.array(3, dtype=GEOMETRY_DTYPE)       # 光线方向
        ray_dir_inv = cuda.local.array(3, dtype=GEOMETRY_DTYPE) # 光线方向的倒数
        node_aabb_min = cuda.local.array(3, dtype=BVH_NODE_DTYPE) # 节点 AABB 最小值
        node_aabb_max = cuda.local.array(3, dtype=BVH_NODE_DTYPE) # 节点 AABB 最大值
        hit_normal = cuda.local.array(3, dtype=GEOMETRY_DTYPE)    # 命中点的法线 (面向相机)
        hit_point = cuda.local.array(3, dtype=GEOMETRY_DTYPE)     # 命中点坐标
        base_color_float = cuda.local.array(3, dtype=GEOMETRY_DTYPE) # 物体基础颜色 [0,1]
        shaded_color_float = cuda.local.array(3, dtype=GEOMETRY_DTYPE) # 光照计算后的颜色 (线性空间)
        final_rgb_float = cuda.local.array(3, dtype=GEOMETRY_DTYPE) # 应用雾效后的最终颜色 (线性空间)

        num_nodes = d_flat_nodes.shape[0] # 从设备数组获取节点数

        # 1. 计算主光线方向 ray_dir (与之前相同)
        u_norm = (GEOMETRY_DTYPE(j) + 0.5) / W - 0.5
        v_norm = (GEOMETRY_DTYPE(i) + 0.5) / H - 0.5 # Y 轴向下 (图像坐标系)
        # 计算屏幕点在相机坐标系下的方向
        ray_dir[0] = cam_dir[0] + u_norm * scr_w * right[0] - v_norm * scr_h * up[0]
        ray_dir[1] = cam_dir[1] + u_norm * scr_w * right[1] - v_norm * scr_h * up[1]
        ray_dir[2] = cam_dir[2] + u_norm * scr_w * right[2] - v_norm * scr_h * up[2]
        # 归一化光线方向
        nrm_sq = ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2
        if nrm_sq < eps_gpu**2: # 处理无效光线 (例如方向向量长度接近零)
             # 设置为天空颜色并返回
             rgb[i, j, 0] = d_sky_color[0]; rgb[i, j, 1] = d_sky_color[1]; rgb[i, j, 2] = d_sky_color[2]
             depth[i, j] = inf_gpu; sem[i, j] = sky_label_gpu
             pts[i, j, 0] = GEOMETRY_DTYPE(np.nan); pts[i, j, 1] = GEOMETRY_DTYPE(np.nan); pts[i, j, 2] = GEOMETRY_DTYPE(np.nan)
             return
        inv_nrm = GEOMETRY_DTYPE(1.0) / math.sqrt(nrm_sq)
        ray_dir[0] *= inv_nrm; ray_dir[1] *= inv_nrm; ray_dir[2] *= inv_nrm

        # 预计算方向倒数，用于 AABB 测试
        for k in range(3):
            rd_k = ray_dir[k]
            if abs(rd_k) < eps_gpu:
                # 如果方向在某个轴上接近 0，使用带符号的无穷大避免除零
                ray_dir_inv[k] = math.copysign(inf_gpu, rd_k)
            else:
                ray_dir_inv[k] = GEOMETRY_DTYPE(1.0) / rd_k

        # 2. 初始化主光线遍历
        tmin = inf_gpu      # 当前找到的最近交点距离，初始化为无穷大
        hit_prim_idx = -1   # 命中图元的索引，初始化为 -1 (未命中)

        # 处理空场景
        if num_nodes == 0:
            rgb[i, j, 0] = d_sky_color[0]; rgb[i, j, 1] = d_sky_color[1]; rgb[i, j, 2] = d_sky_color[2]
            depth[i, j] = inf_gpu; sem[i, j] = sky_label_gpu
            pts[i, j, 0] = GEOMETRY_DTYPE(np.nan); pts[i, j, 1] = GEOMETRY_DTYPE(np.nan); pts[i, j, 2] = GEOMETRY_DTYPE(np.nan)
            return

        # --- BVH 遍历栈 (本地) ---
        BVH_GPU_STACK_SIZE = 64 # 定义栈的最大深度
        node_stack = cuda.local.array(BVH_GPU_STACK_SIZE, dtype=INDEX_DTYPE)
        stack_ptr = 0 # 栈顶指针
        node_stack[stack_ptr] = 0 # 从根节点开始
        stack_ptr += 1

        # 3. BVH 遍历主光线 (查找最近交点)
        while stack_ptr > 0:
            # --- 弹出当前节点 ---
            stack_ptr -= 1
            node_idx = node_stack[stack_ptr]

            # 基本的有效性检查 (理论上不应发生，但作为安全措施)
            if node_idx < 0 or node_idx >= num_nodes: continue

            # --- 获取节点 AABB ---
            node_aabb_min[0] = d_flat_nodes[node_idx, 0]; node_aabb_min[1] = d_flat_nodes[node_idx, 1]; node_aabb_min[2] = d_flat_nodes[node_idx, 2]
            node_aabb_max[0] = d_flat_nodes[node_idx, 3]; node_aabb_max[1] = d_flat_nodes[node_idx, 4]; node_aabb_max[2] = d_flat_nodes[node_idx, 5]

            # --- AABB 求交测试 (剪枝) ---
            # 如果光线不与当前节点的 AABB 相交，或者相交点比已知的最近点还远，则跳过此节点及其子树
            if not ray_aabb_intersect_gpu(cam_o, ray_dir_inv, tmin, node_aabb_min, node_aabb_max):
                continue

            # --- 处理节点 (叶子或内部) ---
            # 读取存储在节点信息字段中的数据
            info1_bits_f = d_flat_nodes[node_idx, 6] # Info1 (float32 bits)
            info2_bits_f = d_flat_nodes[node_idx, 7] # Info2 (float32 bits)
            info2_val = info2_bits_f.view(INDEX_DTYPE) # 将 Info2 转回 int32

            # 判断是叶子节点还是内部节点
            if info2_val < 0: # 叶子节点 (图元数量为负数)
                prim_count = -info2_val # 获取图元数量
                prim_offset = info1_bits_f.view(INDEX_DTYPE) # 获取第一个图元的偏移量

                # 遍历叶子节点中的所有图元（三角形）
                for p_idx_offset in range(prim_count):
                    current_prim_idx = prim_offset + p_idx_offset
                    # 边界检查，确保索引有效 (理论上 BVH 构建应保证，但加上更安全)
                    if current_prim_idx < num_tris_total_gpu:
                        # 从设备内存获取三角形数据 (v0, e1, e2)
                        tri_v0 = d_v0s[current_prim_idx]
                        tri_e1 = d_e1s[current_prim_idx]
                        tri_e2 = d_e2s[current_prim_idx]

                        # 与三角形进行求交测试
                        t = ray_tri_intersect_gpu(cam_o, ray_dir, tri_v0, tri_e1, tri_e2)

                        # 如果找到更近的交点，更新 tmin 和 hit_prim_idx
                        if t < tmin:
                            tmin = t
                            hit_prim_idx = current_prim_idx
                            # 注意：这里不需要读取法线，在循环结束后统一读取最终命中的法线

            else: # 内部节点 (info2_val 是右子节点索引)
                left_child_idx = info1_bits_f.view(INDEX_DTYPE) # 左子节点索引存储在 Info1
                right_child_idx = info2_val                     # 右子节点索引存储在 Info2

                # 将子节点压入栈中继续遍历 (检查栈空间)
                if stack_ptr + 2 <= BVH_GPU_STACK_SIZE:
                    # 通常先压入较远的子节点，以便优先处理较近的子节点
                    # (这里简单地按左右顺序压入，优化可以后续考虑)
                    # 检查子节点索引是否有效
                    if right_child_idx >= 0 and right_child_idx < num_nodes:
                        node_stack[stack_ptr] = right_child_idx; stack_ptr += 1
                    if left_child_idx >= 0 and left_child_idx < num_nodes:
                        node_stack[stack_ptr] = left_child_idx; stack_ptr += 1
                # else: 栈溢出 (对于合理深度的 BVH 和栈大小，不太可能发生)

        # --- 结束 BVH 遍历 ---

        # 4. 处理命中结果
        if hit_prim_idx >= 0:  # 光线命中物体
            # --- 存储主要输出 (深度, 语义, 命中点) ---
            depth[i, j] = tmin
            sem[i, j] = d_labels[hit_prim_idx] # 从设备数组读取标签
            # 计算命中点坐标
            hit_point[0] = cam_o[0] + ray_dir[0] * tmin
            hit_point[1] = cam_o[1] + ray_dir[1] * tmin
            hit_point[2] = cam_o[2] + ray_dir[2] * tmin
            pts[i, j, 0] = hit_point[0]; pts[i, j, 1] = hit_point[1]; pts[i, j, 2] = hit_point[2]

            # --- 获取基础颜色 (语义颜色) ---
            color_uint8 = d_colors[hit_prim_idx] # 从设备数组读取颜色 (uint8)
            # 转换为 [0, 1] 范围的浮点数
            base_color_float[0] = GEOMETRY_DTYPE(color_uint8[0]) / 255.0
            base_color_float[1] = GEOMETRY_DTYPE(color_uint8[1]) / 255.0
            base_color_float[2] = GEOMETRY_DTYPE(color_uint8[2]) / 255.0

            # --- 法线处理 ---
            # 从设备数组读取最终命中的法线到本地 hit_normal
            hit_normal[0] = d_normals[hit_prim_idx, 0]
            hit_normal[1] = d_normals[hit_prim_idx, 1]
            hit_normal[2] = d_normals[hit_prim_idx, 2]
            # 检查法线方向，确保它指向光线来源（相机）
            # 如果法线和光线方向点积为正，说明法线背离相机，需要翻转
            normal_dot_ray = hit_normal[0] * ray_dir[0] + hit_normal[1] * ray_dir[1] + hit_normal[2] * ray_dir[2]
            if normal_dot_ray > 0.0:
                hit_normal[0] = -hit_normal[0]
                hit_normal[1] = -hit_normal[1]
                hit_normal[2] = -hit_normal[2]
                # 重新归一化 (如果需要，但翻转不改变长度)
            # --- 结束法线处理 ---

            # --- 计算光照效果 (环境光 + 漫反射) ---
            # 计算法线和太阳光方向的点积 (注意 d_sun_dir 是光照来的方向)
            # 我们需要的是表面接收到的光照，所以用法线和 d_sun_dir 点积
            light_dot_normal = max(0.0, hit_normal[0] * d_sun_dir[0] +
                                      hit_normal[1] * d_sun_dir[1] +
                                      hit_normal[2] * d_sun_dir[2])
            # 计算漫反射强度
            diffuse_intensity = sun_intensity_gpu * light_dot_normal

            # 组合光照：基础颜色 * (环境光 + 漫反射强度)
            shaded_color_float[0] = base_color_float[0] * (ambient_light_gpu + diffuse_intensity)
            shaded_color_float[1] = base_color_float[1] * (ambient_light_gpu + diffuse_intensity)
            shaded_color_float[2] = base_color_float[2] * (ambient_light_gpu + diffuse_intensity)
            # --- 结束光照计算 ---

            # --- 应用雾效 (基于距离 tmin) ---
            # 初始化最终颜色为光照后的颜色
            final_rgb_float[0] = shaded_color_float[0]
            final_rgb_float[1] = shaded_color_float[1]
            final_rgb_float[2] = shaded_color_float[2]
            if fog_enabled_gpu and fog_density_gpu > eps_gpu: # 仅在启用且密度有效时计算
                # 计算雾的衰减因子 (指数衰减)
                # fog_factor 范围 [0, 1]，距离越远，fog_factor 越接近 0
                fog_factor = math.exp(-fog_density_gpu * tmin)
                fog_factor = max(0.0, min(1.0, fog_factor)) # 限制因子范围 [0, 1]

                # 线性混合: 最终颜色 = 雾颜色 * (1 - fog_factor) + 物体颜色 * fog_factor
                final_rgb_float[0] = d_fog_color_float[0] * (1.0 - fog_factor) + shaded_color_float[0] * fog_factor
                final_rgb_float[1] = d_fog_color_float[1] * (1.0 - fog_factor) + shaded_color_float[1] * fog_factor
                final_rgb_float[2] = d_fog_color_float[2] * (1.0 - fog_factor) + shaded_color_float[2] * fog_factor
            # --- 结束雾效 ---

            # --- 最终颜色处理 ---
            # 限制颜色范围 [0, 1]
            final_r_clamped = max(0.0, min(1.0, final_rgb_float[0]))
            final_g_clamped = max(0.0, min(1.0, final_rgb_float[1]))
            final_b_clamped = max(0.0, min(1.0, final_rgb_float[2]))
            # 转换为 uint8 [0, 255] 并写入输出缓冲
            # 注意：Gamma 校正将在保存图像时进行，这里输出线性 RGB
            rgb[i, j, 0] = int(final_r_clamped * 255.0 + 0.5) # 加 0.5 用于四舍五入
            rgb[i, j, 1] = int(final_g_clamped * 255.0 + 0.5)
            rgb[i, j, 2] = int(final_b_clamped * 255.0 + 0.5)
            # --- 结束最终颜色处理 ---

        else:  # 光线未命中 (天空)
            # 直接写入天空颜色
            rgb[i, j, 0] = d_sky_color[0]
            rgb[i, j, 1] = d_sky_color[1]
            rgb[i, j, 2] = d_sky_color[2]
            # 设置其他输出为默认值
            depth[i, j] = inf_gpu
            sem[i, j] = sky_label_gpu
            pts[i, j, 0] = GEOMETRY_DTYPE(np.nan)
            pts[i, j, 1] = GEOMETRY_DTYPE(np.nan)
            pts[i, j, 2] = GEOMETRY_DTYPE(np.nan)


# --- 输出保存函数 ---

# --- [新增] 保存参数 JSON 函数 ---
def save_parameters_json(filename: Path, params: Dict):
    """将相关参数保存到 JSON 文件。"""
    print(f"    将参数保存到 JSON: {filename}")
    try:
        params_serializable = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                params_serializable[key] = value.tolist() # Numpy 数组转列表
            elif isinstance(value, (np.float32, np.float64, np.int32, np.int64, np.bool_)):
                 params_serializable[key] = value.item() # Numpy 标量转 Python 类型
            elif isinstance(value, dict): # 处理嵌套字典
                nested_serializable = {}
                for nk, nv in value.items():
                    if isinstance(nv, np.ndarray):
                        nested_serializable[nk] = nv.tolist()
                    elif isinstance(nv, (np.float32, np.float64, np.int32, np.int64, np.bool_)):
                         nested_serializable[nk] = nv.item()
                    else:
                        nested_serializable[nk] = nv
                params_serializable[key] = nested_serializable
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], np.ndarray):
                 # 处理列表/元组中的 Numpy 数组 (如果需要)
                 params_serializable[key] = [item.tolist() for item in value]
            else:
                # 确保基本类型可序列化
                if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
                     params_serializable[key] = value
                else:
                     # 对未知类型尝试转字符串
                     print(f"    警告: 参数 '{key}' 的类型 {type(value)} 可能无法直接序列化为 JSON，尝试转换为字符串。")
                     params_serializable[key] = str(value)

        with open(filename, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文正常显示
            json.dump(params_serializable, f, indent=4, ensure_ascii=False)
        print(f"    成功将参数保存到 {filename}")
    except TypeError as e:
        print(f"    错误: 无法将参数序列化为 JSON: {e}")
    except IOError as e:
        print(f"    错误: 无法写入 JSON 文件 {filename}: {e}")
    except Exception as e:
        print(f"    保存 JSON 时发生未知错误: {e}")

# --- [新增] 保存 OBJ 函数 ---
def save_combined_obj(filename: Path,
                      v0s: np.ndarray, e1s: np.ndarray, e2s: np.ndarray,
                      normals: np.ndarray, labels: np.ndarray, label_map: Dict,
                      cam_o: np.ndarray, cam_dir: np.ndarray, right: np.ndarray, up: np.ndarray,
                      screen_w: float, screen_h: float,
                      pts: Optional[np.ndarray], # 点云数据是可选的
                      far: float = 1000.0): # 视锥体远平面距离
    """将场景三角形（按标签分组）、法线、相机视锥体和命中点保存到 OBJ 文件。"""
    print(f"    准备保存带图层的 OBJ 文件: {filename}")
    num_triangles = len(v0s)
    if num_triangles == 0:
        print("    跳过 OBJ 保存 (场景中没有三角形)。")
        return

    # 按标签分组三角形索引
    tris_by_label: Dict[int, List[int]] = {}
    for i in range(num_triangles):
        label = labels[i]
        if label == SKY_LABEL: continue # 跳过天空
        if label not in tris_by_label:
            tris_by_label[label] = []
        tris_by_label[label].append(i)

    if not tris_by_label:
        print("    跳过 OBJ 保存 (场景中没有带标签的三角形)。")
        return

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'# Raytracer 输出: {datetime.now()}\n')
            f.write(f'# 生成了 {num_triangles} 个三角形。\n')
            # 尝试获取图层名称
            try:
                layer_names = ", ".join([str(label_map.get(lbl, f"未知_{lbl}")) for lbl in sorted(tris_by_label.keys())])
            except:
                layer_names = "获取图层名称时出错"
            f.write(f'# 语义图层: {layer_names}\n\n')

            vertex_offset = 1 # OBJ 索引从 1 开始
            normal_offset = 1

            print("    按语义图层写入场景几何体...")
            sorted_labels = sorted(tris_by_label.keys())

            all_vertices_str = []
            all_normals_str = []
            face_definitions_data = [] # 存储 (label, v1_idx, n1_idx, v2_idx, n2_idx, v3_idx, n3_idx)

            vertex_map: Dict[Tuple, int] = {} # (x,y,z) -> index
            normal_map: Dict[Tuple, int] = {} # (nx,ny,nz) -> index
            current_v_idx = 1
            current_n_idx = 1

            # 遍历所有有效的三角形，构建顶点、法线列表和面定义
            for tri_idx in range(num_triangles):
                label = labels[tri_idx]
                if label == SKY_LABEL or label not in tris_by_label: continue

                v0 = v0s[tri_idx]
                v1 = v0 + e1s[tri_idx]
                v2 = v0 + e2s[tri_idx]
                normal = normals[tri_idx] # 使用原始法线

                # 处理法线
                normal_tuple = tuple(np.round(normal, 6)) # 使用圆整值作为 key
                if normal_tuple not in normal_map:
                    normal_map[normal_tuple] = current_n_idx
                    all_normals_str.append(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                    n_idx = current_n_idx
                    current_n_idx += 1
                else:
                    n_idx = normal_map[normal_tuple]

                # 处理顶点
                v_indices = []
                for v in [v0, v1, v2]:
                    v_tuple = tuple(np.round(v, 6)) # 使用圆整值作为 key
                    if v_tuple not in vertex_map:
                        vertex_map[v_tuple] = current_v_idx
                        all_vertices_str.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        v_indices.append(current_v_idx)
                        current_v_idx += 1
                    else:
                        v_indices.append(vertex_map[v_tuple])

                # 存储面定义信息 (label, v_indices, n_idx)
                face_definitions_data.append((label, v_indices[0], n_idx, v_indices[1], n_idx, v_indices[2], n_idx))

            # 写入顶点
            f.write("# 顶点\n")
            f.writelines(all_vertices_str)
            print(f"    写入了 {len(all_vertices_str)} 个唯一顶点。")

            # 写入法线
            f.write("\n# 法线\n")
            f.writelines(all_normals_str)
            print(f"    写入了 {len(all_normals_str)} 个唯一法线。")

            # 按图层写入面
            f.write("\n# 按语义图层划分的面\n")
            total_faces_written = 0
            for label in sorted_labels:
                # 获取图层名，处理 key 不存在的情况
                layer_name_val = label_map.get(label, f"未知标签_{label}")
                # 确保名称对 OBJ 文件有效 (替换空格和特殊字符)
                layer_name = str(layer_name_val).replace(" ", "_").replace("/", "-").replace("\\", "-").replace(":", "-")
                f.write(f"\ng {layer_name}\n") # 使用 object group (g) 来定义图层
                f.write(f"usemtl {layer_name}\n") # 可以关联材质（如果需要定义 .mtl 文件）
                f.write("s off\n") # 关闭平滑组（可选）
                count = 0
                # 写入属于当前图层的面
                for face_data in face_definitions_data:
                    face_label, v1, n1, v2, n2, v3, n3 = face_data
                    if face_label == label:
                        # 写入面定义，格式: f v1//vn1 v2//vn2 v3//vn3 (顶点索引//法线索引)
                        f.write(f"f {v1}//{n1} {v2}//{n2} {v3}//{n3}\n")
                        count += 1
                print(f"    为图层 '{layer_name}' (标签 {label}) 写入了 {count} 个面")
                total_faces_written += count
            if total_faces_written == 0:
                 print("    警告: 没有几何面被写入 OBJ 文件。")

            # --- 写入相机视锥体 (可选) ---
            f.write('\n\n# 相机视锥体\n')
            f.write('g camera_frustum\n') # 将视锥体放入单独的组
            # 写入相机原点作为视锥体的顶点
            f.write(f"v {cam_o[0]:.6f} {cam_o[1]:.6f} {cam_o[2]:.6f}\n")
            cam_v_start_idx = current_v_idx # 记录相机原点的顶点索引
            current_v_idx += 1
            corner_indices = [] # 存储远平面角点的索引

            # 计算视锥体远平面的四个角点
            # 屏幕坐标范围: u_norm [-0.5, 0.5], v_norm [-0.5, 0.5]
            for du_norm in [-0.5, 0.5]: # 对应左右
                for dv_norm in [-0.5, 0.5]: # 对应上下 (注意图像 Y 轴向下)
                    # 计算从相机原点指向屏幕角落的方向向量
                    # 注意 v_norm 的符号，因为屏幕坐标 Y 向下，而相机坐标系通常 Y 或 Z 向上
                    d_corner = cam_dir + (du_norm * screen_w * right) - (dv_norm * screen_h * up) # 假设 up 是正确的相机上方向
                    # 归一化方向向量
                    d_corner_norm = np.linalg.norm(d_corner)
                    if d_corner_norm > INTERSECTION_EPSILON:
                         d_corner /= d_corner_norm
                    # 计算远平面上的角点坐标
                    corner_pt = cam_o + d_corner * far
                    # 写入角点顶点
                    f.write(f"v {corner_pt[0]:.6f} {corner_pt[1]:.6f} {corner_pt[2]:.6f}\n")
                    corner_indices.append(current_v_idx)
                    current_v_idx += 1

            # 写入视锥体的线框 (使用 'l' 表示线段)
            if len(corner_indices) == 4:
                 # 假设角点顺序：左下、右下、左上、右上 (或根据 du/dv 顺序确定)
                 # 例如: (-0.5,-0.5), (0.5,-0.5), (-0.5,0.5), (0.5,0.5)
                 bl, br, tl, tr = corner_indices # 需要根据实际循环顺序调整
                 # 连接相机原点和远平面角点
                 f.write(f"l {cam_v_start_idx} {bl}\n")
                 f.write(f"l {cam_v_start_idx} {br}\n")
                 f.write(f"l {cam_v_start_idx} {tl}\n")
                 f.write(f"l {cam_v_start_idx} {tr}\n")
                 # 连接远平面边缘
                 f.write(f"l {bl} {br}\n")
                 f.write(f"l {br} {tr}\n")
                 f.write(f"l {tr} {tl}\n")
                 f.write(f"l {tl} {bl}\n")
            # --- 结束相机视锥体写入 ---

            # --- 写入采样点云 (可选) ---
            if pts is not None:
                f.write('\n\n# 交点采样\n')
                f.write('g intersection_points\n') # 将点放入单独的组
                H_pts, W_pts = pts.shape[:2]
                # 采样步长，避免写入过多点 (例如每隔 32x32 像素采样一个)
                step = max(1, H_pts // 128, W_pts // 128)
                num_pts_written = 0
                pts_indices = [] # 存储写入的点的顶点索引
                for i in range(0, H_pts, step):
                    for j in range(0, W_pts, step):
                        p = pts[i, j]
                        # 只写入有效的命中点 (非 NaN)
                        if not np.isnan(p[0]):
                            # 写入点坐标作为顶点
                            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                            pts_indices.append(current_v_idx)
                            current_v_idx += 1
                            num_pts_written += 1
                # 将所有采样点写入一个 point group (使用 'p' 定义点)
                if num_pts_written > 0:
                     f.write(f"p {' '.join(map(str, pts_indices))}\n")
                print(f"    写入了 {num_pts_written} 个采样命中点。")
            # --- 结束点云写入 ---

        print(f"    成功将带有图层、法线、视锥体和采样点的 OBJ 保存到 {filename}")
    except IOError as e:
        print(f"    错误: 无法写入 OBJ 文件 {filename}: {e}")
    except Exception as e:
        print(f"    保存 OBJ 时发生未知错误: {e}")
        import traceback
        traceback.print_exc()


# --- 其他输出保存函数 ---
def save_image(output_path: Path, image_array: np.ndarray, mode: Optional[str] = None):
    """使用 PIL 将 NumPy 数组保存为图像。为 RGB 图像应用 Gamma 校正。"""
    try:
        # 检查是否为 RGB 图像 (3 通道 uint8) 并且未指定特殊模式
        if len(image_array.shape) == 3 and image_array.shape[2] == 3 and image_array.dtype == COLOR_DTYPE and mode is None:
             # --- 应用 Gamma 校正 ---
             # 1. 将 uint8 [0, 255] 转换为 float32 [0, 1]
             rgb_float_linear = image_array.astype(np.float32) / 255.0
             # 2. 添加一个微小值防止 log(0) 或 pow(0, neg) 错误
             rgb_float_linear = np.maximum(rgb_float_linear, 1e-9)
             # 3. 应用逆 Gamma (从线性空间转换到 sRGB 空间)
             rgb_float_srgb = np.power(rgb_float_linear, INV_GAMMA_OUT)
             # 4. 裁剪到 [0, 1] 范围
             rgb_clipped = np.clip(rgb_float_srgb, 0.0, 1.0)
             # 5. 转换回 uint8 [0, 255]
             image_to_save = (rgb_clipped * 255.0).astype(COLOR_DTYPE)
             img = Image.fromarray(image_to_save)
             print(f"    已应用 Gamma 校正 ({GAMMA_OUT:.1f}) 并保存视图: {output_path}")
        else:
             # 对于非 RGB 或指定了模式的图像，直接保存
             img = Image.fromarray(image_array, mode=mode)
             print(f"    已保存图像 (无 Gamma 校正): {output_path}")
        img.save(output_path)
    except Exception as e:
        print(f"    保存图像时出错 {output_path}: {e}")

def save_depth_visualization(output_path: Path, depth_map: np.ndarray):
    """将深度图保存为可视化的灰度 PNG。"""
    try:
        # 创建有效深度值的掩码 (有限且大于 epsilon)
        valid_depth_mask = np.isfinite(depth_map) & (depth_map > INTERSECTION_EPSILON)

        if np.any(valid_depth_mask):
            valid_depth_values = depth_map[valid_depth_mask]
            # 计算可视化范围：最小值到 99.8 百分位数 (避免极端值影响)
            dmin = np.min(valid_depth_values)
            # dmax = np.max(valid_depth_values) # 或者使用最大值
            dmax_vis = np.percentile(valid_depth_values, 99.8)
            # 计算缩放比例，避免除零
            scale = max(dmax_vis - dmin, INTERSECTION_EPSILON)

            # 归一化深度值到 [0, 1]
            depth_normalized = np.zeros_like(depth_map, dtype=np.float32)
            # 只处理在可视化范围内的有效深度值
            vis_mask = valid_depth_mask & (depth_map <= dmax_vis)
            depth_normalized[vis_mask] = (depth_map[vis_mask] - dmin) / scale

            # 将归一化深度映射到 [0, 254] (保留 255 给无效/无限远区域)
            depth_scaled = np.clip(depth_normalized * 254.0, 0, 254)

            # 创建可视化图像：有效区域使用 scaled 值，无效区域使用 255 (白色)
            dmap_vis = np.where(vis_mask, depth_scaled, 255).astype(np.uint8)

            # 保存为灰度图像
            save_image(output_path, dmap_vis, mode='L')
            print(f"    深度图可视化范围 (映射到 0-254): [{dmin:.2f} - {dmax_vis:.2f}]")
        else:
            # 如果没有有效的深度值，保存一张全黑的图像
            print("    跳过深度图可视化保存 (无有效的有限深度值)。")
            H, W = depth_map.shape
            Image.new('L', (W, H), 0).save(output_path) # 保存全黑图像
    except Exception as e:
        print(f"    保存深度图可视化时出错 {output_path}: {e}")

def save_semantic_visualization(output_path: Path, semantic_map: np.ndarray, color_map: Dict, sky_color: Sequence[int]):
    """使用提供的颜色映射保存语义图。"""
    try:
        H, W = semantic_map.shape
        sem_img = np.zeros((H, W, 3), dtype=COLOR_DTYPE)

        # 1. 初始化为天空颜色
        sky_color_arr = np.array(sky_color[:3], dtype=COLOR_DTYPE) # 取前三个通道
        if sky_color_arr.shape == (3,):
            sem_img[:, :, :] = sky_color_arr
        else:
            print(f"    警告: 无效的天空颜色 {sky_color}，使用默认灰色。")
            sem_img[:, :, :] = 128 # 默认灰色

        # 2. 查找图中存在的非天空标签
        unique_labels = np.unique(semantic_map[semantic_map != SKY_LABEL])

        # 3. 遍历每个标签，并从 color_map 中查找颜色进行填充
        for label_id in unique_labels:
            # 从 color_map 获取颜色 (期望是 numpy uint8 数组)
            mapped_color = color_map.get(label_id)

            if mapped_color is not None and isinstance(mapped_color, np.ndarray) and mapped_color.shape == (3,) and mapped_color.dtype == COLOR_DTYPE:
                # 如果找到有效的颜色，填充对应标签的像素
                mask = (semantic_map == label_id)
                sem_img[mask] = mapped_color
            else:
                # 如果标签没有对应的颜色或颜色无效，打印警告并使用默认颜色（例如紫色）
                print(f"    警告: 语义标签 {label_id} 的颜色在 color_map 中无效或未找到。使用默认紫色。")
                mask = (semantic_map == label_id)
                sem_img[mask] = [255, 0, 255] # 默认紫色

        # 4. 保存图像 (不需要 Gamma 校正)
        save_image(output_path, sem_img, mode='RGB') # 直接指定 RGB 模式
    except Exception as e:
        print(f"    保存语义图可视化时出错 {output_path}: {e}")


# --- 渲染器类 (已修改) ---
class Renderer:
    def __init__(self):
        """ 初始化渲染器，准备存储场景数据和设备指针。"""
        # 场景数据 (CPU) - 用于重建或调试，存储 BVH 排序后的数据
        self.v0s_reordered: Optional[np.ndarray] = None
        self.e1s_reordered: Optional[np.ndarray] = None
        self.e2s_reordered: Optional[np.ndarray] = None
        self.normals_reordered: Optional[np.ndarray] = None
        self.labels_reordered: Optional[np.ndarray] = None
        self.colors_reordered: Optional[np.ndarray] = None
        self.flat_nodes: Optional[np.ndarray] = None # BVH 节点 (CPU)
        self.label_map: Optional[Dict] = None # 标签 ID -> 标签名称
        self.color_map: Optional[Dict] = None # 标签 ID -> 颜色 (numpy uint8 array)
        self.num_triangles: int = 0

        # 场景数据 (GPU 设备指针) - 用于高效渲染
        self.d_v0s: Optional[cuda.devicearray.DeviceNDArray] = None
        self.d_e1s: Optional[cuda.devicearray.DeviceNDArray] = None
        self.d_e2s: Optional[cuda.devicearray.DeviceNDArray] = None
        self.d_normals: Optional[cuda.devicearray.DeviceNDArray] = None
        self.d_labels: Optional[cuda.devicearray.DeviceNDArray] = None
        self.d_colors: Optional[cuda.devicearray.DeviceNDArray] = None
        self.d_flat_nodes: Optional[cuda.devicearray.DeviceNDArray] = None

        # 状态标志
        self._gpu_available = _GPU_AVAILABLE
        self._scene_prepared = False # 场景数据是否已在 CPU 上准备好 (BVH, 重排)
        self._scene_on_gpu = False   # 静态场景数据是否已上传到 GPU

    def prepare_scene(self, flattened_scene_data: Dict, force_rebuild: bool = False):
        """
        准备场景：构建 BVH，重新排序几何体，并将静态数据上传到 GPU。
        只在需要时执行（或强制执行）。

        Args:
            flattened_scene_data: 包含展平几何数据的字典。
                                 期望 'color_map' 的值是 numpy uint8 数组。
            force_rebuild: 是否强制重新构建和上传，即使场景已准备好。

        Returns:
            bool: True 如果场景成功准备并上传到 GPU，否则 False。
        """
        if not self._gpu_available:
            print("错误: GPU 不可用，无法准备场景。")
            return False

        # 如果场景已准备好且未强制重建，则跳过
        if self._scene_prepared and self._scene_on_gpu and not force_rebuild:
            print("[渲染器] 场景已在 GPU 上准备就绪，跳过准备步骤。")
            return True

        print("\n[渲染器] 准备场景数据 (构建BVH, 重排, 上传GPU)...")
        t0_prepare = _now()

        # --- 1. 提取并验证数据 ---
        required_keys = ['v0s', 'e1s', 'e2s', 'normals', 'labels', 'colors', 'label_map', 'color_map']
        if not all(key in flattened_scene_data for key in required_keys):
            missing = [key for key in required_keys if key not in flattened_scene_data]
            print(f"    错误: 准备场景时缺少必要的展平数据键: {missing}")
            self._scene_prepared = False; self._scene_on_gpu = False
            return False

        v0s = flattened_scene_data['v0s']
        e1s = flattened_scene_data['e1s']
        e2s = flattened_scene_data['e2s']
        normals = flattened_scene_data['normals']
        labels = flattened_scene_data['labels']
        colors = flattened_scene_data['colors']
        self.label_map = flattened_scene_data['label_map']
        self.color_map = flattened_scene_data['color_map'] # 直接存储 color_map
        self.num_triangles = len(v0s)

        # 处理空场景
        if self.num_triangles == 0:
            print("    场景中没有三角形。准备完成（空场景）。")
            # 初始化空数组以避免后续错误
            self.v0s_reordered = np.empty((0, 3), dtype=GEOMETRY_DTYPE)
            self.e1s_reordered = np.empty((0, 3), dtype=GEOMETRY_DTYPE)
            self.e2s_reordered = np.empty((0, 3), dtype=GEOMETRY_DTYPE)
            self.normals_reordered = np.empty((0, 3), dtype=GEOMETRY_DTYPE)
            self.labels_reordered = np.empty(0, dtype=LABEL_DTYPE)
            self.colors_reordered = np.empty((0, 3), dtype=COLOR_DTYPE)
            self.flat_nodes = np.empty((0, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE)
            self._scene_prepared = True
            self._scene_on_gpu = False # 空场景不需要 GPU 数据
            log_step('场景准备', t0_prepare)
            return True # 空场景准备成功
        print(f"    接收到 {self.num_triangles} 个三角形。")

        # --- 2. 构建 BVH (在 CPU 上) ---
        print("    构建 BVH...")
        t0_bvh = _now()
        prim_indices = np.arange(self.num_triangles, dtype=INDEX_DTYPE)
        try:
            bvh_nodes, ordered_prim_indices = build_bvh(v0s, e1s, e2s, prim_indices)
        except Exception as e:
            print(f"    错误: BVH 构建过程中发生异常: {e}")
            import traceback
            traceback.print_exc()
            self._scene_prepared = False; self._scene_on_gpu = False
            return False
        log_step('BVH 构建', t0_bvh)

        # 检查 BVH 构建结果
        if bvh_nodes is None or bvh_nodes.shape[0] == 0:
            print("    错误: BVH 构建失败或返回为空。")
            self._scene_prepared = False; self._scene_on_gpu = False
            return False
        self.flat_nodes = bvh_nodes # 存储 CPU 上的 BVH 节点

        # --- 3. 根据 BVH 顺序重排几何数据 (在 CPU 上) ---
        print("    根据 BVH 重新排序几何数据...")
        t0_reorder = _now()
        if len(ordered_prim_indices) != self.num_triangles:
            print(f"    错误: BVH 排序索引数量 ({len(ordered_prim_indices)}) 与三角形数量 ({self.num_triangles}) 不匹配。")
            self._scene_prepared = False; self._scene_on_gpu = False
            return False
        try:
            # 使用高级索引根据 ordered_prim_indices 重新排序所有几何数据
            self.v0s_reordered = v0s[ordered_prim_indices].copy()
            self.e1s_reordered = e1s[ordered_prim_indices].copy()
            self.e2s_reordered = e2s[ordered_prim_indices].copy()
            self.normals_reordered = normals[ordered_prim_indices].copy()
            self.labels_reordered = labels[ordered_prim_indices].copy()
            self.colors_reordered = colors[ordered_prim_indices].copy()
        except IndexError as e:
            print(f"    错误: 几何数据重新排序期间发生索引错误: {e}。请检查原始数据和 BVH 索引。")
            self._scene_prepared = False; self._scene_on_gpu = False
            return False
        except Exception as e:
            print(f"    错误: 几何数据重新排序时发生未知异常: {e}")
            self._scene_prepared = False; self._scene_on_gpu = False
            return False
        log_step('几何数据重新排序', t0_reorder)
        self._scene_prepared = True # CPU 数据准备完成

        # --- 4. 上传静态数据到 GPU ---
        # 如果之前已上传且需要强制重建，先释放旧的 GPU 内存
        if self._scene_on_gpu and force_rebuild:
             self.release_gpu_memory()

        print("    上传静态场景数据到 GPU...")
        t_upload_start = _now()
        try:
            # 将排序后的数据传输到 GPU 设备内存
            self.d_flat_nodes = cuda.to_device(self.flat_nodes)
            self.d_v0s = cuda.to_device(self.v0s_reordered)
            self.d_e1s = cuda.to_device(self.e1s_reordered)
            self.d_e2s = cuda.to_device(self.e2s_reordered)
            self.d_normals = cuda.to_device(self.normals_reordered)
            self.d_labels = cuda.to_device(self.labels_reordered)
            self.d_colors = cuda.to_device(self.colors_reordered)
            self._scene_on_gpu = True # 标记 GPU 数据已就绪
            log_step('GPU 静态数据上传', t_upload_start)
        except Exception as e:
            print(f"    错误: 上传数据到 GPU 失败: {e}")
            self.release_gpu_memory() # 上传失败时尝试清理
            self._scene_on_gpu = False
            return False

        log_step('场景准备总耗时', t0_prepare)
        return True # 所有步骤成功完成

    def release_gpu_memory(self):
        """释放存储在 GPU 上的静态场景数据。"""
        if not self._scene_on_gpu: return
        print("[渲染器] 释放 GPU 静态场景数据显存...")
        # 安全地删除设备数组
        try:
            del self.d_flat_nodes, self.d_v0s, self.d_e1s, self.d_e2s, self.d_normals, self.d_labels, self.d_colors
        except Exception as e:
            print(f"    释放 GPU 显存时发生错误: {e}")
        # 将指针置为 None
        self.d_flat_nodes = self.d_v0s = self.d_e1s = self.d_e2s = self.d_normals = self.d_labels = self.d_colors = None
        self._scene_on_gpu = False # 更新状态

    def render(self, camera_params: Dict, render_params: Dict):
        """
        使用预先准备好的场景数据和当前参数渲染一帧。

        Args:
            camera_params: 包含相机设置的字典。
            render_params: 包含渲染设置（光照、雾效、输出等）的字典。

        Returns:
             包含输出数组 ('rgb', 'depth', 'semantic', 'points') 的字典，
             如果渲染失败则返回 None。
             如果场景为空，则返回包含默认空/天空值的字典。
        """
        total_t0 = _now()
        print(f"\n[渲染器] 开始渲染帧 @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

        # --- [0] 检查状态 ---
        if not self._gpu_available:
            print("错误: GPU 不可用。渲染无法进行。")
            return None
        if not self._scene_prepared:
            print("错误: 场景尚未在 CPU 上准备好。请先调用 prepare_scene()。")
            return None
        # 对于空场景，特殊处理
        if self.num_triangles == 0:
            print("    场景为空，生成默认输出。")
            try:
                W = int(camera_params['width'])
                H = int(camera_params['height'])
                sky_color_list = render_params.get('sky_color', [180, 210, 255])
                sky_color_arr = np.array(sky_color_list, dtype=COLOR_DTYPE)
                rgb_host = np.tile(sky_color_arr, (H, W, 1))
                depth_host = np.full((H, W), np.inf, dtype=DEPTH_DTYPE)
                sem_lbl_host = np.full((H, W), SKY_LABEL, dtype=LABEL_DTYPE)
                pts_host = np.full((H, W, 3), np.nan, dtype=POINT_DTYPE)
                log_step(f'--- 空场景渲染总耗时', total_t0)
                return {'rgb': rgb_host, 'depth': depth_host, 'semantic': sem_lbl_host, 'points': pts_host}
            except Exception as e:
                print(f"    错误: 处理空场景输出时出错: {e}")
                return None

        # 检查 GPU 数据是否就绪 (非空场景)
        if not self._scene_on_gpu:
            print("错误: 场景尚未上传到 GPU。请先调用 prepare_scene()。")
            return None
        if self.d_flat_nodes is None or self.d_v0s is None: # 基本检查
            print("错误: GPU 设备指针无效或未初始化。")
            return None

        # --- [1] 解析参数和设置相机 ---
        print("    解析参数并设置相机...")
        t0_params = _now()
        try:
            # --- 相机参数 ---
            cam_o = np.array(camera_params['origin'], dtype=GEOMETRY_DTYPE)
            cam_t = np.array(camera_params['target'], dtype=GEOMETRY_DTYPE)
            cam_up_vec = np.array(camera_params.get('up_vector', [0.0, 0.0, 1.0]), dtype=GEOMETRY_DTYPE)
            W = int(camera_params['width'])
            H = int(camera_params['height'])
            fov_degrees = float(camera_params['fov_deg'])

            # --- 计算相机坐标系 ---
            # 前向向量 (从原点指向目标)
            cam_dir = cam_t - cam_o
            norm_cam_dir = np.linalg.norm(cam_dir)
            if norm_cam_dir < INTERSECTION_EPSILON:
                raise ValueError("相机原点和目标点太近，无法确定方向")
            cam_dir /= norm_cam_dir # 归一化

            # 右向向量 (前向向量与上向量的叉积)
            right = np.cross(cam_dir, cam_up_vec)
            norm_right = np.linalg.norm(right)
            # 处理前向向量和上向量平行或接近平行的情况
            if norm_right < INTERSECTION_EPSILON:
                print("    警告: 相机前向向量与上向量平行，尝试使用备用右向量。")
                # 尝试使用世界 X 或 Y 轴作为备用上向量来计算右向量
                alt_up = np.array([1.0, 0.0, 0.0], dtype=GEOMETRY_DTYPE) if abs(cam_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=GEOMETRY_DTYPE)
                right = np.cross(cam_dir, alt_up)
                norm_right = np.linalg.norm(right)
                if norm_right < INTERSECTION_EPSILON:
                    raise ValueError("无法计算有效的相机右向量")
            right /= norm_right # 归一化

            # 上向量 (右向向量与前向向量的叉积，确保正交)
            up = np.cross(right, cam_dir)
            # 不需要再次归一化 up，因为 right 和 cam_dir 都是单位向量且正交

            # --- 计算屏幕尺寸 ---
            fov_radians = np.deg2rad(fov_degrees)
            aspect_ratio = float(W) / float(H)
            # screen_h 是相机距离为 1 时，视锥体在垂直方向的高度的一半
            # tan(fov/2) = (screen_h / 2) / 1
            screen_h = GEOMETRY_DTYPE(2.0 * np.tan(fov_radians / 2.0))
            screen_w = GEOMETRY_DTYPE(screen_h * aspect_ratio)

            # --- 渲染参数 ---
            # (移除 shading_mode, specular*, ks, shadow_bias)
            sun_direction_host = np.array(render_params['sun_direction'], dtype=GEOMETRY_DTYPE)
            sun_direction_norm = np.linalg.norm(sun_direction_host)
            if sun_direction_norm < INTERSECTION_EPSILON:
                print("    警告: 太阳方向向量长度接近零，使用默认方向 [0, 0, -1]")
                sun_direction_host = np.array([0.0, 0.0, -1.0], dtype=GEOMETRY_DTYPE)
            else:
                sun_direction_host /= sun_direction_norm # 归一化

            ambient_light = GEOMETRY_DTYPE(render_params['ambient_light'])
            sun_intensity = GEOMETRY_DTYPE(render_params['sun_intensity'])
            sky_color_list = render_params.get('sky_color', [180, 210, 255])
            sky_color_host = np.array(sky_color_list[:3], dtype=COLOR_DTYPE) # 取前三个

            # --- 雾效参数 ---
            fog_enabled = bool(render_params.get('fog_enabled', False))
            fog_color_list = render_params.get('fog_color', [200, 200, 200])
            fog_color_host_uint8 = np.array(fog_color_list[:3], dtype=COLOR_DTYPE) # 取前三个
            # 将雾颜色转换为 [0, 1] float 类型，用于 GPU 计算
            fog_color_host_float = fog_color_host_uint8.astype(GEOMETRY_DTYPE) / 255.0
            fog_density = GEOMETRY_DTYPE(render_params.get('fog_density', 0.0))
            if fog_density < 0:
                 print(f"    警告: 雾密度 ({fog_density}) 为负数，将使用 0.0。")
                 fog_density = GEOMETRY_DTYPE(0.0)

            # --- 输出控制 ---
            save_outputs_raw = render_params.get("save_outputs", ["rgb"])
            # 确保 save_outputs 是一个集合，方便查找
            if isinstance(save_outputs_raw, str):
                save_outputs_set = {save_outputs_raw.lower()}
            elif isinstance(save_outputs_raw, (list, tuple, set)):
                save_outputs_set = {str(s).lower() for s in save_outputs_raw}
            else:
                print(f"    警告: 无效的 save_outputs 类型 ({type(save_outputs_raw)})，将仅保存 RGB。")
                save_outputs_set = {"rgb"}

            # OBJ 保存需要点云数据，即使 "points" 不在列表中也要下载
            needs_pts_download = "points" in save_outputs_set or "obj" in save_outputs_set

            # --- 文件名和路径 ---
            file_prefix = render_params.get('file_prefix', 'render')
            output_dir_str = render_params.get('output_dir', 'output/render_default')
            output_dir = Path(output_dir_str)
            output_dir.mkdir(parents=True, exist_ok=True) # 确保输出目录存在

            log_step('参数解析和相机设置', t0_params)

        except KeyError as e:
            print(f"    错误: 解析参数时缺少键: {e}")
            return None
        except Exception as e:
            print(f"    错误: 解析参数或设置相机时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return None

        # --- [2] GPU Raytrace ---
        print("    开始 GPU 光线追踪...")
        t0_raytrace = _now()
        rgb_host, depth_host, sem_lbl_host, pts_host = None, None, None, None # Host 结果初始化为 None

        # --- 分配 GPU 输出缓冲 ---
        # 这些数组将在 GPU 上被填充
        try:
            d_rgb = cuda.device_array((H, W, 3), dtype=COLOR_DTYPE)
            d_depth = cuda.device_array((H, W), dtype=DEPTH_DTYPE)
            d_sem = cuda.device_array((H, W), dtype=LABEL_DTYPE)
            d_pts = cuda.device_array((H, W, 3), dtype=POINT_DTYPE)
            # 初始化 GPU 缓冲 (可选，但有助于调试)
            d_depth.copy_to_device(np.full((H, W), GPU_INF, dtype=DEPTH_DTYPE))
            d_sem.copy_to_device(np.full((H, W), SKY_LABEL, dtype=LABEL_DTYPE))
            d_pts.copy_to_device(np.full((H, W, 3), np.nan, dtype=POINT_DTYPE))
        except Exception as e:
            print(f"    错误: 分配 GPU 输出缓冲失败: {e}")
            return None

        # --- 上传动态参数到 GPU ---
        # 这些参数每次渲染可能不同 (例如光照方向、雾颜色)
        try:
            d_sun_dir = cuda.to_device(sun_direction_host)
            d_sky_color = cuda.to_device(sky_color_host)
            d_fog_color_float = cuda.to_device(fog_color_host_float)
        except Exception as e:
            print(f"    错误: 上传动态参数到 GPU 失败: {e}")
            # 清理已分配的缓冲
            del d_rgb, d_depth, d_sem, d_pts
            return None

        try:
            # --- 配置并启动内核 ---
            threads_per_block = (16, 16) # 每个块的线程数 (通常 16x16 或 32x32)
            # 计算网格中的块数，确保覆盖所有像素
            blocks_per_grid_x = (W + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid_y = (H + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)

            print(f"    启动 CUDA 内核: 网格大小 {blocks_per_grid}, 块大小 {threads_per_block}")

            # --- 调用 CUDA 内核 ---
            raytrace_cuda_bvh_kernel[blocks_per_grid, threads_per_block](
                # 静态场景数据 (设备指针)
                self.d_flat_nodes, self.d_v0s, self.d_e1s, self.d_e2s,
                self.d_normals, self.d_labels, self.d_colors,
                self.num_triangles,
                # 相机参数 (Host 值)
                cam_o, cam_dir, right, up,
                # 动态渲染参数 (设备指针 / Host 值)
                d_sun_dir, d_sky_color,
                screen_w, screen_h, W, H,
                ambient_light, sun_intensity,
                # 雾效参数
                fog_enabled, d_fog_color_float, fog_density,
                # 输出缓冲 (设备指针)
                d_rgb, d_depth, d_sem, d_pts
            )
            cuda.synchronize() # 等待 GPU 计算完成
            log_step('GPU 光线追踪内核执行', t0_raytrace)

            # --- 从 GPU 下载请求的结果 ---
            print("    从 GPU 下载结果...")
            t0_download = _now()
            if "rgb" in save_outputs_set:
                 rgb_host = d_rgb.copy_to_host()
            if "depth" in save_outputs_set:
                 depth_host = d_depth.copy_to_host()
                 # 将 GPU 的无穷大值转换回 numpy 的无穷大
                 depth_host[depth_host >= GPU_INF] = np.inf
            if "semantic" in save_outputs_set:
                 sem_lbl_host = d_sem.copy_to_host()
            if needs_pts_download: # 如果需要保存 points 或 obj
                 pts_host = d_pts.copy_to_host()
            log_step('GPU 结果下载', t0_download)

        except Exception as e:
            print(f"\n---!! GPU 执行或数据传输错误: {e} ({type(e).__name__}) !!---")
            import traceback
            traceback.print_exc()
            return None # 返回失败
        finally:
            # --- 清理本次渲染分配的 GPU 资源 ---
            # 删除输出缓冲和动态参数的设备数组
            print("    清理 GPU 临时显存...")
            try:
                del d_rgb, d_depth, d_sem, d_pts
                del d_sun_dir, d_sky_color, d_fog_color_float
            except Exception as e:
                print(f"    清理 GPU 临时显存时出错: {e}")

        # --- [3] 保存选定的输出 ---
        print(f'[渲染器] 保存请求的输出...')
        t0_save = _now()
        ts_save = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] # 时间戳到毫秒
        # 更新文件名前缀，移除不再相关的参数
        output_prefix_full = output_dir / f'{file_prefix}_{ts_save}'

        if "rgb" in save_outputs_set and rgb_host is not None:
            save_image(output_prefix_full.with_suffix(".png"), rgb_host)

        if "depth" in save_outputs_set and depth_host is not None:
             save_depth_visualization(output_prefix_full.with_name(f"{output_prefix_full.stem}_depth.png"), depth_host)

        if "semantic" in save_outputs_set and sem_lbl_host is not None:
             if self.color_map and sky_color_list is not None:
                  save_semantic_visualization(output_prefix_full.with_name(f"{output_prefix_full.stem}_semantic.png"), sem_lbl_host, self.color_map, sky_color_list)
             else:
                  print("   警告: 无法保存语义图，缺少 color_map 或 sky_color。")

        if "points" in save_outputs_set and pts_host is not None:
             try:
                  npz_path = output_prefix_full.with_name(f"{output_prefix_full.stem}_points.npz")
                  np.savez_compressed(npz_path, points=pts_host)
                  print(f"    已保存点云: {npz_path}")
             except Exception as e:
                  print(f"    保存点云时出错: {e}")

        # --- 保存 OBJ ---
        if "obj" in save_outputs_set:
             if self.v0s_reordered is not None and pts_host is not None and self.label_map is not None: # 确保数据存在
                  # 计算合适的 far distance 用于视锥体可视化
                  far_dist = 1000.0 # 默认值
                  if depth_host is not None: # 如果下载了深度图
                       valid_depth_mask_obj = np.isfinite(depth_host) & (depth_host > INTERSECTION_EPSILON)
                       if np.any(valid_depth_mask_obj):
                            # 使用最大有效深度的 1.2 倍作为远平面距离
                            far_dist = max(far_dist, np.max(depth_host[valid_depth_mask_obj]) * 1.2)

                  obj_path = output_prefix_full.with_name(f"{output_prefix_full.stem}_scene.obj")
                  save_combined_obj(
                       obj_path,
                       self.v0s_reordered, self.e1s_reordered, self.e2s_reordered, # 使用 CPU 上的重排数据
                       self.normals_reordered, self.labels_reordered, self.label_map,
                       cam_o, cam_dir, right, up, screen_w, screen_h, # 相机参数
                       pts_host, # 下载的点云数据
                       far=far_dist
                  )
             else:
                  print("   警告: 无法保存 OBJ，缺少必要的几何数据、点云数据或标签映射。")

        # --- 保存参数 ---
        if "params" in save_outputs_set:
            current_params = {
                 "时间戳_渲染开始_unix": total_t0,
                 "时间戳_保存_字符串": ts_save,
                 "分辨率": {"宽度": W, "高度": H},
                 "视场角_度": fov_degrees,
                 "相机": {"原点": cam_o.tolist(), "目标": cam_t.tolist(), "上向量": cam_up_vec.tolist()},
                 "光照": {
                      "太阳方向": sun_direction_host.tolist(),
                      "太阳强度": sun_intensity,
                      "环境光强度": ambient_light,
                 },
                 "雾效": {"启用": fog_enabled, "颜色_RGB": fog_color_list, "密度": fog_density},
                 "渲染": {"天空颜色_RGB": sky_color_list},
                 "场景信息": {
                      "三角形数量": self.num_triangles,
                      "BVH节点数量": self.flat_nodes.shape[0] if self.flat_nodes is not None else 0,
                      "标签映射键": list(self.label_map.keys()) if self.label_map else [],
                 },
                 "输出文件前缀": str(output_prefix_full) # 记录实际使用的文件前缀
            }
            params_path = output_prefix_full.with_name(f"{output_prefix_full.stem}_params.json")
            save_parameters_json(params_path, current_params)

        log_step('保存输出', t0_save)
        log_step(f'--- 单帧渲染总耗时', total_t0)

        # 返回下载到 host 的结果字典
        return {
            'rgb': rgb_host,
            'depth': depth_host,
            'semantic': sem_lbl_host,
            'points': pts_host
        }

    def __del__(self):
        """确保在对象销毁时尝试释放 GPU 内存。"""
        if self._scene_on_gpu:
             print("[渲染器] 对象销毁，尝试释放 GPU 显存...")
             try:
                  # 检查属性是否存在且不为 None
                  if hasattr(self, 'd_flat_nodes') and self.d_flat_nodes is not None:
                       self.release_gpu_memory()
             except Exception as e:
                  # 在 __del__ 中忽略错误，避免程序崩溃
                  print(f"    在 __del__ 中释放 GPU 显存时发生错误 (已忽略): {e}")

