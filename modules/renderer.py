# renderer.py
# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import math
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Sequence # Added Sequence
import numpy as np
from PIL import Image
from numba import njit, prange, cuda


# -- GPU Availability Check ---
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
            print("CUDA 可能仍然可用，但信息不完整。")
            _GPU_AVAILABLE = True # 假设仍然可用，但在渲染时会再次检查
    else:
        print("CUDA 不可用: 未检测到兼容设备或驱动程序。渲染将无法进行。")
except ImportError:
    print("Numba CUDA 扩展未安装或导入失败。渲染将无法进行。")
except Exception as e:
    print(f"CUDA 初始化期间发生未知错误: {e}。渲染可能无法进行。")


# --- Constants ---
GEOMETRY_DTYPE = np.float32
INDEX_DTYPE = np.int32
COLOR_DTYPE = np.uint8
LABEL_DTYPE = np.int32
DEPTH_DTYPE = np.float32
POINT_DTYPE = np.float32
BVH_NODE_DTYPE = np.float32
INF = GEOMETRY_DTYPE(np.inf)
GPU_INF = GEOMETRY_DTYPE(1e20) # GPU上的无穷大表示

INTERSECTION_EPSILON = GEOMETRY_DTYPE(1e-5)
# SHADOW_BIAS is now read from render_params

# --- Lighting Defaults Removed ---
# Default values are now expected to be set in the main script's render_params

# --- BVH Constants ---
BVH_MAX_LEAF_SIZE = 4
BVH_NODE_FIELDS = 8 # AABB min (3), AABB max (3), Info1 (1), Info2 (1)

# --- Semantics ---
SKY_LABEL = -1 # Represents sky in semantic map
# DEFAULT_SKY_COLOR is now read from render_params

# --- Gamma -------------------------------------------------
GAMMA_OUT = 2.2               # 输出到 sRGB 的 Gamma
INV_GAMMA_OUT = 1.0 / GAMMA_OUT


# --- Utility Functions ---
def _now() -> float:
    return time.perf_counter()


def log_step(title: str, t0: float) -> None:
    print(f"    {title} 耗时 {_now() - t0:.3f}s")


@njit # Numba JIT for potential CPU usage in helper functions like this
def int32_to_float32_bits(val_int32):
    """ Bit-casts an int32 to float32 bits. """
    int_array = np.array([val_int32], dtype=INDEX_DTYPE)
    return int_array.view(BVH_NODE_DTYPE)[0]


@njit # Numba JIT for potential CPU usage in helper functions like this
def float32_to_int32_bits(val_float32):
    """ Bit-casts a float32 to int32 bits. """
    float_array = np.array([val_float32], dtype=BVH_NODE_DTYPE)
    return float_array.view(INDEX_DTYPE)[0]


# --- BVH Construction Functions (Numba JIT on CPU - Required for GPU) ---
# These remain unchanged as BVH building happens on CPU before GPU transfer
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


def build_bvh(v0s, e1s, e2s, prim_indices_in):
    """构建 BVH 加速结构 (在 CPU 上使用 Numba)。"""
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
        """用于预计算的并行 Numba 函数。"""
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
    recursive_build_numba(0, nodes_used, flat_nodes, ordered_prim_indices,
                          0, N,
                          tri_aabbs_min, tri_aabbs_max, tri_centers)
    log_step('BVH 递归构建', t0_recursive)

    actual_nodes_used = nodes_used[0]
    flat_nodes = flat_nodes[:actual_nodes_used]
    print(f"    BVH 构建完成: 使用了 {actual_nodes_used} 个节点。")

    if actual_nodes_used > 0:
        root_min = flat_nodes[0, 0:3]
        root_max = flat_nodes[0, 3:6]
        print(f"    根节点 AABB Min: [{root_min[0]:.2f}, {root_min[1]:.2f}, {root_min[2]:.2f}], "
              f"Max: [{root_max[0]:.2f}, {root_max[1]:.2f}, {root_max[2]:.2f}]")

    return flat_nodes, ordered_prim_indices


# --- GPU Ray Intersection Functions ---
if _GPU_AVAILABLE: # Only define if GPU is detected initially
    @cuda.jit(device=True, inline=True)
    def ray_tri_intersect_gpu(orig, dir, v0, e1, e2):
        """Moller-Trumbore 光线-三角形相交测试 (GPU 设备函数)。"""
        eps_gpu = INTERSECTION_EPSILON # 使用全局定义的 Epsilon
        inf_gpu = GPU_INF

        h0 = dir[1] * e2[2] - dir[2] * e2[1]
        h1 = dir[2] * e2[0] - dir[0] * e2[2]
        h2 = dir[0] * e2[1] - dir[1] * e2[0]
        a = e1[0] * h0 + e1[1] * h1 + e1[2] * h2

        # 检查是否接近平行或退化三角形
        if abs(a) < eps_gpu: return inf_gpu

        f = GEOMETRY_DTYPE(1.0) / a
        s0 = orig[0] - v0[0]
        s1 = orig[1] - v0[1]
        s2 = orig[2] - v0[2]
        u = f * (s0 * h0 + s1 * h1 + s2 * h2)

        # 检查重心坐标 u 是否在 [0, 1] 范围内
        if u < 0.0 or u > 1.0: return inf_gpu

        q0 = s1 * e1[2] - s2 * e1[1]
        q1 = s2 * e1[0] - s0 * e1[2]
        q2 = s0 * e1[1] - s1 * e1[0]
        v = f * (dir[0] * q0 + dir[1] * q1 + dir[2] * q2)

        # 检查重心坐标 v 和 u+v 是否在 [0, 1] 范围内
        if v < 0.0 or u + v > 1.0: return inf_gpu

        # 计算交点距离 t
        t = f * (e2[0] * q0 + e2[1] * q1 + e2[2] * q2)

        # 只有当 t 大于一个小的正值（避免起点处的自相交）时才认为是有效命中
        return t if t > eps_gpu else inf_gpu # 使用标准 Epsilon 判断

    @cuda.jit(device=True, inline=True)
    def ray_aabb_intersect_gpu(orig, dir_inv, t_min_global, node_aabb_min, node_aabb_max):
        """光线-AABB 相交测试 (Slab test) (GPU 设备函数)。"""
        t_near = -INF # 注意：这里使用全局的 INF
        t_far = INF
        eps_aabb = INTERSECTION_EPSILON # 可以使用与三角形求交相同的 epsilon

        for k in range(3):
            inv_d = dir_inv[k]
            aabb_min_k = node_aabb_min[k]
            aabb_max_k = node_aabb_max[k]
            t1 = (aabb_min_k - orig[k]) * inv_d
            t2 = (aabb_max_k - orig[k]) * inv_d

            if t1 > t2: t1, t2 = t2, t1 # 确保 t1 <= t2

            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

            # 剪枝条件：
            # 1. t_near >= t_far: slabs 没有重叠部分
            # 2. t_far < eps_aabb: 整个相交区间都在光线起点之前或非常接近起点
            # 3. t_near >= t_min_global: AABB 的最近交点比已知的最近物体交点还要远
            if t_near >= t_far or t_far < eps_aabb or t_near >= t_min_global:
                return False # 未命中或被剪枝
        return True # 命中

    @cuda.jit(device=True)
    def trace_shadow_ray_gpu_device(
        shadow_orig, shadow_dir, max_dist, # max_dist 通常是 inf_gpu
        # 使用设备指针传递场景数据
        d_flat_nodes, d_v0s, d_e1s, d_e2s,
        num_tris_total_gpu, # 显式传递三角形总数
        # 不需要 shadow_bias_gpu 参数，因为它仅用于计算 shadow_orig
    ):
        """在 GPU 上使用 BVH 追踪阴影光线。如果被遮挡则返回 True。"""
        num_nodes = d_flat_nodes.shape[0]
        if num_nodes == 0: return False # 空场景

        # 本地变量
        node_stack = cuda.local.array(64, dtype=INDEX_DTYPE)
        dir_inv = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        node_aabb_min_local = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        node_aabb_max_local = cuda.local.array(3, dtype=BVH_NODE_DTYPE)
        inf_gpu = GPU_INF
        eps_gpu = INTERSECTION_EPSILON

        # 计算阴影光线方向的倒数
        for k in range(3):
            sd_k = shadow_dir[k]
            if abs(sd_k) < eps_gpu:
                dir_inv[k] = math.copysign(inf_gpu, sd_k)
            else:
                dir_inv[k] = GEOMETRY_DTYPE(1.0) / sd_k

        # 初始化 BVH 遍历栈
        stack_ptr = 0
        node_stack[stack_ptr] = 0 # 从根节点开始
        stack_ptr += 1

        # BVH 遍历循环
        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = node_stack[stack_ptr]
            if node_idx < 0 or node_idx >= num_nodes: continue # 无效索引检查

            # 获取当前节点的 AABB
            node_aabb_min_local[0] = d_flat_nodes[node_idx, 0]
            node_aabb_min_local[1] = d_flat_nodes[node_idx, 1]
            node_aabb_min_local[2] = d_flat_nodes[node_idx, 2]
            node_aabb_max_local[0] = d_flat_nodes[node_idx, 3]
            node_aabb_max_local[1] = d_flat_nodes[node_idx, 4]
            node_aabb_max_local[2] = d_flat_nodes[node_idx, 5]

            # 检查阴影光线是否与当前节点的 AABB 相交 (注意：t_min_global 设为 max_dist)
            if not ray_aabb_intersect_gpu(shadow_orig, dir_inv, max_dist, node_aabb_min_local, node_aabb_max_local):
                continue # 不相交，跳过此节点

            # 获取节点信息 (使用位转换)
            info_bits_f = d_flat_nodes[node_idx, 7]
            info_val = info_bits_f.view(INDEX_DTYPE)

            if info_val < 0: # 叶子节点
                prim_count = -info_val
                prim_offset_bits_f = d_flat_nodes[node_idx, 6]
                prim_offset = prim_offset_bits_f.view(INDEX_DTYPE)

                # 遍历叶子节点中的所有图元（三角形）
                for p_idx_offset in range(prim_count):
                    current_prim_idx = prim_offset + p_idx_offset
                    # 边界检查，确保索引有效
                    if current_prim_idx < num_tris_total_gpu:
                        # 从设备内存获取三角形数据
                        tri_v0 = d_v0s[current_prim_idx]
                        tri_e1 = d_e1s[current_prim_idx]
                        tri_e2 = d_e2s[current_prim_idx]
                        # 与三角形进行求交测试
                        t = ray_tri_intersect_gpu(shadow_orig, shadow_dir, tri_v0, tri_e1, tri_e2)
                        # 如果找到任何一个比 max_dist 近的交点，则立即返回 True (被遮挡)
                        if t < max_dist:
                            return True # 找到遮挡物
            else: # 内部节点
                left_child_idx_bits_f = d_flat_nodes[node_idx, 6]
                left_child_idx = left_child_idx_bits_f.view(INDEX_DTYPE)
                right_child_idx = info_val # 右子节点索引存储在 info_val 中

                # 将子节点压入栈中继续遍历 (检查栈空间)
                if stack_ptr + 2 <= 64:
                    # 检查子节点索引是否有效
                    if right_child_idx >= 0 and right_child_idx < num_nodes:
                        node_stack[stack_ptr] = right_child_idx
                        stack_ptr += 1
                    if left_child_idx >= 0 and left_child_idx < num_nodes:
                        node_stack[stack_ptr] = left_child_idx
                        stack_ptr += 1
                # else: 栈溢出 (不太可能发生)

        # 如果遍历完整个 BVH 都没有找到任何交点，则返回 False (未被遮挡)
        return False

# --- GPU Raytracing Kernel (Refactored for Class Structure & Shading Modes) ---
if _GPU_AVAILABLE:
    @cuda.jit
    def raytrace_cuda_bvh_kernel(
            # --- Scene Data (Device Pointers) ---
            # 从 Renderer 类的实例中传递
            d_flat_nodes, d_v0s, d_e1s, d_e2s,
            d_normals, d_labels, d_colors,
            num_tris_total_gpu, # 显式传递三角形总数
            # --- Camera Parameters (Host Values) ---
            cam_o, cam_dir, right, up,
            # --- Rendering Parameters (Host Values / Device Pointers) ---
            d_sun_dir,          # 动态上传的设备数组 (vec3)
            d_specular_color,   # 动态上传的设备数组 (vec3)
            d_sky_color,        # 动态上传的设备数组 (vec3 uint8)
            scr_w, scr_h, W, H,   # Host floats/ints
            ambient_light_gpu,    # Host float
            sun_intensity_gpu,    # Host float
            specular_exponent_gpu,# Host float
            ks_gpu,               # Host float
            shadow_bias_gpu,      # Host float
            # --- Fog Parameters ---
            fog_enabled_gpu,      # Host bool
            d_fog_color_float,  # 动态上传的设备数组 (vec3 float [0,1])
            fog_density_gpu,      # Host float
            # --- NEW: Shading Mode ---
            shading_mode_int, # 0: Phong (完整), 1: Simple Diffuse (简化)
            # --- Output Buffers (Device Pointers) ---
            # 在 render 方法中分配
            rgb, depth, sem, pts
    ):
        """GPU 光线追踪内核 (BVH, 可选着色模式)。"""
        i, j = cuda.grid(2) # 像素坐标 (行 i, 列 j)
        if i >= H or j >= W: return # 边界检查

        # --- 常量和本地变量 ---
        inf_gpu = GPU_INF
        eps_gpu = INTERSECTION_EPSILON
        sky_label_gpu = SKY_LABEL
        MODE_PHONG = 0
        MODE_SIMPLE_DIFFUSE = 1

        # 每个线程的本地数组，用于计算
        d = cuda.local.array(3, dtype=GEOMETRY_DTYPE)       # 光线方向
        dir_inv = cuda.local.array(3, dtype=GEOMETRY_DTYPE) # 光线方向的倒数
        node_aabb_min = cuda.local.array(3, dtype=BVH_NODE_DTYPE) # 节点 AABB 最小值
        node_aabb_max = cuda.local.array(3, dtype=BVH_NODE_DTYPE) # 节点 AABB 最大值
        hit_normal = cuda.local.array(3, dtype=GEOMETRY_DTYPE)    # 命中点的法线 (面向相机)
        hit_point = cuda.local.array(3, dtype=GEOMETRY_DTYPE)     # 命中点坐标
        base_color_float = cuda.local.array(3, dtype=GEOMETRY_DTYPE) # 物体基础颜色 [0,1]
        shadow_ray_origin = cuda.local.array(3, dtype=GEOMETRY_DTYPE) # 阴影光线起点
        view_dir = cuda.local.array(3, dtype=GEOMETRY_DTYPE)      # 视线方向 (从命中点到相机)
        reflect_dir = cuda.local.array(3, dtype=GEOMETRY_DTYPE)   # 光线反射方向
        shaded_color_float = cuda.local.array(3, dtype=GEOMETRY_DTYPE) # 光照计算后的颜色 (线性空间)

        num_nodes = d_flat_nodes.shape[0] # 从设备数组获取节点数

        # 1. 计算主光线方向 d (与之前相同)
        u_norm = (GEOMETRY_DTYPE(j) + 0.5) / W - 0.5
        v_norm = (GEOMETRY_DTYPE(i) + 0.5) / H - 0.5 # Y 轴向下
        d[0] = cam_dir[0] + u_norm * scr_w * right[0] - v_norm * scr_h * up[0]
        d[1] = cam_dir[1] + u_norm * scr_w * right[1] - v_norm * scr_h * up[1]
        d[2] = cam_dir[2] + u_norm * scr_w * right[2] - v_norm * scr_h * up[2]
        # 归一化 d
        nrm_sq = d[0]**2 + d[1]**2 + d[2]**2
        if nrm_sq < eps_gpu**2: # 处理无效光线
             rgb[i, j, 0] = d_sky_color[0]; rgb[i, j, 1] = d_sky_color[1]; rgb[i, j, 2] = d_sky_color[2]
             depth[i, j] = inf_gpu; sem[i, j] = sky_label_gpu
             pts[i, j, 0] = GEOMETRY_DTYPE(np.nan); pts[i, j, 1] = GEOMETRY_DTYPE(np.nan); pts[i, j, 2] = GEOMETRY_DTYPE(np.nan)
             return
        inv_nrm = GEOMETRY_DTYPE(1.0) / math.sqrt(nrm_sq)
        d[0] *= inv_nrm; d[1] *= inv_nrm; d[2] *= inv_nrm
        # 预计算方向倒数
        for k in range(3):
            if abs(d[k]) < eps_gpu: dir_inv[k] = math.copysign(inf_gpu, d[k])
            else: dir_inv[k] = GEOMETRY_DTYPE(1.0) / d[k]

        # 2. 初始化主光线遍历 (与之前相同)
        tmin = inf_gpu
        hit_prim_idx = -1

        # 处理空场景 (与之前相同)
        if num_nodes == 0:
            rgb[i, j, 0] = d_sky_color[0]; rgb[i, j, 1] = d_sky_color[1]; rgb[i, j, 2] = d_sky_color[2]
            depth[i, j] = inf_gpu; sem[i, j] = sky_label_gpu
            pts[i, j, 0] = GEOMETRY_DTYPE(np.nan); pts[i, j, 1] = GEOMETRY_DTYPE(np.nan); pts[i, j, 2] = GEOMETRY_DTYPE(np.nan)
            return

        # --- BVH 遍历栈 (本地) --- (与之前相同)
        BVH_GPU_STACK_SIZE = 64
        node_stack = cuda.local.array(BVH_GPU_STACK_SIZE, dtype=INDEX_DTYPE)
        stack_ptr = 0
        node_stack[stack_ptr] = 0
        stack_ptr += 1

        # 3. BVH 遍历主光线 (与之前相同, 注意使用设备指针 d_flat_nodes, d_v0s 等)
        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = node_stack[stack_ptr]
            if node_idx < 0 or node_idx >= num_nodes: continue

            # 获取 AABB
            node_aabb_min[0] = d_flat_nodes[node_idx, 0]; node_aabb_min[1] = d_flat_nodes[node_idx, 1]; node_aabb_min[2] = d_flat_nodes[node_idx, 2]
            node_aabb_max[0] = d_flat_nodes[node_idx, 3]; node_aabb_max[1] = d_flat_nodes[node_idx, 4]; node_aabb_max[2] = d_flat_nodes[node_idx, 5]

            # AABB 求交测试
            if not ray_aabb_intersect_gpu(cam_o, dir_inv, tmin, node_aabb_min, node_aabb_max):
                continue

            # 处理节点 (叶子或内部)
            info_bits_f = d_flat_nodes[node_idx, 7]
            info_val = info_bits_f.view(INDEX_DTYPE)

            if info_val < 0: # 叶子节点
                prim_count = -info_val
                prim_offset_bits_f = d_flat_nodes[node_idx, 6]
                prim_offset = prim_offset_bits_f.view(INDEX_DTYPE)
                for p_idx_offset in range(prim_count):
                    current_prim_idx = prim_offset + p_idx_offset
                    if current_prim_idx < num_tris_total_gpu:
                        # 使用设备指针获取三角形数据
                        tri_v0 = d_v0s[current_prim_idx]
                        tri_e1 = d_e1s[current_prim_idx]
                        tri_e2 = d_e2s[current_prim_idx]
                        t = ray_tri_intersect_gpu(cam_o, d, tri_v0, tri_e1, tri_e2)
                        if t < tmin: # 找到更近的命中
                            tmin = t
                            hit_prim_idx = current_prim_idx
                            # 注意：这里不再需要读取法线到本地 hit_normal，在循环结束后统一读取最终命中的法线
            else: # 内部节点
                left_child_idx_bits_f = d_flat_nodes[node_idx, 6]
                left_child_idx = left_child_idx_bits_f.view(INDEX_DTYPE)
                right_child_idx = info_val
                if stack_ptr + 2 <= BVH_GPU_STACK_SIZE:
                    if right_child_idx >= 0 and right_child_idx < num_nodes:
                        node_stack[stack_ptr] = right_child_idx; stack_ptr += 1
                    if left_child_idx >= 0 and left_child_idx < num_nodes:
                        node_stack[stack_ptr] = left_child_idx; stack_ptr += 1

        # 4. 处理命中结果
        if hit_prim_idx >= 0:  # 光线命中物体
            # --- 存储主要输出 (深度, 语义, 命中点) ---
            depth[i, j] = tmin
            sem[i, j] = d_labels[hit_prim_idx] # 从设备数组读取标签
            hit_point[0] = cam_o[0] + d[0] * tmin; hit_point[1] = cam_o[1] + d[1] * tmin; hit_point[2] = cam_o[2] + d[2] * tmin
            pts[i, j, 0] = hit_point[0]; pts[i, j, 1] = hit_point[1]; pts[i, j, 2] = hit_point[2]

            # --- 获取基础颜色 ---
            color_uint8 = d_colors[hit_prim_idx] # 从设备数组读取颜色
            base_color_float[0] = GEOMETRY_DTYPE(color_uint8[0]) / 255.0
            base_color_float[1] = GEOMETRY_DTYPE(color_uint8[1]) / 255.0
            base_color_float[2] = GEOMETRY_DTYPE(color_uint8[2]) / 255.0

            # --- 法线处理 (修正后) ---
            # 从设备数组读取最终命中的法线到本地 hit_normal
            hit_normal[0] = d_normals[hit_prim_idx, 0]
            hit_normal[1] = d_normals[hit_prim_idx, 1]
            hit_normal[2] = d_normals[hit_prim_idx, 2]
            # 检查并翻转法线，使其指向相机
            normal_dot_ray = hit_normal[0] * d[0] + hit_normal[1] * d[1] + hit_normal[2] * d[2]
            if normal_dot_ray > 0.0:
                hit_normal[0] = -hit_normal[0]
                hit_normal[1] = -hit_normal[1]
                hit_normal[2] = -hit_normal[2]
            # --- 结束法线处理 ---

            # --- 根据着色模式选择计算路径 ---
            if shading_mode_int == MODE_PHONG:
                # --- 完整 Phong 着色 (带阴影) ---

                # --- 阴影计算 ---
                shadow_ray_origin[0] = hit_point[0] + hit_normal[0] * shadow_bias_gpu
                shadow_ray_origin[1] = hit_point[1] + hit_normal[1] * shadow_bias_gpu
                shadow_ray_origin[2] = hit_point[2] + hit_normal[2] * shadow_bias_gpu
                is_occluded = trace_shadow_ray_gpu_device(
                    shadow_ray_origin, d_sun_dir, inf_gpu,
                    d_flat_nodes, d_v0s, d_e1s, d_e2s, # 传递设备指针
                    num_tris_total_gpu, # 传递三角形总数
                    # shadow_bias_gpu # 不需要传递 bias 给 tracer
                )
                shadow_factor = GEOMETRY_DTYPE(0.0) if is_occluded else GEOMETRY_DTYPE(1.0)
                # --- 结束阴影计算 ---

                # --- 漫反射计算 ---
                light_dot_normal = max(0.0, hit_normal[0] * d_sun_dir[0] +
                                       hit_normal[1] * d_sun_dir[1] +
                                       hit_normal[2] * d_sun_dir[2])
                diffuse_intensity = sun_intensity_gpu * light_dot_normal
                # --- 结束漫反射计算 ---

                # --- 高光计算 (Phong) ---
                # 计算视线向量 V
                view_dir[0] = cam_o[0] - hit_point[0]; view_dir[1] = cam_o[1] - hit_point[1]; view_dir[2] = cam_o[2] - hit_point[2]
                view_dir_norm_sq = view_dir[0] ** 2 + view_dir[1] ** 2 + view_dir[2] ** 2
                if view_dir_norm_sq > eps_gpu ** 2:
                    inv_view_norm = GEOMETRY_DTYPE(1.0) / math.sqrt(view_dir_norm_sq)
                    view_dir[0] *= inv_view_norm; view_dir[1] *= inv_view_norm; view_dir[2] *= inv_view_norm
                # 计算反射向量 R
                n_dot_l_raw = (hit_normal[0] * d_sun_dir[0] + hit_normal[1] * d_sun_dir[1] + hit_normal[2] * d_sun_dir[2])
                reflect_dot = 2.0 * n_dot_l_raw
                reflect_dir[0] = reflect_dot * hit_normal[0] - d_sun_dir[0]
                reflect_dir[1] = reflect_dot * hit_normal[1] - d_sun_dir[1]
                reflect_dir[2] = reflect_dot * hit_normal[2] - d_sun_dir[2]
                # 归一化 R
                reflect_norm_sq = reflect_dir[0] ** 2 + reflect_dir[1] ** 2 + reflect_dir[2] ** 2
                if reflect_norm_sq > eps_gpu ** 2:
                    inv_reflect_norm = GEOMETRY_DTYPE(1.0) / math.sqrt(reflect_norm_sq)
                    reflect_dir[0] *= inv_reflect_norm; reflect_dir[1] *= inv_reflect_norm; reflect_dir[2] *= inv_reflect_norm
                # 计算高光强度
                specular_dot_view = max(0.0, view_dir[0] * reflect_dir[0] + view_dir[1] * reflect_dir[1] + view_dir[2] * reflect_dir[2])
                specular_intensity = ks_gpu * (specular_dot_view ** specular_exponent_gpu)
                # --- 结束高光计算 ---

                # --- 组合光照分量 (Phong) ---
                shaded_color_float[0] = base_color_float[0] * (ambient_light_gpu + diffuse_intensity * shadow_factor) + d_specular_color[0] * specular_intensity * shadow_factor
                shaded_color_float[1] = base_color_float[1] * (ambient_light_gpu + diffuse_intensity * shadow_factor) + d_specular_color[1] * specular_intensity * shadow_factor
                shaded_color_float[2] = base_color_float[2] * (ambient_light_gpu + diffuse_intensity * shadow_factor) + d_specular_color[2] * specular_intensity * shadow_factor
                # --- 结束组合光照 (Phong) ---

            elif shading_mode_int == MODE_SIMPLE_DIFFUSE:
                # --- 简化着色 (仅环境光 + 漫反射, 无阴影/高光) ---

                # --- 简化漫反射计算 ---
                light_dot_normal = max(0.0, hit_normal[0] * d_sun_dir[0] +
                                       hit_normal[1] * d_sun_dir[1] +
                                       hit_normal[2] * d_sun_dir[2])
                diffuse_factor = light_dot_normal # 仅计算点积因子
                # --- 结束简化漫反射计算 ---

                # --- 组合简化光照分量 ---
                shaded_color_float[0] = base_color_float[0] * (ambient_light_gpu + sun_intensity_gpu * diffuse_factor)
                shaded_color_float[1] = base_color_float[1] * (ambient_light_gpu + sun_intensity_gpu * diffuse_factor)
                shaded_color_float[2] = base_color_float[2] * (ambient_light_gpu + sun_intensity_gpu * diffuse_factor)
                # --- 结束组合简化光照 ---

            else: # 其他模式或默认回退 (例如仅环境光)
                shaded_color_float[0] = base_color_float[0] * ambient_light_gpu
                shaded_color_float[1] = base_color_float[1] * ambient_light_gpu
                shaded_color_float[2] = base_color_float[2] * ambient_light_gpu

            # --- 应用雾效 (所有着色模式通用) ---
            final_rgb_float_r = shaded_color_float[0]
            final_rgb_float_g = shaded_color_float[1]
            final_rgb_float_b = shaded_color_float[2]
            if fog_enabled_gpu:
                fog_factor = math.exp(-fog_density_gpu * tmin)
                fog_factor = max(0.0, min(1.0, fog_factor)) # 限制因子范围
                final_rgb_float_r = d_fog_color_float[0] * (1.0 - fog_factor) + shaded_color_float[0] * fog_factor
                final_rgb_float_g = d_fog_color_float[1] * (1.0 - fog_factor) + shaded_color_float[1] * fog_factor
                final_rgb_float_b = d_fog_color_float[2] * (1.0 - fog_factor) + shaded_color_float[2] * fog_factor
            # --- 结束雾效 ---

            # --- 最终颜色处理 (所有着色模式通用) ---
            # 限制颜色范围 [0, 1]
            final_r_clamped = max(0.0, min(1.0, final_rgb_float_r))
            final_g_clamped = max(0.0, min(1.0, final_rgb_float_g))
            final_b_clamped = max(0.0, min(1.0, final_rgb_float_b))
            # 转换为 uint8 [0, 255] 并写入输出缓冲
            rgb[i, j, 0] = int(final_r_clamped * 255.0)
            rgb[i, j, 1] = int(final_g_clamped * 255.0)
            rgb[i, j, 2] = int(final_b_clamped * 255.0)
            # --- 结束最终颜色处理 ---

        else:  # 光线未命中 (天空)
            rgb[i, j, 0] = d_sky_color[0]
            rgb[i, j, 1] = d_sky_color[1]
            rgb[i, j, 2] = d_sky_color[2]
            depth[i, j] = inf_gpu
            sem[i, j] = sky_label_gpu
            pts[i, j, 0] = GEOMETRY_DTYPE(np.nan)
            pts[i, j, 1] = GEOMETRY_DTYPE(np.nan)
            pts[i, j, 2] = GEOMETRY_DTYPE(np.nan)


# --- 输出保存函数 (修改后支持选择性保存和 Gamma 校正) ---

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
            json.dump(params_serializable, f, indent=4, ensure_ascii=False) # ensure_ascii=False 保证中文正常显示
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
            f.write(f'# Raytracer output: {datetime.now()}\n')
            f.write(f'# Generated with {num_triangles} triangles.\n')
            # 尝试获取图层名称
            try: layer_names = ", ".join([str(label_map.get(lbl, f"unknown_{lbl}")) for lbl in sorted(tris_by_label.keys())])
            except: layer_names = "Error retrieving layer names"
            f.write(f'# Semantic Layers: {layer_names}\n\n')

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
            f.write("# Vertices\n")
            f.writelines(all_vertices_str)
            print(f"    写入了 {len(all_vertices_str)} 个唯一顶点。")

            # 写入法线
            f.write("\n# Normals\n")
            f.writelines(all_normals_str)
            print(f"    写入了 {len(all_normals_str)} 个唯一法线。")

            # 按图层写入面
            f.write("\n# Faces by Semantic Layer\n")
            total_faces_written = 0
            for label in sorted_labels:
                # 获取图层名，处理 key 不存在的情况
                layer_name_val = label_map.get(label, f"unknown_label_{label}")
                layer_name = str(layer_name_val).replace(" ", "_").replace("/", "-") # 确保名称有效
                f.write(f"\ng {layer_name}\n") # 使用 object group
                f.write(f"usemtl {layer_name}\n") # 可以关联材质（如果需要定义 .mtl 文件）
                f.write("s off\n") # 关闭平滑组（可选）
                count = 0
                # 写入属于当前图层的面
                for face_data in face_definitions_data:
                    face_label, v1, n1, v2, n2, v3, n3 = face_data
                    if face_label == label:
                        f.write(f"f {v1}//{n1} {v2}//{n2} {v3}//{n3}\n")
                        count += 1
                print(f"    为图层 '{layer_name}' (标签 {label}) 写入了 {count} 个面")
                total_faces_written += count
            if total_faces_written == 0:
                 print("    警告: 没有几何面被写入 OBJ 文件。")

            # 写入相机视锥体 (可选)
            f.write('\n\n# Camera Frustum\n')
            f.write('g camera_frustum\n')
            f.write(f"v {cam_o[0]:.6f} {cam_o[1]:.6f} {cam_o[2]:.6f}\n") # 相机原点
            cam_v_start_idx = current_v_idx
            current_v_idx += 1
            corner_indices = []
            # 计算视锥体远平面的四个角点
            for du_norm in [-0.5, 0.5]:
                for dv_norm in [-0.5, 0.5]: # 注意 Y 轴向下，所以 dv_norm 符号可能需要调整，取决于 up 向量
                    # 屏幕坐标到方向向量
                    d_corner = cam_dir + (du_norm * screen_w * right) - (dv_norm * screen_h * up) # 假设 up 是正确的相机上方向
                    d_corner_norm = np.linalg.norm(d_corner)
                    if d_corner_norm > INTERSECTION_EPSILON:
                         d_corner /= d_corner_norm # 归一化方向
                    # 计算远平面上的点
                    corner_pt = cam_o + d_corner * far
                    f.write(f"v {corner_pt[0]:.6f} {corner_pt[1]:.6f} {corner_pt[2]:.6f}\n")
                    corner_indices.append(current_v_idx)
                    current_v_idx += 1
            # 写入视锥体的线框
            if len(corner_indices) == 4:
                 bl, tl, br, tr = corner_indices # 假设顺序：左下、左上、右下、右上
                 f.write(f"l {cam_v_start_idx} {bl}\n") # 连接原点和角点
                 f.write(f"l {cam_v_start_idx} {tl}\n")
                 f.write(f"l {cam_v_start_idx} {br}\n")
                 f.write(f"l {cam_v_start_idx} {tr}\n")
                 f.write(f"l {bl} {tl}\n") # 连接远平面边缘
                 f.write(f"l {tl} {tr}\n")
                 f.write(f"l {tr} {br}\n")
                 f.write(f"l {br} {bl}\n")

            # 写入采样点云 (可选)
            if pts is not None:
                f.write('\n\n# Intersection Points (Sampled)\n')
                f.write('g intersection_points\n')
                H_pts, W_pts = pts.shape[:2]
                # 采样步长，避免写入过多点
                step = max(1, H_pts // 64, W_pts // 64) # 每 64x64 采样一个点
                num_pts_written = 0
                pts_indices = []
                for i in range(0, H_pts, step):
                    for j in range(0, W_pts, step):
                        p = pts[i, j]
                        # 只写入有效的命中点
                        if not np.isnan(p[0]):
                            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                            pts_indices.append(current_v_idx)
                            current_v_idx += 1
                            num_pts_written += 1
                # 将所有点写入一个 point group
                if num_pts_written > 0:
                     f.write(f"p {' '.join(map(str, pts_indices))}\n")
                print(f"    写入了 {num_pts_written} 个采样命中点。")

        print(f"    成功将带有图层、法线、视锥体和采样点的 OBJ 保存到 {filename}")
    except IOError as e:
        print(f"    错误: 无法写入 OBJ 文件 {filename}: {e}")
    except Exception as e:
        print(f"    保存 OBJ 时发生未知错误: {e}")
        import traceback
        traceback.print_exc()


# --- 其他输出保存函数 (与之前相同) ---
def save_image(output_path: Path, image_array: np.ndarray, mode: Optional[str] = None):
    """使用 PIL 将 NumPy 数组保存为图像。为 RGB 图像应用 Gamma 校正。"""
    try:
        if len(image_array.shape) == 3 and image_array.shape[2] == 3 and mode is None:
             rgb_float_linear = image_array.astype(np.float32) / 255.0
             rgb_float_linear = np.maximum(rgb_float_linear, 1e-9)
             rgb_float_srgb = np.power(rgb_float_linear, INV_GAMMA_OUT)
             image_to_save = (np.clip(rgb_float_srgb, 0.0, 1.0) * 255.0).astype(COLOR_DTYPE)
             img = Image.fromarray(image_to_save)
             print(f"    已应用 Gamma 校正 ({GAMMA_OUT}) 并保存视图: {output_path}")
        else:
             img = Image.fromarray(image_array, mode=mode)
             print(f"    已保存图像: {output_path}")
        img.save(output_path)
    except Exception as e:
        print(f"    保存图像时出错 {output_path}: {e}")

def save_depth_visualization(output_path: Path, depth_map: np.ndarray):
    """将深度图保存为可视化的灰度 PNG。"""
    try:
        valid_depth_mask = np.isfinite(depth_map) & (depth_map > INTERSECTION_EPSILON)
        if np.any(valid_depth_mask):
            valid_depth_values = depth_map[valid_depth_mask]
            dmin = np.min(valid_depth_values)
            dmax_vis = np.percentile(valid_depth_values, 99.8)
            scale = max(dmax_vis - dmin, INTERSECTION_EPSILON)
            depth_normalized = np.zeros_like(depth_map)
            vis_mask = valid_depth_mask & (depth_map <= dmax_vis)
            depth_normalized[vis_mask] = (depth_map[vis_mask] - dmin) / scale
            depth_scaled = np.clip(depth_normalized * 254.0, 0, 254)
            dmap_vis = np.where(vis_mask, depth_scaled, 255).astype(np.uint8)
            save_image(output_path, dmap_vis, mode='L')
            print(f"    深度图可视化范围: [{dmin:.2f} - {dmax_vis:.2f}]")
        else:
            print("    跳过深度图保存 (无有效的有限深度值)。")
            H, W = depth_map.shape
            Image.new('L', (W, H), 0).save(output_path)
    except Exception as e:
        print(f"    保存深度图像时出错 {output_path}: {e}")

def save_semantic_visualization(output_path: Path, semantic_map: np.ndarray, color_map: Dict, sky_color: Sequence[int]):
    """使用提供的颜色映射保存语义图。"""
    try:
        H, W = semantic_map.shape
        sem_img = np.zeros((H, W, 3), dtype=COLOR_DTYPE)
        sky_color_arr = np.array(sky_color, dtype=COLOR_DTYPE)
        if sky_color_arr.shape == (3,): sem_img[:, :, :] = sky_color_arr
        else: sem_img[:, :, :] = 128
        unique_labels = np.unique(semantic_map[semantic_map != SKY_LABEL])
        for label_id in unique_labels:
            mapped_color = color_map.get(label_id)
            if mapped_color is not None and isinstance(mapped_color, np.ndarray) and mapped_color.shape == (3,) and mapped_color.dtype == COLOR_DTYPE:
                mask = (semantic_map == label_id)
                sem_img[mask] = mapped_color
            else:
                print(f"    警告: 语义标签 {label_id} 颜色无效。使用默认紫色。")
                mask = (semantic_map == label_id)
                sem_img[mask] = [255, 0, 255]
        save_image(output_path, sem_img)
    except Exception as e:
        print(f"    保存语义图像时出错 {output_path}: {e}")


# --- Renderer Class (Refactored) ---
class Renderer:
    def __init__(self):
        """ 初始化渲染器，准备存储场景数据和设备指针。"""
        # 场景数据 (CPU) - 用于重建或调试
        self.v0s_reordered: Optional[np.ndarray] = None
        self.e1s_reordered: Optional[np.ndarray] = None
        self.e2s_reordered: Optional[np.ndarray] = None
        self.normals_reordered: Optional[np.ndarray] = None
        self.labels_reordered: Optional[np.ndarray] = None
        self.colors_reordered: Optional[np.ndarray] = None
        self.flat_nodes: Optional[np.ndarray] = None
        self.label_map: Optional[Dict] = None
        self.color_map: Optional[Dict] = None # 存储处理后的 color map (value 是 ndarray)
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
        """
        if not self._gpu_available:
            print("错误: GPU 不可用，无法准备场景。")
            return False

        if self._scene_prepared and self._scene_on_gpu and not force_rebuild:
            print("[Renderer] 场景已在 GPU 上准备就绪，跳过准备步骤。")
            return True

        print("\n[Renderer] 准备场景数据 (构建BVH, 重排, 上传GPU)...")
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
        self.color_map = flattened_scene_data['color_map']
        self.num_triangles = len(v0s)

        if self.num_triangles == 0:
            print("    场景中没有三角形。准备完成（空场景）。")
            self.v0s_reordered = np.empty((0, 3), dtype=GEOMETRY_DTYPE)
            self.e1s_reordered = np.empty((0, 3), dtype=GEOMETRY_DTYPE)
            self.e2s_reordered = np.empty((0, 3), dtype=GEOMETRY_DTYPE)
            self.normals_reordered = np.empty((0, 3), dtype=GEOMETRY_DTYPE)
            self.labels_reordered = np.empty(0, dtype=LABEL_DTYPE)
            self.colors_reordered = np.empty((0, 3), dtype=COLOR_DTYPE)
            self.flat_nodes = np.empty((0, BVH_NODE_FIELDS), dtype=BVH_NODE_DTYPE)
            self._scene_prepared = True
            self._scene_on_gpu = False
            log_step('场景准备', t0_prepare)
            return True
        print(f"    接收到 {self.num_triangles} 个三角形。")

        # --- 2. 构建 BVH (在 CPU 上) ---
        print("    构建 BVH...")
        t0_bvh = _now()
        prim_indices = np.arange(self.num_triangles, dtype=INDEX_DTYPE)
        bvh_nodes, ordered_prim_indices = build_bvh(v0s, e1s, e2s, prim_indices)
        log_step('BVH 构建', t0_bvh)
        if bvh_nodes is None or (bvh_nodes.shape[0] == 0):
            print("    错误: BVH 构建失败或为空。")
            self._scene_prepared = False; self._scene_on_gpu = False
            return False
        self.flat_nodes = bvh_nodes

        # --- 3. 根据 BVH 顺序重排几何数据 (在 CPU 上) ---
        print("    根据 BVH 重新排序几何数据...")
        t0_reorder = _now()
        if len(ordered_prim_indices) != self.num_triangles:
            print(f"    错误: BVH 排序索引计数与三角形计数不匹配。")
            self._scene_prepared = False; self._scene_on_gpu = False
            return False
        try:
            self.v0s_reordered = v0s[ordered_prim_indices].copy()
            self.e1s_reordered = e1s[ordered_prim_indices].copy()
            self.e2s_reordered = e2s[ordered_prim_indices].copy()
            self.normals_reordered = normals[ordered_prim_indices].copy()
            self.labels_reordered = labels[ordered_prim_indices].copy()
            self.colors_reordered = colors[ordered_prim_indices].copy()
        except IndexError as e:
            print(f"    错误: 几何数据重新排序期间发生索引错误: {e}。")
            self._scene_prepared = False; self._scene_on_gpu = False
            return False
        log_step('几何数据重新排序', t0_reorder)
        self._scene_prepared = True

        # --- 4. 上传静态数据到 GPU ---
        if self._scene_on_gpu and force_rebuild:
             self.release_gpu_memory()
        print("    上传静态场景数据到 GPU...")
        t_upload_start = _now()
        try:
            self.d_flat_nodes = cuda.to_device(self.flat_nodes)
            self.d_v0s = cuda.to_device(self.v0s_reordered)
            self.d_e1s = cuda.to_device(self.e1s_reordered)
            self.d_e2s = cuda.to_device(self.e2s_reordered)
            self.d_normals = cuda.to_device(self.normals_reordered)
            self.d_labels = cuda.to_device(self.labels_reordered)
            self.d_colors = cuda.to_device(self.colors_reordered)
            self._scene_on_gpu = True
            log_step('GPU 静态数据上传', t_upload_start)
        except Exception as e:
            print(f"    错误: 上传数据到 GPU 失败: {e}")
            self.release_gpu_memory()
            self._scene_on_gpu = False
            return False

        log_step('场景准备总耗时', t0_prepare)
        return True

    def release_gpu_memory(self):
        """释放存储在 GPU 上的静态场景数据。"""
        if not self._scene_on_gpu: return
        print("[Renderer] 释放 GPU 静态场景数据显存...")
        del self.d_flat_nodes, self.d_v0s, self.d_e1s, self.d_e2s, self.d_normals, self.d_labels, self.d_colors
        self.d_flat_nodes = self.d_v0s = self.d_e1s = self.d_e2s = self.d_normals = self.d_labels = self.d_colors = None
        self._scene_on_gpu = False

    def render(self, camera_params: Dict, render_params: Dict):
        """
        使用预先准备好的场景数据和当前参数渲染一帧。

        Args:
            camera_params: 包含相机设置的字典。
            render_params: 包含渲染设置（光照、雾效、输出、着色模式等）的字典。

        Returns:
             包含输出数组 ('rgb', 'depth', 'semantic', 'points') 的字典，
             如果渲染失败则返回 None。
             如果场景为空，则返回包含默认空/天空值的字典。
        """
        total_t0 = _now()

        # --- [0] 检查状态 ---
        if not self._gpu_available: print("错误: GPU 不可用。"); return None
        if not self._scene_prepared: print("错误: 场景尚未在 CPU 上准备好。请先调用 prepare_scene()"); return None
        if not self._scene_on_gpu: print("错误: 场景尚未上传到 GPU。请先调用 prepare_scene()"); return None
        if self.d_flat_nodes is None or self.d_v0s is None: print("错误: GPU 设备指针无效。"); return None

        # --- [1] 解析参数和设置相机 ---
        try:
            # 相机参数
            cam_o = np.array(camera_params['origin'], dtype=GEOMETRY_DTYPE)
            cam_t = np.array(camera_params['target'], dtype=GEOMETRY_DTYPE)
            cam_up_vec = np.array(camera_params.get('up_vector', [0.0, 0.0, 1.0]), dtype=GEOMETRY_DTYPE)
            W = int(camera_params['width'])
            H = int(camera_params['height'])
            fov_degrees = float(camera_params['fov_deg'])
            # 计算相机坐标系
            cam_dir = cam_t - cam_o
            norm_cam_dir = np.linalg.norm(cam_dir)
            if norm_cam_dir < INTERSECTION_EPSILON: raise ValueError("相机原点和目标点太近")
            cam_dir /= norm_cam_dir
            right = np.cross(cam_dir, cam_up_vec)
            norm_right = np.linalg.norm(right)
            if norm_right < INTERSECTION_EPSILON:
                if abs(np.dot(cam_dir, cam_up_vec)) > 1.0 - INTERSECTION_EPSILON:
                    alt_right = np.array([1.0, 0.0, 0.0], dtype=GEOMETRY_DTYPE)
                    right = np.cross(cam_dir, alt_right)
                    norm_right = np.linalg.norm(right)
                    if norm_right < INTERSECTION_EPSILON:
                         alt_right = np.array([0.0, 1.0, 0.0], dtype=GEOMETRY_DTYPE)
                         right = np.cross(cam_dir, alt_right)
                         norm_right = np.linalg.norm(right)
                else:
                     right = np.array([1.0, 0.0, 0.0], dtype=GEOMETRY_DTYPE)
                     norm_right = 1.0
                if norm_right < INTERSECTION_EPSILON: raise ValueError("无法计算有效的相机右向量")
            right /= norm_right
            up = np.cross(right, cam_dir)
            fov_radians = np.deg2rad(fov_degrees)
            aspect_ratio = float(W) / float(H)
            screen_h = GEOMETRY_DTYPE(2.0 * np.tan(fov_radians / 2.0))
            screen_w = GEOMETRY_DTYPE(screen_h * aspect_ratio)

            # 渲染参数
            shading_mode_str = render_params.get("shading_mode", "phong").lower()
            if shading_mode_str == "phong": shading_mode_int = 0
            elif shading_mode_str == "simple_diffuse": shading_mode_int = 1
            else: print(f"警告: 未知 shading_mode '{shading_mode_str}', 使用 phong。"); shading_mode_int = 0

            sun_direction_host = np.array(render_params['sun_direction'], dtype=GEOMETRY_DTYPE)
            sun_direction_norm = np.linalg.norm(sun_direction_host)
            if sun_direction_norm < INTERSECTION_EPSILON: raise ValueError("太阳方向向量长度为零")
            sun_direction_host /= sun_direction_norm

            ambient_light = GEOMETRY_DTYPE(render_params['ambient_light'])
            sun_intensity = GEOMETRY_DTYPE(render_params['sun_intensity'])
            specular_color_host = np.array(render_params['specular_color'], dtype=GEOMETRY_DTYPE)
            specular_exponent = GEOMETRY_DTYPE(render_params['specular_exponent'])
            ks = GEOMETRY_DTYPE(render_params['ks'])
            shadow_bias = GEOMETRY_DTYPE(render_params.get('shadow_bias', 1e-3))
            sky_color_list = render_params.get('sky_color', [135, 206, 235])
            sky_color_host = np.array(sky_color_list, dtype=COLOR_DTYPE)
            fog_enabled = bool(render_params.get('fog_enabled', False))
            fog_color_list = render_params.get('fog_color', [200, 200, 200])
            fog_color_host_uint8 = np.array(fog_color_list, dtype=COLOR_DTYPE)
            fog_color_host_float = fog_color_host_uint8.astype(GEOMETRY_DTYPE) / 255.0
            fog_density = GEOMETRY_DTYPE(render_params.get('fog_density', 0.0))

            # 输出控制
            save_outputs_raw = render_params.get("save_outputs", ["rgb"])
            if isinstance(save_outputs_raw, str): save_outputs_list = {save_outputs_raw.lower()} # 使用集合提高查找效率
            elif isinstance(save_outputs_raw, (list, tuple)): save_outputs_list = {str(s).lower() for s in save_outputs_raw}
            else: save_outputs_list = {"rgb"}

            # OBJ 保存需要点云数据，即使 "points" 不在列表中也要下载
            needs_pts_download = "points" in save_outputs_list or "obj" in save_outputs_list

            file_prefix = render_params.get('file_prefix', 'render')
            output_dir_str = render_params.get('output_dir', 'output/render_default')
            output_dir = Path(output_dir_str)
            output_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            print(f"    错误: 解析参数或设置相机时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        # --- [2] GPU Raytrace ---
        t0_raytrace = _now()
        rgb_host, depth_host, sem_lbl_host, pts_host = None, None, None, None # Host 结果

        # 分配 GPU 输出缓冲
        d_rgb = cuda.device_array((H, W, 3), dtype=COLOR_DTYPE)
        d_depth = cuda.to_device(np.full((H, W), GPU_INF, dtype=DEPTH_DTYPE))
        d_sem = cuda.to_device(np.full((H, W), SKY_LABEL, dtype=LABEL_DTYPE))
        d_pts = cuda.to_device(np.full((H, W, 3), np.nan, dtype=POINT_DTYPE))

        # 上传动态参数到 GPU
        d_sun_dir = cuda.to_device(sun_direction_host)
        d_specular_color = cuda.to_device(specular_color_host)
        d_sky_color = cuda.to_device(sky_color_host)
        d_fog_color_float = cuda.to_device(fog_color_host_float)

        try:
            # --- 配置并启动内核 ---
            threads_per_block = (16, 16)
            blocks_per_grid_x = (W + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid_y = (H + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)

            raytrace_cuda_bvh_kernel[blocks_per_grid, threads_per_block](
                self.d_flat_nodes, self.d_v0s, self.d_e1s, self.d_e2s,
                self.d_normals, self.d_labels, self.d_colors,
                self.num_triangles,
                cam_o, cam_dir, right, up,
                d_sun_dir, d_specular_color, d_sky_color,
                screen_w, screen_h, W, H,
                ambient_light, sun_intensity, specular_exponent, ks,
                shadow_bias,
                fog_enabled, d_fog_color_float, fog_density,
                shading_mode_int,
                d_rgb, d_depth, d_sem, d_pts
            )
            cuda.synchronize()

            # --- 从 GPU 下载请求的结果 ---
            if "rgb" in save_outputs_list: rgb_host = d_rgb.copy_to_host()
            if "depth" in save_outputs_list:
                 depth_host = d_depth.copy_to_host()
                 depth_host[depth_host >= GPU_INF] = np.inf
            if "semantic" in save_outputs_list: sem_lbl_host = d_sem.copy_to_host()
            if needs_pts_download: pts_host = d_pts.copy_to_host() # 下载点云如果需要保存点或OBJ

        except Exception as e:
            print(f"\n---!! GPU 执行或数据传输错误: {e} ({type(e).__name__}) !!---")
            return None # 返回失败
        finally:
            # 清理动态分配的 GPU 资源
            del d_rgb, d_depth, d_sem, d_pts
            del d_sun_dir, d_specular_color, d_sky_color, d_fog_color_float

        # --- [3] 保存选定的输出 ---
        print(f'[Renderer] 保存请求的输出 (模式: {shading_mode_str}, Bias: {shadow_bias:.1E})...')
        t0_save = _now()
        ts_save = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # 文件名前缀包含模式和 bias
        output_prefix_full = output_dir / f'{file_prefix}_{shading_mode_str}_bias{shadow_bias:.2E}_{ts_save}'

        if "rgb" in save_outputs_list and rgb_host is not None:
            save_image(output_prefix_full.with_suffix(".png"), rgb_host)

        if "depth" in save_outputs_list and depth_host is not None:
             save_depth_visualization(output_prefix_full.with_name(f"{output_prefix_full.name}_depth.png"), depth_host)

        if "semantic" in save_outputs_list and sem_lbl_host is not None:
             if self.color_map and sky_color_list is not None:
                  save_semantic_visualization(output_prefix_full.with_name(f"{output_prefix_full.name}_semantic.png"), sem_lbl_host, self.color_map, sky_color_list)
             else: print("   警告: 无法保存语义图，缺少 color_map 或 sky_color。")

        if "points" in save_outputs_list and pts_host is not None:
             try:
                  np.savez_compressed(output_prefix_full.with_name(f"{output_prefix_full.name}_points.npz"), points=pts_host)
                  print(f"    已保存点云: {output_prefix_full.with_name(f'{output_prefix_full.name}_points.npz')}")
             except Exception as e: print(f"    保存点云时出错: {e}")

        # --- [新增] 保存 OBJ ---
        if "obj" in save_outputs_list:
             if self.v0s_reordered is not None and pts_host is not None: # 确保数据存在
                  # 计算合适的 far distance 用于视锥体可视化
                  far_dist = 1000.0 # 默认值
                  if depth_host is not None: # 如果下载了深度图
                       valid_depth_mask_obj = np.isfinite(depth_host) & (depth_host > 0)
                       if np.any(valid_depth_mask_obj):
                            far_dist = max(far_dist, np.max(depth_host[valid_depth_mask_obj]) * 1.2) # 稍微超出最大深度

                  save_combined_obj(
                       output_prefix_full.with_name(f"{output_prefix_full.name}_scene.obj"),
                       self.v0s_reordered, self.e1s_reordered, self.e2s_reordered, # 使用 CPU 上的重排数据
                       self.normals_reordered, self.labels_reordered, self.label_map,
                       cam_o, cam_dir, right, up, screen_w, screen_h, # 相机参数
                       pts_host, # 下载的点云数据
                       far=far_dist
                  )
             else:
                  print("   警告: 无法保存 OBJ，缺少必要的几何数据或点云数据。")

        if "params" in save_outputs_list:
            current_params = {
                 "timestamp_render_start_unix": total_t0,
                 "timestamp_save_str": ts_save,
                 "shading_mode": shading_mode_str,
                 "resolution": {"width": W, "height": H},
                 "fov_degrees": fov_degrees,
                 "camera": {"origin": cam_o.tolist(), "target": cam_t.tolist(), "up_vector": cam_up_vec.tolist()},
                 "lighting": {
                      "sun_direction": sun_direction_host.tolist(),
                      "sun_intensity": sun_intensity,
                      "ambient_light": ambient_light,
                      "specular_color": specular_color_host.tolist(),
                      "specular_exponent": specular_exponent,
                      "specular_coefficient_Ks": ks,
                 },
                 "shadow": {"shadow_bias": shadow_bias},
                 "fog": {"enabled": fog_enabled, "color_rgb": fog_color_list, "density": fog_density},
                 "rendering": {"sky_color_rgb": sky_color_list},
                 # 可以添加更多静态场景信息，如果需要的话
                 "scene_info": {
                      "num_triangles": self.num_triangles,
                      "bvh_node_count": self.flat_nodes.shape[0] if self.flat_nodes is not None else 0,
                      "label_map_keys": list(self.label_map.keys()) if self.label_map else [],
                 }
            }
            save_parameters_json(output_prefix_full.with_name(f"{output_prefix_full.name}_params.json"), current_params)

        # log_step('保存输出耗时', t0_save)
        # log_step(f'--- 单帧渲染总耗时', total_t0)

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
             try:
                  if hasattr(self, 'd_flat_nodes') and self.d_flat_nodes is not None:
                       self.release_gpu_memory()
             except Exception: pass # 忽略删除过程中的错误
