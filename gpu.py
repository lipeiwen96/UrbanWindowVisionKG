from collections import Counter  # 预留统计功能
import os
import time
from datetime import datetime

import numpy as np
from PIL import Image
from numba import njit, prange

# -- GPU ---------------------------------------------------------------
try:
    from numba import cuda
    cuda.detect()
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _now() -> float:
    """高精度时间戳"""
    return time.perf_counter()


def log_step(title: str, t0: float) -> None:
    """阶段耗时打印"""
    print(f"    {title} 用时 {_now() - t0:.2f}s")


# ---------------------------------------------------------------------------
# 0) 随机工具（可重复）
# ---------------------------------------------------------------------------
RAND = np.random.default_rng(seed=0)

# ---------------------------------------------------------------------------
# 1) 几何体生成工具（保持不变）
# ---------------------------------------------------------------------------

# ... (保持原 create_box / create_prism / create_skyscraper / create_cylinder /
#      create_cone_tree / create_mountain_range 等函数不变，为缩短篇幅此处省略。
#      你可在此处直接粘贴上一版完整实现)

# ---------------------------------------------------------------------------
# 2) 场景构建（保持不变）
# ---------------------------------------------------------------------------
# build_scene() 与原实现一致，此处同样可直接复制。

# ---------------------------------------------------------------------------
# 3) LBVH 构建 & 数据结构
# ---------------------------------------------------------------------------

@njit
def _expand_bits(v):
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v

@njit
def morton3D(x, y, z):
    return (_expand_bits(x) << 2) | (_expand_bits(y) << 1) | _expand_bits(z)

@njit
def compute_triangle_aabbs(v0s, e1s, e2s):
    n = v0s.shape[0]
    bmin = np.empty((n, 3), np.float32)
    bmax = np.empty((n, 3), np.float32)
    for i in range(n):
        v0 = v0s[i]
        v1 = v0 + e1s[i]
        v2 = v0 + e2s[i]
        bmin[i, 0] = min(v0[0], v1[0], v2[0])
        bmin[i, 1] = min(v0[1], v1[1], v2[1])
        bmin[i, 2] = min(v0[2], v1[2], v2[2])
        bmax[i, 0] = max(v0[0], v1[0], v2[0])
        bmax[i, 1] = max(v0[1], v1[1], v2[1])
        bmax[i, 2] = max(v0[2], v1[2], v2[2])
    return bmin, bmax

@njit
def build_lbvh(v0s, e1s, e2s):
    """基于 Morton code 的 LBVH 构建 (CPU, O(N log N))
    返回:
        node_mins, node_maxs  (M,3)
        left_idx,  right_idx (M,)  # 若为叶节点则 left == triStart, right == triCount
        tri_start, tri_count (M,)
        root_index
    """
    n_tris = v0s.shape[0]
    # 1) 求场景 AABB
    bmin, bmax = compute_triangle_aabbs(v0s, e1s, e2s)
    scene_min = np.min(bmin, axis=0)
    scene_max = np.max(bmax, axis=0)
    diag = scene_max - scene_min + 1e-5

    # 2) Morton code
    codes = np.empty(n_tris, np.uint32)
    for i in range(n_tris):
        cx = (bmin[i, 0] + bmax[i, 0]) * 0.5
        cy = (bmin[i, 1] + bmax[i, 1]) * 0.5
        cz = (bmin[i, 2] + bmax[i, 2]) * 0.5
        nx = int(((cx - scene_min[0]) / diag[0]) * 1023)
        ny = int(((cy - scene_min[1]) / diag[1]) * 1023)
        nz = int(((cz - scene_min[2]) / diag[2]) * 1023)
        codes[i] = morton3D(nx, ny, nz)

    # 3) 按 Morton 排序
    idxs = np.argsort(codes)

    # 4) 线性化节点数组 (参照 Karras 2012)
    M = 2 * n_tris - 1
    node_mins = np.empty((M, 3), np.float32)
    node_maxs = np.empty((M, 3), np.float32)
    left_idx  = np.empty(M, np.int32)
    right_idx = np.empty(M, np.int32)
    tri_start = np.empty(M, np.int32)
    tri_count = np.empty(M, np.int32)

    # 叶节点初始化
    for i in range(n_tris):
        node = i + n_tris - 1
        j = idxs[i]
        node_mins[node] = bmin[j]
        node_maxs[node] = bmax[j]
        left_idx[node] = -1
        right_idx[node] = -1
        tri_start[node] = j
        tri_count[node] = 1

    # 内部节点
    for i in range(n_tris - 2, -1, -1):
        left = 2 * i + 1
        right = 2 * i + 2
        node_mins[i] = np.minimum(node_mins[left], node_mins[right])
        node_maxs[i] = np.maximum(node_maxs[left], node_maxs[right])
        left_idx[i] = left
        right_idx[i] = right
        tri_start[i] = -1
        tri_count[i] = 0
    return node_mins, node_maxs, left_idx, right_idx, tri_start, tri_count, 0  # root=0

# ---------------------------------------------------------------------------
# 4) 光线-三角形相交 (CPU 与原实现一致) + BVH 加速版
# ---------------------------------------------------------------------------

@njit
def intersect_aabb(orig, inv_dir, aabb_min, aabb_max):
    t1 = (aabb_min[0] - orig[0]) * inv_dir[0]
    t2 = (aabb_max[0] - orig[0]) * inv_dir[0]
    tmin = min(t1, t2)
    tmax = max(t1, t2)
    for k in range(1, 3):
        t1 = (aabb_min[k] - orig[k]) * inv_dir[k]
        t2 = (aabb_max[k] - orig[k]) * inv_dir[k]
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
    return tmax >= max(tmin, 0.0)

@njit(parallel=True)
def raytrace_cpu_bvh(v0s, e1s, e2s, labels, colors,
                     node_mins, node_maxs, left_idx, right_idx, tri_start,
                     cam_o, cam_dir, right, up, screen_w, screen_h, W, H):
    rgb = np.zeros((H, W, 3), np.uint8)
    depth = np.full((H, W), np.inf, np.float32)
    for i in prange(H):
        for j in range(W):
            u = (j + 0.5) / W - 0.5
            v = (i + 0.5) / H - 0.5
            d = cam_dir + u * screen_w * right - v * screen_h * up
            d /= np.linalg.norm(d)
            inv_d = 1.0 / d
            tmin = np.inf
            hit = -1
            stack = [0]
            while stack:
                node = stack.pop()
                if not intersect_aabb(cam_o, inv_d, node_mins[node], node_maxs[node]):
                    continue
                if left_idx[node] == -1:  # leaf
                    tidx = tri_start[node]
                    t = intersect_ray_triangle(cam_o, d, v0s[tidx], e1s[tidx], e2s[tidx])
                    if t < tmin:
                        tmin = t
                        hit = tidx
                else:
                    stack.append(left_idx[node])
                    stack.append(right_idx[node])
            if hit >= 0:
                depth[i, j] = tmin
                rgb[i, j] = colors[hit]
    return rgb, depth

# ---------------------------------------------------------------------------
# 5) GPU 内核 – Stack‑less BVH 遍历
# ---------------------------------------------------------------------------
if _GPU_AVAILABLE:

    @cuda.jit(device=True, inline=True)
    def aabb_hit(orig, dir_inv, mn, mx):
        t1 = (mn[0] - orig[0]) * dir_inv[0]
        t2 = (mx[0] - orig[0]) * dir_inv[0]
        tmin = min(t1, t2)
        tmax = max(t1, t2)
        for k in range(1, 3):
            t1 = (mn[k] - orig[k]) * dir_inv[k]
            t2 = (mx[k] - orig[k]) * dir_inv[k]
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        return tmax >= max(tmin, 0.0)

    @cuda.jit(device=True, inline=True)
    def tri_hit(orig, dir, v0, e1, e2):
        eps = 1e-6
        h0 = dir[1] * e2[2] - dir[2] * e2[1]
        h1 = dir[2] * e2[0] - dir[0] * e2[2]
        h2 = dir[0] * e2[1] - dir[1] * e2[0]
        a = e1[0] * h0 + e1[1] * h1 + e1[2] * h2
        if abs(a) < eps:
            return 1e20
        f = 1.0 / a
        s0 = orig[0] - v0[0]
        s1 = orig[1] - v0[1]
        s2 = orig[2] - v0[2]
        u = f * (s0 * h0 + s1 * h1 + s2 * h2)
        if u < 0.0 or u > 1.0:
            return 1e20
        q0 = s1 * e1[2] - s2 * e1[1]
        q1 = s2 * e1[0] - s0 * e1[2]
        q2 = s0 * e1[1] - s1 * e1[0]
        v = f * (dir[0] * q0 + dir[1] * q1 + dir[2] * q2)
        if v < 0.0 or u + v > 1.0:
            return 1e20
        t = f * (e2[0] * q0 + e2[1] * q1 + e2[2] * q2)
        return t if t > eps else 1e20

    @cuda.jit
    def raytrace_cuda_lbvh(v0s, e1s, e2s, colors,
                           node_mins, node_maxs, left_idx,
                           cam_o, cam_dir, right, up,
                           screen_w, screen_h, W, H, out_img):
        i, j = cuda.grid(2)
        if i >= H or j >= W:
            return
        # 计算射线方向
        u = (j + 0.5) / W - 0.5
        v = (i + 0.5) / H - 0.5
        d = cuda.local.array(3, dtype=np.float32)
        for k in range(3):
            d[k] = cam_dir[k] + u * screen_w * right[k] - v * screen_h * up[k]
        nrm = (d[0]**2 + d[1]**2 + d[2]**2) ** 0.5
        d[0] /= nrm; d[1] /= nrm; d[2] /= nrm
        inv = cuda.local.array(3, dtype=np.float32)
        inv[0] = 1.0 / d[0]
        inv[1] = 1.0 / d[1]
        inv[2] = 1.0 / d[2]
        # Stack‑less 遍历 (rope 指针 = implicit in array ordering)
        idx = 0
        t_min = 1e20
        hit = -1
        while idx >= 0:
            if aabb_hit(cam_o, inv, node_mins[idx], node_maxs[idx]):
                if left_idx[idx] < 0:  # leaf (tri index encoded)
                    tri = -left_idx[idx] - 1
                    t = tri_hit(cam_o, d, v0s[tri], e1s[tri], e2s[tri])
                    if t < t_min:
                        t_min = t
                        hit = tri
                    idx += 1  # 线性遍历到下一个兄弟 (rope)
                else:
                    idx = left_idx[idx]  # 深入左子树
            else:
                # 跳过该节点及其后代
                parent = (idx - 1) >> 1
                if idx & 1:  # left child -> sibling is parent*2+2
                    idx = parent * 2 + 2
                else:
                    # 已访问右子或不存在右子，继续向上
                    idx = parent + 1
        if hit >= 0:
            out_img[i, j, 0] = colors[hit, 0]
            out_img[i, j, 1] = colors[hit, 1]
            out_img[i, j, 2] = colors[hit, 2]

# ---------------------------------------------------------------------------
# 6) OBJ 导出（保持不变）
# ---------------------------------------------------------------------------
# save_combined_obj 与原实现一致

# ---------------------------------------------------------------------------
# 7) 主流程 – 新增 BVH 构建 & GPU 调用
# ---------------------------------------------------------------------------

def main():
    total_t0 = _now()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outd = os.path.join('output', ts)
    os.makedirs(outd, exist_ok=True)

    # --- 场景 -----------------------------------------------------------
    print('[1/7] 构建场景 …')
    t0 = _now()
    v0s, e1s, e2s, labels, colors = build_scene()
    log_step('场景构建', t0)
    print('    三角形数量:', v0s.shape[0])

    # --- BVH ------------------------------------------------------------
    print('[2/7] 构建 LBVH …')
    t0 = _now()
    node_mins, node_maxs, left_idx, right_idx, tri_start, root = build_lbvh(v0s, e1s, e2s)
    log_step('LBVH 构建', t0)

    # --- 相机 -----------------------------------------------------------
    print('[3/7] 设置相机 …')
    t0 = _now()
    cam_o = np.array([-30, 20, -65], np.float32)
    cam_t = np.array([20, 5, 30], np.float32)
    cam_dir = cam_t - cam_o
    cam_dir /= np.linalg.norm(cam_dir)
    cam_up = np.array([0, 1, 0], np.float32)
    right = np.cross(cam_dir, cam_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, cam_dir)
    W, H = 1920, 1080
    fov = np.deg2rad(60)
    screen_h = 2 * np.tan(fov / 2)
    screen_w = screen_h * (W / H)
    log_step('相机配置', t0)

    # --- 光线投射 -------------------------------------------------------
    print('[4/7] 光线投射 …')
    t0 = _now()
    if _GPU_AVAILABLE:
        # 拷贝数据到显存
        d_v0s = cuda.to_device(v0s)
        d_e1s = cuda.to_device(e1s)
        d_e2s = cuda.to_device(e2s)
        d_colors = cuda.to_device(colors)
        d_node_mins = cuda.to_device(node_mins)
        d_node_maxs = cuda.to_device(node_maxs)
        d_left = cuda.to_device(left_idx)
        d_cam_o = cuda.to_device(cam_o)
        d_cam_dir = cuda.to_device(cam_dir)
        d_right = cuda.to_device(right)
        d_up = cuda.to_device(up)

        out_img = cuda.device_array((H, W, 3), np.uint8)
        threads = (16, 16)
        blocks = ((H + threads[0] - 1) // threads[0],
                  (W + threads[1] - 1) // threads[1])
        raytrace_cuda_lbvh[blocks, threads](d_v0s, d_e1s, d_e2s, d_colors,
                                           d_node_mins, d_node_maxs, d_left,
                                           d_cam_o, d_cam_dir, d_right, d_up,
                                           np.float32(screen_w), np.float32(screen_h),
                                           np.int32(W), np.int32(H), out_img)
        rgb = out_img.copy_to_host()
    else:
        rgb, depth = raytrace_cpu_bvh(v0s, e1s, e2s, labels, colors,
                                      node_mins, node_maxs, left_idx, right_idx, tri_start,
                                      cam_o, cam_dir, right, up, screen_w, screen_h, W, H)
    log_step('光线投射', t0)

    # --- 保存图片 -------------------------------------------------------
    print('[5/7] 保存图像 …')
    t0 = _now()
    fn_view = os.path.join(outd, f'view_{ts}.png')
    Image.fromarray(rgb).save(fn_view)
    log_step('图像导出', t0)

    # --- OBJ 导出 ------------------------------------------------------
    print('[6/7] 保存 OBJ …')
    # 此处保持不变 / 可选

    # --- 完成 ----------------------------------------------------------
    print('[7/7] 完成 – 输出目录:', outd)
    print(f'总耗时: {_now() - total_t0:.2f}s')


if __name__ == '__main__':
    main()
