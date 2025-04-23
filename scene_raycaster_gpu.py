from collections import Counter               # 预留统计功能
import os
import time
from datetime import datetime

import numpy as np
from PIL import Image
from numba import njit, prange

# -- GPU ---------------------------------------------------------------
try:
    from numba import cuda
    cuda.detect()          # 无 GPU 时抛 CudaSupportError
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
# 1) 几何体生成工具
# ---------------------------------------------------------------------------


def create_box(center, size):
    """生成轴对齐长方体 (8 verts, 12 tris)"""
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    verts = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ], dtype=np.float32)
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (3, 2, 6), (3, 6, 7),
        (0, 3, 7), (0, 7, 4),
        (1, 5, 6), (1, 6, 2),
    ]
    return verts, faces


def create_prism(center, height, radius, sides=3):
    cx, cy, cz = center
    half = height / 2
    verts = []
    for i in range(sides):
        th = 2 * np.pi * i / sides
        x, z = cx + radius * np.cos(th), cz + radius * np.sin(th)
        verts.append((x, cy - half, z))
        verts.append((x, cy + half, z))
    verts = np.asarray(verts, np.float32)
    faces = []
    for i in range(sides):
        i0, i1 = 2 * i, (2 * (i + 1)) % (2 * sides)
        faces.append((i0, i1, i1 + 1))
        faces.append((i0, i1 + 1, i0 + 1))
    return verts, faces


def create_skyscraper(center, base, height, levels=5, taper=0.7):
    verts, faces = [], []
    cx, cy, cz = center
    seg_h = height / levels
    cur_w = base
    offset = 0
    for lv in range(levels):
        seg_center = (cx, cy + offset + seg_h / 2, cz)
        v, f = create_box(seg_center, (cur_w, seg_h, cur_w))
        o = len(verts)
        verts.extend(v)
        faces.extend([(a + o, b + o, c + o) for a, b, c in f])
        offset += seg_h
        cur_w *= taper
    return np.asarray(verts, np.float32), faces


def create_cylinder(center, height, radius, *, segments: int = 12):
    cx, cy, cz = center
    verts = []
    for i in range(segments):
        th = 2 * np.pi * i / segments
        x = cx + radius * np.cos(th); z = cz + radius * np.sin(th)
        verts.append((x, cy - height / 2, z))
        verts.append((x, cy + height / 2, z))
    verts = np.asarray(verts, np.float32)
    faces = []
    for i in range(segments):
        i0 = 2 * i
        i1 = (i0 + 2) % (2 * segments)
        faces.append((i0, i1, i1 + 1))
        faces.append((i0, i1 + 1, i0 + 1))
    return verts, faces


def create_cone_tree(base_center, height, radius, *, segments: int = 12):
    trunk_h = height * 0.3
    trunk_r = radius * 0.2
    trunk_verts, trunk_faces = create_cylinder(
        center=(base_center[0], base_center[1] + trunk_h / 2, base_center[2]),
        height=trunk_h, radius=trunk_r, segments=segments)
    cx, cy, cz = base_center
    crown_h = height * 0.7
    apex = np.array([cx, cy + trunk_h + crown_h, cz], np.float32)
    circle = []
    for i in range(segments):
        th = 2 * np.pi * i / segments
        x, z = cx + radius * np.cos(th), cz + radius * np.sin(th)
        circle.append((x, cy + trunk_h, z))
    circle = np.asarray(circle, np.float32)
    cone_faces = [(0, i + 1, (i + 1) % segments + 1) for i in range(segments)]
    cone_verts = np.vstack((apex.reshape(1, 3), circle))
    verts = np.vstack((trunk_verts, cone_verts))
    faces = trunk_faces + [(a + len(trunk_verts), b + len(trunk_verts), c + len(trunk_verts))
                           for a, b, c in cone_faces]
    return verts, faces


# ---------------------------------------------------------------------------
# 2) 改进山体 – 抖动脊线
# ---------------------------------------------------------------------------

def create_mountain_range(x_start, x_end, z_pos, *, segs=120, depth=12,
                          h_min=18, h_max=45):
    xs = np.linspace(x_start, x_end, segs + 1)
    base_noise = RAND.uniform(-1, 1, xs.shape)
    base_noise = np.interp(base_noise, (-1, 1), (h_min, h_max))
    verts, faces = [], []
    for i in range(segs):
        x0, x1 = xs[i], xs[i + 1]
        h0, h1 = base_noise[i], base_noise[i + 1]
        v = [
            (x0, 0, z_pos), (x1, 0, z_pos), (x0, h0, z_pos), (x1, h1, z_pos),
            (x0, 0, z_pos + depth), (x1, 0, z_pos + depth),
            (x0, h0, z_pos + depth), (x1, h1, z_pos + depth),
        ]
        idx0 = len(verts)
        verts.extend(v)
        f = lambda a, b, c: (idx0 + a, idx0 + b, idx0 + c)
        faces += [f(0, 1, 3), f(0, 3, 2), f(4, 6, 7), f(4, 7, 5),
                  f(0, 2, 6), f(0, 6, 4), f(1, 5, 7), f(1, 7, 3),
                  f(2, 3, 7), f(2, 7, 6)]
    return np.asarray(verts, np.float32), faces


# ---------------------------------------------------------------------------
# 3) 场景构建
# ---------------------------------------------------------------------------

def build_scene():
    tris, labs, cols = [], [], []
    C = {0: (150, 200, 150), 1: (200, 200, 200), 2: (220, 180, 50),
         3: (90, 140, 210), 4: (210, 80, 80), 5: (0, 140, 0), 6: (120, 120, 120)}

    # 草地
    g_v, g_f = create_box((0, -0.05, 0), (160, 0.1, 160))
    for a, b, c in g_f:
        tris.append((g_v[a], g_v[b], g_v[c])); labs.append(0); cols.append(C[0])

    # 湖泊（仅顶部两面）
    lake_v, lake_f = create_box((30, 0.02, 40), (40, 0.05, 30))
    for idx, (a, b, c) in enumerate(lake_f):
        if idx in (2, 3):      # (4,6,5)(4,7,6) 的顺序索引
            tris.append((lake_v[a], lake_v[b], lake_v[c]))
            labs.append(3); cols.append(C[3])

    # 道路网
    road_w = 4
    grid = np.linspace(-60, 60, 9)
    for x in grid:
        v, f = create_box((x, 0.05, 0), (road_w, 0.1, 160))
        for a, b, c in f:
            tris.append((v[a], v[b], v[c])); labs.append(2); cols.append(C[2])
    for z in grid:
        v, f = create_box((0, 0.05, z), (160, 0.1, road_w))
        for a, b, c in f:
            tris.append((v[a], v[b], v[c])); labs.append(2); cols.append(C[2])

    # 普通建筑
    for _ in range(120):
        x, z = RAND.uniform(-60, 60), RAND.uniform(-60, 60)
        w = RAND.uniform(4, 10); h = RAND.uniform(6, 12)
        v, f = create_box((x, h/2, z), (w, h, w))
        for a, b, c in f:
            tris.append((v[a], v[b], v[c])); labs.append(1); cols.append(C[1])

    # 摩天楼
    for _ in range(25):
        x, z = RAND.uniform(-50, 50), RAND.uniform(-50, 50)
        base = RAND.uniform(6, 10); h = RAND.uniform(30, 45)
        v, f = create_skyscraper((x, 0, z), base, h, levels=6)
        for a, b, c in f:
            tris.append((v[a], v[b], v[c])); labs.append(4); cols.append(C[4])

    # 三棱柱建筑
    for _ in range(30):
        x, z = RAND.uniform(-55, 55), RAND.uniform(-55, 55)
        r = RAND.uniform(4, 6); h = RAND.uniform(10, 18)
        v, f = create_prism((x, h/2, z), h, r, sides=3)
        for a, b, c in f:
            tris.append((v[a], v[b], v[c])); labs.append(4); cols.append(C[4])

    # 树木
    for _ in range(450):
        x, z = RAND.uniform(-70, 70), RAND.uniform(-70, 70)
        if 15 < x < 45 and 25 < z < 55:
            continue
        h = RAND.uniform(5, 9)
        v, f = create_cone_tree((x, 0, z), h, 1.8, segments=16)
        for a, b, c in f:
            tris.append((v[a], v[b], v[c])); labs.append(5); cols.append(C[5])

    # 山脉
    m_v, m_f = create_mountain_range(-90, 90, z_pos=70, segs=150)
    for a, b, c in m_f:
        tris.append((m_v[a], m_v[b], m_v[c])); labs.append(6); cols.append(C[6])

    # 转成 Numpy 批量数组
    N = len(tris)
    v0s = np.empty((N, 3), np.float32)
    e1s = np.empty((N, 3), np.float32)
    e2s = np.empty((N, 3), np.float32)
    labels = np.empty(N, np.int32)
    colors = np.empty((N, 3), np.uint8)
    for i, (v0, v1, v2) in enumerate(tris):
        v0s[i] = v0; e1s[i] = v1 - v0; e2s[i] = v2 - v0
        labels[i] = labs[i]; colors[i] = cols[i]
    return v0s, e1s, e2s, labels, colors


# ---------------------------------------------------------------------------
# 4) 光线‑三角形相交（CPU 版）
# ---------------------------------------------------------------------------
@njit
def intersect_ray_triangle(orig, dir, v0, e1, e2):
    eps = 1e-6
    h0 = dir[1] * e2[2] - dir[2] * e2[1]
    h1 = dir[2] * e2[0] - dir[0] * e2[2]
    h2 = dir[0] * e2[1] - dir[1] * e2[0]
    a = e1[0]*h0 + e1[1]*h1 + e1[2]*h2
    if abs(a) < eps:
        return np.inf
    f = 1.0 / a
    s0 = orig[0]-v0[0]; s1=orig[1]-v0[1]; s2=orig[2]-v0[2]
    u = f*(s0*h0 + s1*h1 + s2*h2)
    if u < 0. or u > 1.:
        return np.inf
    q0 = s1*e1[2]-s2*e1[1]
    q1 = s2*e1[0]-s0*e1[2]
    q2 = s0*e1[1]-s1*e1[0]
    v = f*(dir[0]*q0 + dir[1]*q1 + dir[2]*q2)
    if v < 0. or u+v > 1.:
        return np.inf
    t = f*(e2[0]*q0 + e2[1]*q1 + e2[2]*q2)
    return t if t > eps else np.inf


@njit(parallel=True)
def raytrace_cpu(v0s, e1s, e2s, labels, colors,
                 cam_o, cam_dir, right, up,
                 screen_w, screen_h, W, H):
    rgb = np.zeros((H, W, 3), np.uint8)
    depth = np.full((H, W), np.inf, np.float32)
    sem   = np.zeros((H, W), np.int32)
    pts   = np.full((H, W, 3), np.nan, np.float32)
    for i in prange(H):
        for j in range(W):
            u = (j + 0.5)/W - 0.5
            v = (i + 0.5)/H - 0.5
            d = cam_dir + u*screen_w*right - v*screen_h*up
            d /= np.linalg.norm(d)
            tmin = np.inf; idx = -1
            for k in range(v0s.shape[0]):
                t = intersect_ray_triangle(cam_o, d, v0s[k], e1s[k], e2s[k])
                if t < tmin:
                    tmin, idx = t, k
            if idx >= 0:
                depth[i,j] = tmin
                sem  [i,j] = labels[idx]
                rgb  [i,j] = colors[idx]
                pts  [i,j] = cam_o + d*tmin
    return rgb, depth, sem, pts


# ---------------------------------------------------------------------------
# 5) GPU 版内核（仅当 _GPU_AVAILABLE=True）
# ---------------------------------------------------------------------------
if _GPU_AVAILABLE:

    @cuda.jit(device=True, inline=True)
    def ray_tri_t(orig, dir, v0, e1, e2):
        eps = 1e-6
        h0 = dir[1]*e2[2] - dir[2]*e2[1]
        h1 = dir[2]*e2[0] - dir[0]*e2[2]
        h2 = dir[0]*e2[1] - dir[1]*e2[0]
        a  = e1[0]*h0 + e1[1]*h1 + e1[2]*h2
        if abs(a) < eps:
            return 1e20
        f  = 1.0 / a
        s0 = orig[0]-v0[0]; s1=orig[1]-v0[1]; s2=orig[2]-v0[2]
        u = f*(s0*h0 + s1*h1 + s2*h2)
        if u < 0. or u > 1.:
            return 1e20
        q0 = s1*e1[2]-s2*e1[1]
        q1 = s2*e1[0]-s0*e1[2]
        q2 = s0*e1[1]-s1*e1[0]
        v = f*(dir[0]*q0 + dir[1]*q1 + dir[2]*q2)
        if v < 0. or u+v > 1.:
            return 1e20
        t = f*(e2[0]*q0 + e2[1]*q1 + e2[2]*q2)
        return t if t > eps else 1e20


    @cuda.jit
    def raytrace_cuda(v0s, e1s, e2s, labels, colors,
                      cam_o, cam_dir, right, up,
                      scr_w, scr_h, W, H,
                      rgb, depth, sem, pts):
        i, j = cuda.grid(2)
        if i >= H or j >= W:
            return
        # 主光线方向
        u = (j + 0.5)/W - 0.5
        v = (i + 0.5)/H - 0.5
        d0 = cuda.local.array(3, np.float32)
        for k in range(3):
            d0[k] = cam_dir[k] + u*scr_w*right[k] - v*scr_h*up[k]
        nrm = (d0[0]**2 + d0[1]**2 + d0[2]**2)**0.5
        d0[0]/=nrm; d0[1]/=nrm; d0[2]/=nrm
        tmin = 1e20
        hit  = -1
        for k in range(v0s.shape[0]):
            t = ray_tri_t(cam_o, d0, v0s[k], e1s[k], e2s[k])
            if t < tmin:
                tmin, hit = t, k
        if hit >= 0:
            depth[i, j] = tmin
            sem  [i, j] = labels[hit]
            rgb  [i, j, 0] = colors[hit,0]
            rgb  [i, j, 1] = colors[hit,1]
            rgb  [i, j, 2] = colors[hit,2]
            pts[i, j, 0] = cam_o[0] + d0[0]*tmin
            pts[i, j, 1] = cam_o[1] + d0[1]*tmin
            pts[i, j, 2] = cam_o[2] + d0[2]*tmin
        else:
            depth[i, j] = 1e20
            sem  [i, j] = 0


# ---------------------------------------------------------------------------
# 6) OBJ 导出（保持不变）
# ---------------------------------------------------------------------------

def save_combined_obj(filename, v0s, e1s, e2s,
                      cam_o, cam_dir, right, up,
                      pts, far):
    with open(filename, 'w') as f:
        f.write('# Combined scene + frustum + rays\n')
        vidx = 1
        for i in range(v0s.shape[0]):
            v0 = v0s[i]
            v1 = v0 + e1s[i]
            v2 = v0 + e2s[i]
            f.write(f"v {v0[0]} {v0[1]} {v0[2]}\n")
            f.write(f"v {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"v {v2[0]} {v2[1]} {v2[2]}\n")
            f.write(f"f {vidx} {vidx+1} {vidx+2}\n")
            vidx += 3
        corners = []
        for du in (-0.5, 0.5):
            for dv in (-0.5, 0.5):
                d = cam_dir + du*screen_w*right - dv*screen_h*up
                d /= np.linalg.norm(d)
                corners.append(cam_o + d*far)
        f.write(f"v {cam_o[0]} {cam_o[1]} {cam_o[2]}\n")
        for c in corners:
            f.write(f"v {c[0]} {c[1]} {c[2]}\n")
        for i in range(1,5):
            f.write(f"l 1 {vidx+i}\n")
        vidx += 4
        for i in range(pts.shape[0]):
            for j in range(pts.shape[1]):
                p = pts[i,j]
                if not np.isnan(p[0]):
                    f.write(f"v {p[0]} {p[1]} {p[2]}\n")
                    f.write(f"l 1 {vidx}\n")
                    vidx += 1


# ---------------------------------------------------------------------------
# 7) 主流程
# ---------------------------------------------------------------------------

def main():
    total_t0 = _now()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outd = os.path.join('output', ts)
    os.makedirs(outd, exist_ok=True)

    # --- 场景 -----------------------------------------------------------
    print('[1/6] 构建场景 …')
    t0 = _now()
    v0s, e1s, e2s, labels, colors = build_scene()
    log_step('场景构建', t0)
    print('    三角形数量:', len(v0s))

    # --- 相机 -----------------------------------------------------------
    print('[2/6] 设置相机 …')
    global cam_dir, screen_w, screen_h, right, up
    t0 = _now()
    cam_o = np.array([-30, 20, -65], np.float32)
    cam_t = np.array([20, 5, 30], np.float32)
    cam_dir = (cam_t - cam_o); cam_dir /= np.linalg.norm(cam_dir)
    cam_up  = np.array([0,1,0], np.float32)
    right = np.cross(cam_dir, cam_up); right /= np.linalg.norm(right)
    up    = np.cross(right, cam_dir)
    W, H = 2048, 1024
    fov = np.deg2rad(60)
    screen_h = 2*np.tan(fov/2)
    screen_w = screen_h*(W/H)
    log_step('相机配置', t0)

    # --- 光线投射 -------------------------------------------------------
    print('[3/6] 光线投射 …')
    t0 = _now()
    if _GPU_AVAILABLE:
        # 复制到 GPU
        d_v0s    = cuda.to_device(v0s)
        d_e1s    = cuda.to_device(e1s)
        d_e2s    = cuda.to_device(e2s)
        d_labels = cuda.to_device(labels)
        d_colors = cuda.to_device(colors)
        d_rgb    = cuda.device_array((H, W, 3), np.uint8)
        d_depth  = cuda.device_array((H, W   ), np.float32)
        d_sem    = cuda.device_array((H, W   ), np.int32)
        d_pts    = cuda.device_array((H, W, 3), np.float32)
        # 常量参数也放显存，便于索引
        d_cam_o   = cuda.to_device(cam_o)
        d_cam_dir = cuda.to_device(cam_dir)
        d_right   = cuda.to_device(right)
        d_up      = cuda.to_device(up)
        threads = (16,16)
        blocks  = ((H+threads[0]-1)//threads[0], (W+threads[1]-1)//threads[1])
        raytrace_cuda[blocks, threads](d_v0s, d_e1s, d_e2s,
                                       d_labels, d_colors,
                                       d_cam_o, d_cam_dir, d_right, d_up,
                                       np.float32(screen_w), np.float32(screen_h),
                                       np.int32(W), np.int32(H),
                                       d_rgb, d_depth, d_sem, d_pts)
        rgb     = d_rgb.copy_to_host()
        depth   = d_depth.copy_to_host()
        sem_lbl = d_sem.copy_to_host()
        pts     = d_pts.copy_to_host()
    else:
        rgb, depth, sem_lbl, pts = raytrace_cpu(
            v0s, e1s, e2s, labels, colors,
            cam_o, cam_dir, right, up,
            screen_w, screen_h, W, H)
    log_step('光线投射', t0)

    # --- 导出图像 -------------------------------------------------------
    print('[4/6] 保存图像 …')
    t0 = _now()
    fn_view  = os.path.join(outd, f'view_{ts}.png')
    fn_depth = os.path.join(outd, f'depth_{ts}.png')
    fn_sem   = os.path.join(outd, f'semantic_{ts}.png')
    Image.fromarray(rgb).save(fn_view)
    dmax = np.nanmax(depth[np.isfinite(depth)])
    dmap = np.where(np.isfinite(depth), (depth/dmax*255).astype(np.uint8), 255)
    Image.fromarray(dmap, 'L').save(fn_depth)
    sem_img = np.zeros_like(rgb)
    palette = {0:(0,0,0),1:(200,200,200),2:(220,180,50),3:(100,150,220),4:(200,100,100),5:(0,150,0)}
    for s,c in palette.items():
        sem_img[sem_lbl==s] = c
    Image.fromarray(sem_img).save(fn_sem)
    log_step('图像导出', t0)

    # --- OBJ -----------------------------------------------------------
    print('[5/6] 保存 OBJ …')
    t0 = _now()
    fn_obj = os.path.join(outd, f'combined_{ts}.obj')
    far = np.nanmax(depth[np.isfinite(depth)])
    save_combined_obj(fn_obj, v0s, e1s, e2s, cam_o, cam_dir, right, up, pts, far)
    log_step('OBJ 导出', t0)

    # --- 完成 ----------------------------------------------------------
    print('[6/6] 完成 – 输出目录:', outd)
    print(f'总耗时: {_now() - total_t0:.2f}s')


if __name__ == '__main__':
    main()
