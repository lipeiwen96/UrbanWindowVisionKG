# scene_view.py
from collections import Counter
import sys
import os
import numpy as np
from PIL import Image
import time
from datetime import datetime
from numba import njit, prange


# ============================
# 1) 几何体生成模块
# ============================

def create_box(center, size):
    """
    生成轴对齐长方体网格，返回顶点列表和三角形面列表。
    center: (x,y,z) 中心坐标
    size:   (dx,dy,dz) 长宽高
    """
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    verts = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ], dtype=np.float32)
    faces = [
        (0, 1, 2), (0, 2, 3),  # 底面
        (4, 6, 5), (4, 7, 6),  # 顶面
        (0, 4, 5), (0, 5, 1),  # 前面
        (2, 6, 7), (2, 7, 3),  # 背面
        (0, 3, 7), (0, 7, 4),  # 左侧
        (1, 5, 6), (1, 6, 2),  # 右侧
    ]
    return verts, faces


def create_cylinder(center, height, radius, segments=12):
    """
    生成圆柱侧面（无上下盖），可用作树干。
    """
    cx, cy, cz = center
    verts = []
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x, z = cx + radius * np.cos(theta), cz + radius * np.sin(theta)
        verts.append((x, cy - height / 2, z))
        verts.append((x, cy + height / 2, z))
    verts = np.array(verts, dtype=np.float32)
    faces = []
    for i in range(segments):
        i0 = 2 * i
        i1 = (i0 + 2) % (2 * segments)
        faces.append((i0, i1, i1 + 1))
        faces.append((i0, i1 + 1, i0 + 1))
    return verts, faces


def create_cone_tree(base_center, height, radius, segments=12):
    """
    生成简易树冠-树干模型：将圆锥放在圆柱上。
    base_center: 圆柱底部中心
    """
    # 树干
    trunk_h = height * 0.3
    trunk_r = radius * 0.2
    trunk_verts, trunk_faces = create_cylinder(
        center=(base_center[0], base_center[1]+trunk_h/2, base_center[2]),
        height=trunk_h, radius=trunk_r, segments=segments)
    # 树冠圆锥
    cx, cy, cz = base_center
    crown_h = height * 0.7
    apex = np.array([cx, cy+trunk_h+crown_h, cz], dtype=np.float32)
    circle_verts = []
    for i in range(segments):
        theta = 2*np.pi * i / segments
        x = cx + radius*np.cos(theta)
        z = cz + radius*np.sin(theta)
        circle_verts.append((x, cy+trunk_h, z))
    circle_verts = np.array(circle_verts, dtype=np.float32)
    cone_faces = []
    for i in range(segments):
        cone_faces.append((0, i+1, (i+1)%segments+1))
    cone_verts = np.vstack((apex.reshape(1,3), circle_verts))
    # 合并几何
    verts = np.vstack((trunk_verts, cone_verts))
    faces = trunk_faces + [(f[0]+len(trunk_verts), f[1]+len(trunk_verts), f[2]+len(trunk_verts)) for f in cone_faces]
    return verts, faces

# ============================
# 2) 场景构建模块
# ============================
def build_scene():
    """
    构建城市场景，包含：
      - 住宅 (Box)
      - 地标 (Box, 高度更大)
      - 道路 (Thin Box 凸面)
      - 水域 (Flat Box + 蓝色)
      - 树木 (Cone+Cylinder)
    返回光线追踪所需数据。
    """
    np.random.seed(0)
    tris, labs, cols = [], [], []
    sem_colors = {
        0: (150,200,150),  # 地面
        1: (200,200,200),  # 住宅
        2: (220,180, 50),  # 道路
        3: (100,150,220),  # 水域
        4: (200,100,100),  # 地标
        5: (  0,150,  0),  # 树木
    }
    # 地面
    gv = np.array([[-50,0,-50],[50,0,-50],[50,0,50],[-50,0,50]],dtype=np.float32)
    for idx in [(0,1,2),(0,2,3)]: tris.append((gv[idx[0]],gv[idx[1]],gv[idx[2]])); labs.append(0); cols.append(sem_colors[0])
    # 建筑群 + 地标
    for i in range(10):
        x,z = np.random.uniform(-40,40), np.random.uniform(-40,40)
        h = np.random.uniform(5,10) if i<8 else np.random.uniform(12,20)
        w = np.random.uniform(5,8)
        verts, faces = create_box((x,h/2,z),(w,h,w))
        sem = 1 if i<8 else 4
        for a,b,c in faces: tris.append((verts[a],verts[b],verts[c])); labs.append(sem); cols.append(sem_colors[sem])
    # 道路
    for road in [((-50,0,-5),(50,0,5)), ((-5,0,-50),(5,0,50))]:
        (x1,y1,z1),(x2,y2,z2) = road
        length = np.linalg.norm([x2-x1, z2-z1])
        angle = np.arctan2(z2-z1, x2-x1)
        # thin box along road direction
        verts, faces = create_box(
            center=((x1+x2)/2, 0.01, (z1+z2)/2),
            size=(length,0.02,3)
        )
        # rotate around Y if needed
        for a,b,c in faces:
            tris.append((verts[a],verts[b],verts[c])); labs.append(2); cols.append(sem_colors[2])
    # 水域
    verts, faces = create_box((20, -0.01, 20),(30,0.02,30))
    for a,b,c in faces: tris.append((verts[a],verts[b],verts[c])); labs.append(3); cols.append(sem_colors[3])
    # 树木
    for _ in range(30):
        x,z = np.random.uniform(-40,40), np.random.uniform(-40,40)
        h = np.random.uniform(4,8)
        verts, faces = create_cone_tree((x,0,z), height=h, radius=1.5)
        for a,b,c in faces: tris.append((verts[a],verts[b],verts[c])); labs.append(5); cols.append(sem_colors[5])
    # 转数组
    N = len(tris)
    v0s = np.empty((N,3),dtype=np.float32)
    e1s = np.empty((N,3),dtype=np.float32)
    e2s = np.empty((N,3),dtype=np.float32)
    labels_arr = np.empty(N,dtype=np.int32)
    colors_arr = np.empty((N,3),dtype=np.uint8)
    for i,(v0,v1,v2) in enumerate(tris):
        v0s[i], e1s[i], e2s[i] = v0, v1-v0, v2-v0
        labels_arr[i], colors_arr[i] = labs[i], cols[i]
    return v0s, e1s, e2s, labels_arr, colors_arr

# ============================
# 3) 光线-三角形相交
# ============================
@njit
def intersect_ray_triangle(orig, dir, v0, e1, e2):
    eps = np.float32(1e-6)
    # 手写 cross = dir × e2
    h0 = dir[1] * e2[2] - dir[2] * e2[1]
    h1 = dir[2] * e2[0] - dir[0] * e2[2]
    h2 = dir[0] * e2[1] - dir[1] * e2[0]
    # 手写 dot = e1 ⋅ h
    a = e1[0] * h0 + e1[1] * h1 + e1[2] * h2
    if abs(a) < eps:
        return np.inf

    f = np.float32(1.0) / a
    # s = orig - v0
    s0 = orig[0] - v0[0]
    s1 = orig[1] - v0[1]
    s2 = orig[2] - v0[2]
    # u = f * (s ⋅ h)
    u = f * (s0 * h0 + s1 * h1 + s2 * h2)
    if u < 0.0 or u > 1.0:
        return np.inf

    # q = s × e1
    q0 = s1 * e1[2] - s2 * e1[1]
    q1 = s2 * e1[0] - s0 * e1[2]
    q2 = s0 * e1[1] - s1 * e1[0]
    # v = f * (dir ⋅ q)
    v = f * (dir[0] * q0 + dir[1] * q1 + dir[2] * q2)
    if v < 0.0 or u + v > 1.0:
        return np.inf

    # t = f * (e2 ⋅ q)
    t = f * (e2[0] * q0 + e2[1] * q1 + e2[2] * q2)
    return t if t > eps else np.inf

# ============================
# 4) 光线投射 (并行加速)
# ============================
@njit(parallel=True)
def raytrace_fast(v0s, e1s, e2s, labels, colors,
                  cam_o, cam_dir, right, up,
                  screen_w, screen_h, W, H,
                  verbose=False):
    """
    逐像素发射光线，返回视图RGB、深度、语义标签、交点坐标阵列。
    verbose=True 打印行进度。
    """
    rgb   = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), np.inf, dtype=np.float32)
    sem   = np.zeros((H, W), dtype=np.int32)
    pts   = np.full((H, W, 3), np.nan, dtype=np.float32)
    for i in range(H):
        if verbose: print(f"Raytrace row {i+1}/{H}")
        for j in range(W):
            u = (j + 0.5)/W - 0.5
            v = (i + 0.5)/H - 0.5
            d = cam_dir + u*screen_w*right - v*screen_h*up
            d /= np.linalg.norm(d)
            tmin, idx = np.inf, -1
            for k in range(v0s.shape[0]):
                t = intersect_ray_triangle(cam_o, d, v0s[k], e1s[k], e2s[k])
                if t < tmin:
                    tmin, idx = t, k
            if idx >= 0:
                depth[i,j] = tmin
                sem[i,j]   = labels[idx]
                rgb[i,j]   = colors[idx]
                pts[i,j]   = cam_o + d*tmin
    return rgb, depth, sem, pts

# ============================
# 5) 导出 OBJ
# ============================
def save_combined_obj(filename, v0s, e1s, e2s,
                      cam_o, cam_dir, right, up,
                      pts, far):
    """
    将场景网格、视锥和射线交点统一写入单个 OBJ 文件。
    """
    with open(filename, 'w') as f:
        f.write('# Combined scene + frustum + rays\n')
        vert_idx = 1
        # 场景三角面
        for i in range(v0s.shape[0]):
            v0 = v0s[i]; v1 = v0+e1s[i]; v2 = v0+e2s[i]
            f.write(f'v {v0[0]} {v0[1]} {v0[2]}\n')
            f.write(f'v {v1[0]} {v1[1]} {v1[2]}\n')
            f.write(f'v {v2[0]} {v2[1]} {v2[2]}\n')
            f.write(f'f {vert_idx} {vert_idx+1} {vert_idx+2}\n')
            vert_idx += 3
        # 视锥顶点
        corners = []
        for u in (-0.5,0.5):
            for v in (-0.5,0.5):
                d = cam_dir + u*screen_w*right - v*screen_h*up
                d /= np.linalg.norm(d)
                corners.append(cam_o + d*far)
        # 写入相机原点和四个角点
        f.write(f'v {cam_o[0]} {cam_o[1]} {cam_o[2]}\n')
        for c in corners:
            f.write(f'v {c[0]} {c[1]} {c[2]}\n')
        # 绘制视锥线段
        for i in range(1,5): f.write(f'l 1 {vert_idx + i}\n')
        vert_idx += 4
        # 写入射线交点并连线
        for i in range(pts.shape[0]):
            for j in range(pts.shape[1]):
                p = pts[i,j]
                if not np.isnan(p[0]):
                    f.write(f'v {p[0]} {p[1]} {p[2]}\n')
                    f.write(f'l 1 {vert_idx}\n')
                    vert_idx += 1

# ============================
# 6) 主流程
# ============================
def main():
    # 创建输出目录
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outd = os.path.join('output', ts)
    os.makedirs(outd, exist_ok=True)

    print('[1/6] Building scene...')
    t0 = time.time()
    v0s, e1s, e2s, labs, cols = build_scene()
    print(f'    Scene built: {len(v0s)} tris, time {time.time()-t0:.2f}s')

    print('[2/6] Setting up camera...')
    cam_o = np.array([0,5,-15], dtype=np.float32)
    cam_t = np.array([0,5,0], dtype=np.float32)
    global cam_dir, screen_w, screen_h, right, up
    cam_dir = (cam_t-cam_o)/np.linalg.norm(cam_t-cam_o)
    cam_up = np.array([0,1,0], dtype=np.float32)
    right = np.cross(cam_dir, cam_up); right /= np.linalg.norm(right)
    up    = np.cross(right, cam_dir)
    W, H = 512, 256
    fov = np.deg2rad(60)
    screen_h = 2*np.tan(fov/2)
    screen_w = screen_h*(W/H)
    print(f'    Camera configured: {W}x{H}, FOV {np.rad2deg(fov):.1f}°')

    print('[3/6] Raytracing...')
    t1 = time.time()
    rgb, depth, sem_lbl, pts = raytrace_fast(
        v0s, e1s, e2s, labs, cols,
        cam_o, cam_dir, right, up,
        screen_w, screen_h, W, H
    )
    print(f'    Raytracing done: {time.time()-t1:.2f}s')

    print('[4/6] Saving images...')
    fn_view  = os.path.join(outd, f'view_{ts}.png')
    fn_depth = os.path.join(outd, f'depth_{ts}.png')
    fn_sem   = os.path.join(outd, f'semantic_{ts}.png')
    Image.fromarray(rgb).save(fn_view);  print('    ', fn_view)
    maxd = np.nanmax(depth[np.isfinite(depth)])
    dmap = np.where(np.isfinite(depth),(depth/maxd*255).astype(np.uint8),255)
    Image.fromarray(dmap,mode='L').save(fn_depth);  print('    ', fn_depth)
    sem_img = np.zeros_like(rgb)
    sc = {0:(0,0,0),1:(200,200,200),2:(220,180,50),3:(100,150,220),4:(200,100,100),5:(0,150,0)}
    for s,c in sc.items(): sem_img[sem_lbl==s] = c
    Image.fromarray(sem_img).save(fn_sem); print('    ', fn_sem)

    print('[5/6] Saving combined OBJ...')
    fn_obj = os.path.join(outd, f'combined_{ts}.obj')
    far = np.nanmax(depth[np.isfinite(depth)])
    save_combined_obj(fn_obj, v0s, e1s, e2s, cam_o, cam_dir, right, up, pts, far)
    print('    ', fn_obj)

    print('Done! All outputs are in:', outd)


if __name__ == '__main__':
    main()