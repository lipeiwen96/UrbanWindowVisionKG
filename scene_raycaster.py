# scene_view.py
import numpy as np
from PIL import Image
import time

# ----------------------------
# 1）几何体生成函数
# ----------------------------
def create_box(center, size):
    """
    生成一个轴对齐的长方体的三角形网格。
    返回顶点 (8,3) 和三角形索引列表。
    """
    cx, cy, cz = center
    sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
    verts = np.array([
        [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz],
        [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz],
        [cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz],
        [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz],
    ], dtype=np.float32)
    faces = [
        (0,1,2),(0,2,3),  # bottom
        (4,6,5),(4,7,6),  # top
        (0,4,5),(0,5,1),  # front
        (2,6,7),(2,7,3),  # back
        (0,3,7),(0,7,4),  # left
        (1,5,6),(1,6,2),  # right
    ]
    return verts, faces


def create_cylinder(center, height, radius, segments=8):
    """
    生成一个垂直圆柱侧面网格（无上下盖）。
    返回顶点 (2*segments,3) 和三角形索引列表。
    """
    cx, cy, cz = center
    verts = []
    for i in range(segments):
        theta = 2*np.pi * i / segments
        x, z = cx + radius*np.cos(theta), cz + radius*np.sin(theta)
        verts.append((x, cy-height/2, z))
        verts.append((x, cy+height/2, z))
    verts = np.array(verts, dtype=np.float32)
    faces = []
    for i in range(segments):
        i0 = 2*i
        i1 = (i0+2) % (2*segments)
        faces.append((i0, i1, i1+1))
        faces.append((i0, i1+1, i0+1))
    return verts, faces

# ----------------------------
# 2）场景构建：地面 + 随机建筑 + 树
# ----------------------------
def build_scene():
    """
    构建简单场景，返回三角形基础数据与语义标签/颜色。
    返回:
      v0s,e1s,e2s: (N,3) 数组用于光线相交
      labels: (N,) 语义标签
      colors: (N,3) RGB 语义颜色
    """
    np.random.seed(42)
    tris, labs, cols = [], [], []
    sem_colors = {0:(150,200,150),1:(200,200,200),2:(200,100,100),3:(0,150,0)}
    # 地面
    gv = np.array([[-50,0,-50],[50,0,-50],[50,0,50],[-50,0,50]],dtype=np.float32)
    for f in [(0,1,2),(0,2,3)]:
        tris.append((gv[f[0]],gv[f[1]],gv[f[2]])); labs.append(0); cols.append(sem_colors[0])
    # 建筑
    for i in range(5):
        cx,cz = np.random.uniform(-20,20), np.random.uniform(-20,20)
        h = np.random.uniform(6,12); sx,sz = np.random.uniform(6,10), np.random.uniform(6,10)
        verts, faces = create_box((cx,h/2,cz),(sx,h,sz))
        sem = 1 if i<4 else 2  # 前四个为住宅，第五个为地标
        for a,b,c in faces:
            tris.append((verts[a], verts[b], verts[c])); labs.append(sem); cols.append(sem_colors[sem])
    # 树
    for _ in range(15):
        cx,cz = np.random.uniform(-20,20), np.random.uniform(-20,20)
        h = np.random.uniform(5,10)
        verts, faces = create_cylinder((cx,h/2,cz), h, 0.8, segments=8)
        for a,b,c in faces:
            tris.append((verts[a], verts[b], verts[c])); labs.append(3); cols.append(sem_colors[3])
    # 转为 NumPy 数组
    N = len(tris)
    v0s = np.empty((N,3),dtype=np.float32)
    e1s = np.empty((N,3),dtype=np.float32)
    e2s = np.empty((N,3),dtype=np.float32)
    labels = np.empty(N,dtype=np.int32)
    colors = np.empty((N,3),dtype=np.uint8)
    for i,(v0,v1,v2) in enumerate(tris):
        v0s[i], e1s[i], e2s[i] = v0, v1-v0, v2-v0
        labels[i], colors[i] = labs[i], cols[i]
    return v0s, e1s, e2s, labels, colors

# ----------------------------
# 3）Möller–Trumbore 光线-三角形相交
# ----------------------------
def intersect_ray_triangle(orig, dir, v0, e1, e2):
    eps = 1e-6
    h = np.cross(dir, e2)
    a = np.dot(e1, h)
    if abs(a) < eps:
        return np.inf
    f = 1.0 / a
    s = orig - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return np.inf
    q = np.cross(s, e1)
    v = f * np.dot(dir, q)
    if v < 0.0 or u + v > 1.0:
        return np.inf
    t = f * np.dot(e2, q)
    return t if t > eps else np.inf

# ----------------------------
# 4）光线投射（纯 Python）
# ----------------------------
def raytrace(v0s, e1s, e2s, labels, colors,
             cam_o, cam_dir, right, up,
             screen_w, screen_h, W, H,
             verbose=False):
    """
    逐像素发射光线，返回 RGB, depth, semantic label.
    verbose=True 时按行打印进度。
    """
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), np.inf, dtype=np.float32)
    sem = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        if verbose:
            print(f"    Raytrace progress: row {i+1}/{H}")
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
            if idx >= 0 and tmin < np.inf:
                depth[i, j] = tmin
                sem[i, j] = labels[idx]
                rgb[i, j] = colors[idx]
    return rgb, depth, sem

# ----------------------------
# 5）入口函数
# ----------------------------
def main():
    print("[1/5] Building scene...")
    t0 = time.time()
    v0s, e1s, e2s, labs, cols = build_scene()
    print(f"    Scene ready: {v0s.shape[0]} tris, time: {time.time()-t0:.2f}s")

    print("[2/5] Initializing camera...")
    cam_o = np.array([0,5,-15], dtype=np.float32)
    cam_t = np.array([0,5,0], dtype=np.float32)
    cam_dir = (cam_t-cam_o)/np.linalg.norm(cam_t-cam_o)
    cam_up = np.array([0,1,0], dtype=np.float32)
    right = np.cross(cam_dir, cam_up); right /= np.linalg.norm(right)
    up = np.cross(right, cam_dir)
    W, H = 256, 128
    fov = np.deg2rad(60)
    screen_h = 2*np.tan(fov/2)
    screen_w = screen_h * (W/H)
    print(f"    Camera: {W}x{H}, FOV={np.rad2deg(fov):.1f}°")

    print("[3/5] Raytracing...")
    t1 = time.time()
    rgb, depth, sem_lbl = raytrace(
        v0s, e1s, e2s, labs, cols,
        cam_o, cam_dir, right, up,
        screen_w, screen_h, W, H,
        verbose=True
    )
    print(f"    Raytrace done, time: {time.time()-t1:.2f}s")

    print("[4/5] Saving RGB image...")
    Image.fromarray(rgb).save('view.png')
    print("    view.png saved")

    print("[5/5] Saving depth and semantic maps...")
    maxd = np.nanmax(depth[np.isfinite(depth)])
    dimg = np.where(np.isfinite(depth), (depth/maxd*255).astype(np.uint8), 255)
    Image.fromarray(dimg, mode='L').save('depth.png')
    print("    depth.png saved")

    sem_img = np.zeros_like(rgb)
    sc = {0:(0,0,0),1:(200,200,200),2:(200,100,100),3:(0,150,0)}
    for s,c in sc.items(): sem_img[sem_lbl==s] = c
    Image.fromarray(sem_img).save('semantic.png')
    print("    semantic.png saved")

    print("All done!")


if __name__=='__main__':
    main()
