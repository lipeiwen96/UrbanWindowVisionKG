# scene_view.py
import os
import numpy as np
from PIL import Image
import time
from datetime import datetime
from numba import cuda

# ============================
# 1) 几何体生成模块 (CPU)
# ============================

def create_box(center, size):
    cx, cy, cz = center
    sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
    verts = np.array([
        [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz],
        [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz],
        [cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz],
        [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz],
    ], dtype=np.float32)
    faces = np.array([
        (0,1,2),(0,2,3),  # 底面
        (4,6,5),(4,7,6),  # 顶面
        (0,4,5),(0,5,1),  # 前面
        (2,6,7),(2,7,3),  # 背面
        (0,3,7),(0,7,4),  # 左侧
        (1,5,6),(1,6,2),  # 右侧
    ], dtype=np.int32)
    return verts, faces


def create_cylinder(center, height, radius, segments=12):
    cx, cy, cz = center
    verts = np.zeros((2*segments,3), dtype=np.float32)
    for i in range(segments):
        theta = 2*np.pi * i / segments
        x, z = cx + radius*np.cos(theta), cz + radius*np.sin(theta)
        verts[2*i]   = (x, cy-height/2, z)
        verts[2*i+1] = (x, cy+height/2, z)
    faces = np.zeros((2*segments,3), dtype=np.int32)
    idx = 0
    for i in range(segments):
        i0 = 2*i
        i1 = (i0+2) % (2*segments)
        faces[idx]   = (i0, i1, i1+1); idx += 1
        faces[idx]   = (i0, i1+1, i0+1); idx += 1
    return verts, faces


def create_cone_tree(base_center, height, radius, segments=12):
    # 树干
    trunk_h = height * 0.3
    trunk_r = radius * 0.2
    t_verts, t_faces = create_cylinder(
        (base_center[0], base_center[1]+trunk_h/2, base_center[2]),
        trunk_h, trunk_r, segments)
    # 树冠圆锥
    apex = np.array([base_center[0], base_center[1]+trunk_h+height*0.7, base_center[2]], dtype=np.float32)
    circle = np.zeros((segments,3), dtype=np.float32)
    for i in range(segments):
        theta = 2*np.pi*i/segments
        circle[i] = (base_center[0]+radius*np.cos(theta), base_center[1]+trunk_h, base_center[2]+radius*np.sin(theta))
    c_verts = np.vstack((apex.reshape(1,3), circle))
    c_faces = np.zeros((segments,3), dtype=np.int32)
    for i in range(segments): c_faces[i] = (0, i+1, (i+1)%segments+1)
    verts = np.vstack((t_verts, c_verts)).astype(np.float32)
    faces = np.vstack((t_faces, c_faces + t_verts.shape[0])).astype(np.int32)
    return verts, faces

# ============================
# 2) 场景构建模块 (CPU)
# ============================

def build_scene():
    np.random.seed(0)
    tris, labs, cols = [], [], []
    sem_colors = {
        0:(150,200,150), # 地面
        1:(200,200,200), # 住宅
        2:(220,180, 50), # 道路
        3:(100,150,220), # 水域
        4:(200,100,100), # 地标
        5:(0,150,  0),   # 树木
    }
    # 地面
    gv = np.array([[-50,0,-50],[50,0,-50],[50,0,50],[-50,0,50]],dtype=np.float32)
    for f in [(0,1,2),(0,2,3)]: tris.append((gv[f[0]],gv[f[1]],gv[f[2]])); labs.append(0); cols.append(sem_colors[0])
    # 建筑 + 地标
    for i in range(10):
        x,z = np.random.uniform(-40,40), np.random.uniform(-40,40)
        h = np.random.uniform(5,10) if i<8 else np.random.uniform(12,20)
        verts, faces = create_box((x,h/2,z),(np.random.uniform(5,8),h,np.random.uniform(5,8)))
        sem = 1 if i<8 else 4
        for a,b,c in faces: tris.append((verts[a],verts[b],verts[c])); labs.append(sem); cols.append(sem_colors[sem])
    # 道路
    for p1,p2 in [((-50,0,-5),(50,0,5)),((-5,0,-50),(5,0,50))]:
        length = np.linalg.norm(np.array(p2)-np.array(p1))
        center = ((p1[0]+p2[0])/2,0.01,(p1[2]+p2[2])/2)
        verts, faces = create_box(center,(length,0.02,3))
        for a,b,c in faces: tris.append((verts[a],verts[b],verts[c])); labs.append(2); cols.append(sem_colors[2])
    # 水域
    verts, faces = create_box((20,-0.01,20),(30,0.02,30))
    for a,b,c in faces: tris.append((verts[a],verts[b],verts[c])); labs.append(3); cols.append(sem_colors[3])
    # 树木
    for _ in range(30):
        x,z = np.random.uniform(-40,40), np.random.uniform(-40,40)
        verts, faces = create_cone_tree((x,0,z), height=np.random.uniform(4,8), radius=1.5)
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
# 3) GPU Kernel: 光线投射
# ============================
@cuda.jit
def gpu_raytrace(v0s, e1s, e2s, labels, colors,
                 cam_o, cam_dir, right, up,
                 screen_w, screen_h,
                 out_rgb, out_depth, out_sem, out_pts):
    i, j = cuda.grid(2)
    H, W = out_sem.shape
    if i < H and j < W:
        u = (j + 0.5)/W - 0.5
        v = (i + 0.5)/H - 0.5
        d0x = cam_dir[0] + u*screen_w*right[0] - v*screen_h*up[0]
        d0y = cam_dir[1] + u*screen_w*right[1] - v*screen_h*up[1]
        d0z = cam_dir[2] + u*screen_w*right[2] - v*screen_h*up[2]
        norm = (d0x*d0x + d0y*d0y + d0z*d0z) ** 0.5
        dx,dy,dz = d0x/norm, d0y/norm, d0z/norm
        tmin = 1e20
        idx = -1
        for k in range(v0s.shape[0]):
            v0x,v0y,v0z = v0s[k,0], v0s[k,1], v0s[k,2]
            e1x,e1y,e1z = e1s[k,0], e1s[k,1], e1s[k,2]
            e2x,e2y,e2z = e2s[k,0], e2s[k,1], e2s[k,2]
            hx = dy*e2z - dz*e2y
            hy = dz*e2x - dx*e2z
            hz = dx*e2y - dy*e2x
            a = e1x*hx + e1y*hy + e1z*hz
            if abs(a) < 1e-6: continue
            f = 1.0 / a
            sx = cam_o[0] - v0x; sy = cam_o[1] - v0y; sz = cam_o[2] - v0z
            u_ = f * (sx*hx + sy*hy + sz*hz)
            if u_ < 0.0 or u_ > 1.0: continue
            qx = sy*e1z - sz*e1y
            qy = sz*e1x - sx*e1z
            qz = sx*e1y - sy*e1x
            v_ = f * (dx*qx + dy*qy + dz*qz)
            if v_ < 0.0 or u_+v_ > 1.0: continue
            t = f * (e2x*qx + e2y*qy + e2z*qz)
            if t > 1e-6 and t < tmin:
                tmin, idx = t, k
        if idx >= 0:
            out_depth[i,j] = tmin
            out_sem[i,j]   = labels[idx]
            out_rgb[i,j,0] = colors[idx,0]
            out_rgb[i,j,1] = colors[idx,1]
            out_rgb[i,j,2] = colors[idx,2]
            out_pts[i,j,0] = cam_o[0] + dx*tmin
            out_pts[i,j,1] = cam_o[1] + dy*tmin
            out_pts[i,j,2] = cam_o[2] + dz*tmin

# ============================
# 4) 导出合并 OBJ
# ============================
def save_combined_obj(filename, v0s, e1s, e2s,
                      cam_o, cam_dir, right, up,
                      pts, far, screen_w, screen_h):
    with open(filename, 'w') as f:
        f.write('# Combined scene + frustum + rays')
        vert_idx = 1
        for i in range(v0s.shape[0]):
            v0 = v0s[i]; v1 = v0+e1s[i]; v2 = v0+e2s[i]
            for v in (v0, v1, v2): f.write(f'v {v[0]} {v[1]} {v[2]}')
            f.write(f'f {vert_idx} {vert_idx+1} {vert_idx+2}')
            vert_idx += 3
        corners = []
        for u in (-0.5,0.5):
            for v in (-0.5,0.5):
                d0x = cam_dir[0] + u*screen_w*right[0] - v*screen_h*up[0]
                d0y = cam_dir[1] + u*screen_w*right[1] - v*screen_h*up[1]
                d0z = cam_dir[2] + u*screen_w*right[2] - v*screen_h*up[2]
                norm = (d0x*d0x + d0y*d0y + d0z*d0z) ** 0.5
                corners.append(cam_o + np.array([d0x/norm, d0y/norm, d0z/norm]) * far)
        f.write(f'v {cam_o[0]} {cam_o[1]} {cam_o[2]}')
        for c in corners: f.write(f'v {c[0]} {c[1]} {c[2]}')
        for i in range(1,5): f.write(f'l 1 {vert_idx + i}')
        vert_idx += 4
        for i in range(pts.shape[0]):
            for j in range(pts.shape[1]):
                p = pts[i,j]
                if not np.isnan(p[0]):
                    f.write(f'v {p[0]} {p[1]} {p[2]}')
                    f.write(f'l 1 {vert_idx}')
                    vert_idx += 1

# ============================
# 5) 主流程
# ============================
def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outd = os.path.join('output', ts)
    os.makedirs(outd, exist_ok=True)

    print('[1/5] Building scene...')
    t0 = time.time()
    v0s, e1s, e2s, labs, cols = build_scene()
    print(f'    Scene built: {v0s.shape[0]} tris, {time.time()-t0:.2f}s')

    print('[2/5] Setting up camera...')
    cam_o = np.array([0,5,-15], dtype=np.float32)
    cam_t = np.array([0,5,0],  dtype=np.float32)
    cam_dir = (cam_t-cam_o)/np.linalg.norm(cam_t-cam_o)
    cam_up = np.array([0,1,0], dtype=np.float32)
    right  = np.cross(cam_dir, cam_up); right  /= np.linalg.norm(right)
    up     = np.cross(right, cam_dir)
    W, H   = 512, 256
    fov    = np.deg2rad(60)
    screen_h = 2*np.tan(fov/2)
    screen_w = screen_h*(W/H)
    print(f'    Camera: {W}x{H}, FOV={np.rad2deg(fov):.1f}°')

    print('[3/5] Raytracing on GPU...')
    t1 = time.time()
    dv0s = cuda.to_device(v0s)
    de1s = cuda.to_device(e1s)
    de2s = cuda.to_device(e2s)
    dlabs = cuda.to_device(labs)
    dcols = cuda.to_device(cols)
    d_cam_o = cuda.to_device(cam_o)
    d_dir   = cuda.to_device(cam_dir)
    d_r     = cuda.to_device(right)
    d_u     = cuda.to_device(up)
    out_rgb   = cuda.device_array((H, W, 3), dtype=np.uint8)
    out_depth = cuda.device_array((H, W),    dtype=np.float32)
    out_sem   = cuda.device_array((H, W),    dtype=np.int32)
    out_pts   = cuda.device_array((H, W, 3), dtype=np.float32)
    tpbx, tpby = 16,16
    bpgx = (H+tpbx-1)//tpbx; bpgy = (W+tpby-1)//tpby
    gpu_raytrace[(bpgx,bpgy),(tpbx,tpby)](
        dv0s, de1s, de2s, dlabs, dcols,
        d_cam_o, d_dir, d_r, d_u,
        screen_w, screen_h,
        out_rgb, out_depth, out_sem, out_pts
    )
    rgb   = out_rgb.copy_to_host()
    depth = out_depth.copy_to_host()
    sem   = out_sem.copy_to_host()
    pts   = out_pts.copy_to_host()
    print(f'    GPU raytrace: {time.time()-t1:.2f}s')

    print('[4/5] Saving images...')
    fn_view  = os.path.join(outd, f'view_{ts}.png')
    fn_depth = os.path.join(outd, f'depth_{ts}.png')
    fn_sem   = os.path.join(outd, f'semantic_{ts}.png')
    Image.fromarray(rgb).save(fn_view)
    maxd = np.nanmax(depth[np.isfinite(depth)])
    dmap = np.where(np.isfinite(depth),(depth/maxd*255).astype(np.uint8),255)
    Image.fromarray(dmap,mode='L').save(fn_depth)
    sem_img = np.zeros_like(rgb)
    sc = {0:(0,0,0),1:(200,200,200),2:(220,180,50),3:(100,150,220),4:(200,100,100),5:(0,150,0)}
    for s,c in sc.items(): sem_img[sem==s]=c
    Image.fromarray(sem_img).save(fn_sem)
    print('    Images saved')

    print('[5/5] Saving combined OBJ...')
    fn_obj = os.path.join(outd, f'combined_{ts}.obj')
    far = np.nanmax(depth[np.isfinite(depth)])
    save_combined_obj(fn_obj, v0s, e1s, e2s, cam_o, cam_dir, right, up, pts, far, screen_w, screen_h)
    print('    OBJ saved:', fn_obj)
    print('Done. Outputs in', outd)


if __name__ == '__main__':
    main()


# pip install numba numpy pillow
# conda install cudatoolkit