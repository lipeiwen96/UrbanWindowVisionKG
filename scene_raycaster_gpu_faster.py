# -*- coding: utf-8 -*-
# from collections import Counter # 预留统计功能
import os
import time
from datetime import datetime

import numpy as np
from PIL import Image
from numba import njit, prange

# -- GPU Availability Check --------------------------------------------------
try:
    from numba import cuda
    cuda.detect()  # Raises CudaSupportError if no GPU or CUDA toolkit found
    _GPU_AVAILABLE = True
    print(f"CUDA available: {_GPU_AVAILABLE}")
    print(f"Using GPU: {cuda.get_current_device().name.decode()}")
except Exception as e:
    print(f"CUDA not available or error during detection: {e}")
    _GPU_AVAILABLE = False

# --- Constants for Precision and Data Types ---
# NOTE: Ray tracing heavily relies on floating-point precision for geometry
#       calculations (directions, intersections, distances). Using integers
#       directly for these would lead to significant accuracy loss and incorrect results.
#       float32 is generally the standard and provides a good balance of
#       precision and performance on GPUs. float16 could be an option for
#       memory/bandwidth savings but might introduce precision issues.
#       Optimizations like scaling/quantizing to integers for transfer often
#       introduce more overhead (conversion cost) than they save, especially
#       compared to acceleration structures.
GEOMETRY_DTYPE = np.float32
INDEX_DTYPE = np.int32
COLOR_DTYPE = np.uint8
LABEL_DTYPE = np.int32
DEPTH_DTYPE = np.float32
POINT_DTYPE = np.float32 # Intersection points require float precision

# --- Global Color Map --- <<< MOVE C HERE
# Define colors using the COLOR_DTYPE, accessible globally
C = {
    0: np.array([150, 200, 150], dtype=COLOR_DTYPE), # Grass (Also often used for background in palette)
    1: np.array([200, 200, 200], dtype=COLOR_DTYPE), # Building
    2: np.array([220, 180, 50], dtype=COLOR_DTYPE),  # Road
    3: np.array([90, 140, 210], dtype=COLOR_DTYPE),  # Water
    4: np.array([210, 80, 80], dtype=COLOR_DTYPE),   # Skyscraper/Special
    5: np.array([0, 140, 0], dtype=COLOR_DTYPE),     # Tree
    6: np.array([120, 120, 120], dtype=COLOR_DTYPE)  # Mountain
}
# Optional: Define a separate explicit background color for the semantic image palette
PALETTE_BACKGROUND_COLOR = (0, 0, 0)

# --- Utility Functions ------------------------------------------------------
def _now() -> float:
    """High-resolution timestamp"""
    return time.perf_counter()

def log_step(title: str, t0: float) -> None:
    """Log duration of a step"""
    print(f"    {title} took {_now() - t0:.2f}s")

# --- Random Number Generator (Reproducible) --------------------------------
RAND = np.random.default_rng(seed=0)

# --- Geometry Generation Functions -----------------------------------------
# (Keep these functions as they are, ensuring they output GEOMETRY_DTYPE)
def create_box(center, size):
    """Generates an axis-aligned box (8 verts, 12 tris)"""
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    verts = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ], dtype=GEOMETRY_DTYPE) # Use defined dtype
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
        (3, 2, 6), (3, 6, 7), (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2),
    ]
    return verts, faces

def create_prism(center, height, radius, sides=3):
    """Generates a prism"""
    cx, cy, cz = center
    half = height / 2
    verts = []
    for i in range(sides):
        th = 2 * np.pi * i / sides
        x, z = cx + radius * np.cos(th), cz + radius * np.sin(th)
        verts.append((x, cy - half, z))
        verts.append((x, cy + half, z))
    verts = np.asarray(verts, dtype=GEOMETRY_DTYPE) # Use defined dtype
    faces = []
    for i in range(sides):
        i0, i1 = 2 * i, (2 * (i + 1)) % (2 * sides)
        faces.append((i0, i1, i1 + 1))
        faces.append((i0, i1 + 1, i0 + 1))
    return verts, faces

def create_skyscraper(center, base, height, levels=5, taper=0.7):
    """Generates a tapered skyscraper"""
    verts, faces = [], []
    cx, cy, cz = center
    seg_h = height / levels
    cur_w = base
    offset = 0
    for lv in range(levels):
        seg_center = (cx, cy + offset + seg_h / 2, cz)
        v, f = create_box(seg_center, (cur_w, seg_h, cur_w))
        o = len(verts)
        verts.extend(v) # Extends with GEOMETRY_DTYPE arrays
        faces.extend([(a + o, b + o, c + o) for a, b, c in f])
        offset += seg_h
        cur_w *= taper
    return np.asarray(verts, dtype=GEOMETRY_DTYPE), faces # Final conversion

def create_cylinder(center, height, radius, *, segments: int = 12):
    """Generates a cylinder"""
    cx, cy, cz = center
    verts = []
    half_h = height / 2
    for i in range(segments):
        th = 2 * np.pi * i / segments
        x = cx + radius * np.cos(th); z = cz + radius * np.sin(th)
        verts.append((x, cy - half_h, z))
        verts.append((x, cy + half_h, z))
    verts = np.asarray(verts, dtype=GEOMETRY_DTYPE) # Use defined dtype
    faces = []
    for i in range(segments):
        i0 = 2 * i
        i1 = (i0 + 2) % (2 * segments)
        faces.append((i0, i1, i1 + 1))
        faces.append((i0, i1 + 1, i0 + 1))
    return verts, faces

def create_cone_tree(base_center, height, radius, *, segments: int = 12):
    """Generates a simple cone tree"""
    cx, cy, cz = base_center
    trunk_h = height * 0.3
    trunk_r = radius * 0.2
    trunk_base_cy = cy + trunk_h / 2
    trunk_verts, trunk_faces = create_cylinder(
        center=(cx, trunk_base_cy, cz),
        height=trunk_h, radius=trunk_r, segments=segments) # Returns GEOMETRY_DTYPE

    crown_h = height * 0.7
    crown_base_y = cy + trunk_h
    apex = np.array([cx, crown_base_y + crown_h, cz], dtype=GEOMETRY_DTYPE) # Use defined dtype
    circle = []
    for i in range(segments):
        th = 2 * np.pi * i / segments
        x, z = cx + radius * np.cos(th), cz + radius * np.sin(th)
        circle.append((x, crown_base_y, z))
    circle = np.asarray(circle, dtype=GEOMETRY_DTYPE) # Use defined dtype

    # Cone faces reference vertices relative to cone_verts array
    cone_faces = [(0, i + 1, (i + 1) % segments + 1) for i in range(segments)]
    cone_verts = np.vstack((apex.reshape(1, 3), circle)) # Already GEOMETRY_DTYPE

    verts = np.vstack((trunk_verts, cone_verts)) # Combine GEOMETRY_DTYPE arrays
    # Adjust cone face indices to account for trunk vertices
    faces = trunk_faces + [(a + len(trunk_verts), b + len(trunk_verts), c + len(trunk_verts))
                           for a, b, c in cone_faces]
    return verts, faces

def create_mountain_range(x_start, x_end, z_pos, *, segs=120, depth=12,
                          h_min=18, h_max=45):
    """Generates a mountain range segment"""
    xs = np.linspace(x_start, x_end, segs + 1, dtype=GEOMETRY_DTYPE)
    # Generate smooth random heights using interpolation
    # Use a smaller number of control points for noise to make it smoother
    num_noise_points = max(5, segs // 10)
    noise_xs = np.linspace(x_start, x_end, num_noise_points)
    noise_ys_raw = RAND.uniform(-1, 1, noise_xs.shape)
    # Basic smoothing (moving average) - optional
    # kernel_size = 3
    # noise_ys_raw = np.convolve(noise_ys_raw, np.ones(kernel_size)/kernel_size, mode='same')
    noise_ys = np.interp(xs, noise_xs, noise_ys_raw) # Interpolate to full resolution
    base_heights = np.interp(noise_ys, (-1, 1), (h_min, h_max))

    verts, faces = [], []
    zero_y = GEOMETRY_DTYPE(0) # Use dtype for constants
    z_pos_dtype = GEOMETRY_DTYPE(z_pos)
    depth_dtype = GEOMETRY_DTYPE(depth)

    for i in range(segs):
        x0, x1 = xs[i], xs[i + 1]
        h0, h1 = base_heights[i], base_heights[i + 1]
        # Ensure heights are GEOMETRY_DTYPE
        h0_dtype = GEOMETRY_DTYPE(h0)
        h1_dtype = GEOMETRY_DTYPE(h1)

        # Define vertices for the current segment (ensure dtype)
        v = [
            (x0, zero_y, z_pos_dtype), (x1, zero_y, z_pos_dtype),
            (x0, h0_dtype, z_pos_dtype), (x1, h1_dtype, z_pos_dtype),
            (x0, zero_y, z_pos_dtype + depth_dtype), (x1, zero_y, z_pos_dtype + depth_dtype),
            (x0, h0_dtype, z_pos_dtype + depth_dtype), (x1, h1_dtype, z_pos_dtype + depth_dtype),
        ]
        idx0 = len(verts)
        verts.extend(v) # Extends list with tuples, will be converted later

        # Define faces using vertex indices relative to the start of this segment
        f = lambda a, b, c: (idx0 + a, idx0 + b, idx0 + c)
        # Front face, Back face, Side faces, Top face
        faces += [f(0, 1, 3), f(0, 3, 2),  # Front
                  f(4, 6, 7), f(4, 7, 5),  # Back
                  f(0, 4, 6), f(0, 6, 2),  # Left side
                  f(1, 3, 7), f(1, 7, 5),  # Right side
                  f(2, 6, 7), f(2, 7, 3)]  # Top
                  # Add bottom faces if needed: f(0,5,1), f(0,4,5)

    return np.asarray(verts, dtype=GEOMETRY_DTYPE), faces # Convert list of tuples


# --- Scene Building -------------------------------------------------------
def build_scene():
    """Builds the entire scene geometry"""
    tris_geom, tris_labs, tris_cols = [], [], [] # Use more descriptive names
    # Define colors using the COLOR_DTYPE
    C = {
        0: np.array([150, 200, 150], dtype=COLOR_DTYPE), # Grass
        1: np.array([200, 200, 200], dtype=COLOR_DTYPE), # Building
        2: np.array([220, 180, 50], dtype=COLOR_DTYPE),  # Road
        3: np.array([90, 140, 210], dtype=COLOR_DTYPE),  # Water
        4: np.array([210, 80, 80], dtype=COLOR_DTYPE),   # Skyscraper/Special
        5: np.array([0, 140, 0], dtype=COLOR_DTYPE),     # Tree
        6: np.array([120, 120, 120], dtype=COLOR_DTYPE)  # Mountain
    }

    # Helper to add objects
    def add_object(verts, faces, label, color_map):
        color = color_map[label]
        for a, b, c in faces:
            # Ensure indices are within bounds
            if a < len(verts) and b < len(verts) and c < len(verts):
                tris_geom.append((verts[a], verts[b], verts[c]))
                tris_labs.append(label)
                tris_cols.append(color)
            else:
                 print(f"Warning: Face indices ({a},{b},{c}) out of bounds for verts length {len(verts)}")


    # Ground
    g_v, g_f = create_box((0, -0.05, 0), (160, 0.1, 160))
    add_object(g_v, g_f, 0, C)

    # Lake (only top surface)
    lake_v, lake_f = create_box((30, 0.02, 40), (40, 0.05, 30))
    # Indices for the top face: (4,6,5) -> verts[4], verts[6], verts[5]
    #                          (4,7,6) -> verts[4], verts[7], verts[6]
    # These correspond to face indices 2 and 3 in the standard box face list
    add_object(lake_v, [lake_f[i] for i in [2, 3]], 3, C)

    # Road Network
    road_w = 4.0
    road_h = 0.1 # Slightly above ground/lake
    grid_coords = np.linspace(-60, 60, 9, dtype=GEOMETRY_DTYPE)
    for x in grid_coords:
        v, f = create_box((x, road_h/2, 0), (road_w, road_h, 160.0 + road_w)) # Extend slightly
        add_object(v, f, 2, C)
    for z in grid_coords:
        # Avoid double-drawing intersections by adjusting length slightly? Or accept overlap.
        v, f = create_box((0, road_h/2, z), (160.0 + road_w, road_h, road_w)) # Extend slightly
        add_object(v, f, 2, C)

    # Buildings
    num_buildings = 120
    for _ in range(num_buildings):
        x, z = RAND.uniform(-60, 60), RAND.uniform(-60, 60)
        # Simple check to avoid placing directly on lake (approximate)
        if 10 < x < 50 and 25 < z < 55: continue
        w = RAND.uniform(4, 10); h = RAND.uniform(6, 12)
        v, f = create_box((x, h/2, z), (w, h, w))
        add_object(v, f, 1, C)

    # Skyscrapers
    num_skyscrapers = 25
    for _ in range(num_skyscrapers):
        x, z = RAND.uniform(-50, 50), RAND.uniform(-50, 50)
        if 10 < x < 50 and 25 < z < 55: continue # Avoid lake
        base = RAND.uniform(6, 10); h = RAND.uniform(30, 45)
        v, f = create_skyscraper((x, 0, z), base, h, levels=6, taper=0.8)
        add_object(v, f, 4, C) # Use label 4

    # Prisms
    num_prisms = 30
    for _ in range(num_prisms):
        x, z = RAND.uniform(-55, 55), RAND.uniform(-55, 55)
        if 10 < x < 50 and 25 < z < 55: continue # Avoid lake
        r = RAND.uniform(4, 6); h = RAND.uniform(10, 18)
        v, f = create_prism((x, h/2, z), h, r, sides=3)
        add_object(v, f, 4, C) # Use label 4 (or a different one if needed)

    # Trees
    num_trees = 450
    for _ in range(num_trees):
        x, z = RAND.uniform(-70, 70), RAND.uniform(-70, 70)
        # Avoid lake and roads (approximate checks)
        on_lake = (10 < x < 50 and 25 < z < 55)
        on_road = False
        for gx in grid_coords: # Check vertical roads
             if abs(x - gx) < road_w / 1.8: on_road = True; break
        if not on_road:
            for gz in grid_coords: # Check horizontal roads
                if abs(z - gz) < road_w / 1.8: on_road = True; break
        if on_lake or on_road: continue

        h = RAND.uniform(5, 9)
        v, f = create_cone_tree((x, 0, z), h, 1.8, segments=8) # Reduced segments for trees
        add_object(v, f, 5, C)

    # Mountain Range
    m_v, m_f = create_mountain_range(x_start=-90, x_end=90, z_pos=70, segs=100, depth=25, h_min=20, h_max=55) # Adjusted params
    add_object(m_v, m_f, 6, C)
    m_v2, m_f2 = create_mountain_range(x_start=-90, x_end=90, z_pos=95, segs=80, depth=20, h_min=15, h_max=40) # Second, further range
    add_object(m_v2, m_f2, 6, C)


    # Convert to Numpy arrays (Structure of Arrays - good for GPU)
    N = len(tris_geom)
    if N == 0: # Handle empty scene case
        print("Warning: Scene is empty!")
        return (np.empty((0, 3), dtype=GEOMETRY_DTYPE),
                np.empty((0, 3), dtype=GEOMETRY_DTYPE),
                np.empty((0, 3), dtype=GEOMETRY_DTYPE),
                np.empty(0, dtype=LABEL_DTYPE),
                np.empty((0, 3), dtype=COLOR_DTYPE))

    # Precompute edges and structure data
    v0s = np.empty((N, 3), dtype=GEOMETRY_DTYPE)
    e1s = np.empty((N, 3), dtype=GEOMETRY_DTYPE) # v1 - v0
    e2s = np.empty((N, 3), dtype=GEOMETRY_DTYPE) # v2 - v0
    labels = np.empty(N, dtype=LABEL_DTYPE)
    colors = np.empty((N, 3), dtype=COLOR_DTYPE)

    for i, (v0, v1, v2) in enumerate(tris_geom):
        v0s[i] = v0
        e1s[i] = np.subtract(v1, v0) # Use numpy subtract for clarity
        e2s[i] = np.subtract(v2, v0)
        labels[i] = tris_labs[i]
        colors[i] = tris_cols[i]

    return v0s, e1s, e2s, labels, colors


# --- Ray-Triangle Intersection (CPU - Numba JIT) --------------------------
# This is the Möller–Trumbore algorithm. It inherently uses float ops.
@njit(fastmath=True) # Enable fastmath potentially unsafe optimizations
def intersect_ray_triangle_cpu(orig, dir, v0, e1, e2):
    """Möller–Trumbore intersection algorithm (CPU version)"""
    eps = GEOMETRY_DTYPE(1e-6) # Use defined dtype for epsilon
    # Calculate determinant components using cross product (dir x e2)
    h0 = dir[1] * e2[2] - dir[2] * e2[1]
    h1 = dir[2] * e2[0] - dir[0] * e2[2]
    h2 = dir[0] * e2[1] - dir[1] * e2[0]
    # Determinant = e1 . (dir x e2)
    a = e1[0] * h0 + e1[1] * h1 + e1[2] * h2

    # Check if ray is parallel to triangle plane or determinant is near zero
    if abs(a) < eps:
        return GEOMETRY_DTYPE(np.inf)

    f = GEOMETRY_DTYPE(1.0) / a
    s = orig - v0 # Vector from v0 to ray origin

    # Calculate u (barycentric coordinate) = f * (s . (dir x e2))
    u = f * (s[0] * h0 + s[1] * h1 + s[2] * h2)
    if u < 0.0 or u > 1.0:
        return GEOMETRY_DTYPE(np.inf)

    # Calculate v (barycentric coordinate) = f * (dir . (s x e1))
    # (s x e1) components
    q0 = s[1] * e1[2] - s[2] * e1[1]
    q1 = s[2] * e1[0] - s[0] * e1[2]
    q2 = s[0] * e1[1] - s[1] * e1[0]
    v = f * (dir[0] * q0 + dir[1] * q1 + dir[2] * q2)
    if v < 0.0 or u + v > 1.0:
        return GEOMETRY_DTYPE(np.inf)

    # Calculate t = f * (e2 . (s x e1))
    t = f * (e2[0] * q0 + e2[1] * q1 + e2[2] * q2)

    # Return t only if it's a forward intersection
    return t if t > eps else GEOMETRY_DTYPE(np.inf)

# --- Ray Tracing (CPU - Numba Parallel) -----------------------------------
# NOTE: Without an acceleration structure (like BVH), this will be slow
#       for large scenes, as every ray checks every triangle.
@njit(parallel=True, fastmath=True) # Enable parallel execution and fastmath
def raytrace_cpu(v0s, e1s, e2s, labels, colors,
                 cam_o, cam_dir, right, up,
                 screen_w, screen_h, W, H):
    """Performs ray tracing on the CPU using Numba for parallelization."""
    # Output arrays initialization
    rgb = np.zeros((H, W, 3), dtype=COLOR_DTYPE)
    depth = np.full((H, W), GEOMETRY_DTYPE(np.inf), dtype=DEPTH_DTYPE)
    sem = np.zeros((H, W), dtype=LABEL_DTYPE)
    pts = np.full((H, W, 3), GEOMETRY_DTYPE(np.nan), dtype=POINT_DTYPE)

    # Pre-calculate constants for the loop
    inv_W = GEOMETRY_DTYPE(1.0 / W)
    inv_H = GEOMETRY_DTYPE(1.0 / H)
    half_sw = GEOMETRY_DTYPE(0.5 * screen_w)
    half_sh = GEOMETRY_DTYPE(0.5 * screen_h)
    num_triangles = v0s.shape[0]

    # Loop over pixels in parallel
    for i in prange(H): # Use prange for parallel loops
        for j in range(W):
            # Calculate ray direction for the pixel center
            u = (j + GEOMETRY_DTYPE(0.5)) * inv_W - GEOMETRY_DTYPE(0.5)
            v = (i + GEOMETRY_DTYPE(0.5)) * inv_H - GEOMETRY_DTYPE(0.5) # Invert v for image coords? No, using up vector convention.

            # Direction = cam_dir + u * screen_w * right - v * screen_h * up
            # (Using +v because image coordinates usually start from top-left,
            #  but the formula uses typical graphics coordinates where +y is up)
            # Let's stick to the original formula: -v * up
            dir_x = cam_dir[0] + u * screen_w * right[0] - v * screen_h * up[0]
            dir_y = cam_dir[1] + u * screen_w * right[1] - v * screen_h * up[1]
            dir_z = cam_dir[2] + u * screen_w * right[2] - v * screen_h * up[2]

            # Normalize direction vector
            norm = (dir_x**2 + dir_y**2 + dir_z**2)**0.5
            # Avoid division by zero if norm is very small
            if norm < GEOMETRY_DTYPE(1e-9): norm = GEOMETRY_DTYPE(1.0)
            inv_norm = GEOMETRY_DTYPE(1.0) / norm
            d = np.array([dir_x * inv_norm, dir_y * inv_norm, dir_z * inv_norm], dtype=GEOMETRY_DTYPE)

            # Find the closest intersection
            tmin = GEOMETRY_DTYPE(np.inf)
            hit_idx = -1
            for k in range(num_triangles):
                t = intersect_ray_triangle_cpu(cam_o, d, v0s[k], e1s[k], e2s[k])
                if t < tmin:
                    tmin = t
                    hit_idx = k

            # If an intersection was found, record hit information
            if hit_idx >= 0:
                depth[i, j] = tmin
                sem[i, j] = labels[hit_idx]
                rgb[i, j, 0] = colors[hit_idx, 0]
                rgb[i, j, 1] = colors[hit_idx, 1]
                rgb[i, j, 2] = colors[hit_idx, 2]
                # Calculate intersection point: P = O + t * D
                pts[i, j, 0] = cam_o[0] + tmin * d[0]
                pts[i, j, 1] = cam_o[1] + tmin * d[1]
                pts[i, j, 2] = cam_o[2] + tmin * d[2]
            # else: pixel remains background (inf depth, 0 sem, nan point, 0 rgb)

    return rgb, depth, sem, pts

# --- GPU Kernels (Only if _GPU_AVAILABLE) ---------------------------------
if _GPU_AVAILABLE:

    # NOTE: This is the core intersection logic ported to CUDA device function.
    #       It still uses float32 (GEOMETRY_DTYPE). Attempting integer-only
    #       math here would be complex and likely slower due to lack of native
    #       support for the required geometric operations.
    @cuda.jit(device=True, inline=True) # Aggressive inlining hint
    def ray_tri_intersect_gpu(orig, dir, v0, e1, e2):
        """Möller–Trumbore intersection algorithm (CUDA Device Function)"""
        # Use explicit type for constants within device code if needed, though Numba often infers correctly
        eps = GEOMETRY_DTYPE(1e-6)
        inf = GEOMETRY_DTYPE(1e20) # Use a large float for infinity representation in CUDA kernel

        # Calculate determinant components (dir x e2)
        h0 = dir[1] * e2[2] - dir[2] * e2[1]
        h1 = dir[2] * e2[0] - dir[0] * e2[2]
        h2 = dir[0] * e2[1] - dir[1] * e2[0]
        a = e1[0] * h0 + e1[1] * h1 + e1[2] * h2 # Determinant

        if abs(a) < eps:
            return inf

        f = GEOMETRY_DTYPE(1.0) / a
        # s = orig - v0 (calculate components directly)
        s0 = orig[0] - v0[0]; s1 = orig[1] - v0[1]; s2 = orig[2] - v0[2]

        # Calculate u = f * (s . (dir x e2))
        u = f * (s0 * h0 + s1 * h1 + s2 * h2)
        if u < 0.0 or u > 1.0:
            return inf

        # Calculate v = f * (dir . (s x e1))
        # (s x e1) components
        q0 = s1 * e1[2] - s2 * e1[1]
        q1 = s2 * e1[0] - s0 * e1[2]
        q2 = s0 * e1[1] - s1 * e1[0]
        v = f * (dir[0] * q0 + dir[1] * q1 + dir[2] * q2)
        if v < 0.0 or u + v > 1.0:
            return inf

        # Calculate t = f * (e2 . (s x e1))
        t = f * (e2[0] * q0 + e2[1] * q1 + e2[2] * q2)

        # Check for positive t (intersection in front of ray origin)
        return t if t > eps else inf

    # NOTE: This CUDA kernel implements the same naive O(N*M) ray tracing.
    #       The primary speedup comes from massive parallelism of the GPU,
    #       not from algorithmic improvements (like BVH).
    @cuda.jit #(fastmath=True) # fastmath can sometimes be unstable in CUDA JIT
    def raytrace_cuda_kernel(v0s, e1s, e2s, labels, colors, # Scene data (device arrays)
                             cam_o, cam_dir, right, up,      # Camera params (device arrays/scalars)
                             scr_w, scr_h, W, H,             # Screen params (scalars)
                             rgb, depth, sem, pts):          # Output arrays (device arrays)
        """CUDA kernel for ray tracing one pixel per thread."""
        # Get thread indices for the pixel
        i, j = cuda.grid(2) # (row, col) or (y, x)

        # Check if thread indices are within the image bounds
        if i >= H or j >= W:
            return

        # Calculate ray direction (same logic as CPU version)
        # Use GEOMETRY_DTYPE for calculations inside the kernel
        u = (GEOMETRY_DTYPE(j) + GEOMETRY_DTYPE(0.5)) / GEOMETRY_DTYPE(W) - GEOMETRY_DTYPE(0.5)
        v = (GEOMETRY_DTYPE(i) + GEOMETRY_DTYPE(0.5)) / GEOMETRY_DTYPE(H) - GEOMETRY_DTYPE(0.5) # Corresponds to -v*up in world space

        # Direction vector components (using local array for potential register optimization)
        # Local array might not be necessary for simple vector math like this
        # d0_x = cam_dir[0] + u*scr_w*right[0] - v*scr_h*up[0]
        # d0_y = cam_dir[1] + u*scr_w*right[1] - v*scr_h*up[1]
        # d0_z = cam_dir[2] + u*scr_w*right[2] - v*scr_h*up[2]
        # Or use cuda.local.array if profiling shows register pressure:
        d0 = cuda.local.array(3, dtype=GEOMETRY_DTYPE)
        for k in range(3):
             d0[k] = cam_dir[k] + u * scr_w * right[k] - v * scr_h * up[k]

        # Normalize the direction vector
        # Use cuda.local.array for norm calculation if needed, but direct calc is fine
        nrm_sq = d0[0]**2 + d0[1]**2 + d0[2]**2
        if nrm_sq < GEOMETRY_DTYPE(1e-18): # Check for near-zero norm before sqrt
            # Handle degenerate case (e.g., set to a default direction or skip pixel)
            # For now, just prevent division by zero / sqrt of zero
             nrm = GEOMETRY_DTYPE(1.0)
        else:
            nrm = nrm_sq**0.5

        inv_nrm = GEOMETRY_DTYPE(1.0) / nrm
        d0[0] *= inv_nrm; d0[1] *= inv_nrm; d0[2] *= inv_nrm

        # --- Intersection Testing Loop ---
        # This is the O(M) part. For large M, this dominates runtime.
        # A BVH traversal would replace this loop.
        tmin = GEOMETRY_DTYPE(1e20) # Initialize with large float value
        hit = -1                   # Use -1 to indicate no hit initially
        num_triangles = v0s.shape[0]
        for k in range(num_triangles):
            # Call the device function for intersection test
            # Pass device array elements directly
            t = ray_tri_intersect_gpu(cam_o, d0, v0s[k], e1s[k], e2s[k])
            # Update minimum hit distance and index if a closer intersection is found
            if t < tmin:
                tmin = t
                hit = k
        # --- End Intersection Loop ---

        # If a triangle was hit (hit >= 0)
        if hit >= 0:
            # Write results to the output device arrays for this pixel (i, j)
            depth[i, j] = tmin
            sem[i, j] = labels[hit] # Read label from device array
            # Read color from device array and write to rgb output array
            rgb[i, j, 0] = colors[hit, 0]
            rgb[i, j, 1] = colors[hit, 1]
            rgb[i, j, 2] = colors[hit, 2]
            # Calculate intersection point P = O + t * D
            pts[i, j, 0] = cam_o[0] + d0[0] * tmin
            pts[i, j, 1] = cam_o[1] + d0[1] * tmin
            pts[i, j, 2] = cam_o[2] + d0[2] * tmin
        else:
            # No hit: Write background values
            # Depth already initialized to inf basically (1e20 here)
            depth[i, j] = GEOMETRY_DTYPE(np.inf) # Or keep 1e20, but np.inf might be clearer on host
            sem[i, j] = 0 # Assuming label 0 is background/unassigned
            # RGB defaults to 0 (black)
            # pts defaults to NaN (already initialized on host, not explicitly set here, but could be)
            # Optional: Explicitly set background values if needed
            # rgb[i, j, 0] = 0; rgb[i, j, 1] = 0; rgb[i, j, 2] = 0
            # pts[i, j, 0] = GEOMETRY_DTYPE(np.nan); pts[i, j, 1] = GEOMETRY_DTYPE(np.nan); pts[i, j, 2] = GEOMETRY_DTYPE(np.nan)


# --- OBJ Export Function (Keep as is) -------------------------------------
def save_combined_obj(filename, v0s, e1s, e2s,
                      cam_o, cam_dir, right, up, screen_w, screen_h, # Added screen params
                      pts, far):
    """Saves scene triangles, camera frustum, and hit points to OBJ"""
    with open(filename, 'w') as f:
        f.write(f'# Raytracer output: {datetime.now()}\n')
        f.write('# Scene triangles\n')
        vidx = 1 # OBJ indices start from 1

        # Write triangle vertices and faces
        for i in range(v0s.shape[0]):
            v0 = v0s[i]
            v1 = v0 + e1s[i]
            v2 = v0 + e2s[i]
            f.write(f"v {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            f.write(f"v {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"v {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write(f"f {vidx}// {vidx+1}// {vidx+2}//\n") # Using // for faces without texture/normals
            vidx += 3

        # Write camera frustum lines
        f.write('\n# Camera frustum\n')
        f.write(f"o camera_frustum\n")
        f.write(f"v {cam_o[0]:.6f} {cam_o[1]:.6f} {cam_o[2]:.6f}\n") # Camera origin (index vidx)
        cam_v_start = vidx
        vidx += 1

        corners = []
        # Define frustum corners at 'far' distance
        for du in [-0.5, 0.5]: # u ranges from -0.5 to 0.5
            for dv in [-0.5, 0.5]: # v ranges from -0.5 to 0.5
                # Direction = cam_dir + u * screen_w * right - v * screen_h * up
                d = cam_dir + (du * screen_w * right) - (dv * screen_h * up)
                d /= np.linalg.norm(d) # Normalize
                corner_pt = cam_o + d * far
                corners.append(corner_pt)
                f.write(f"v {corner_pt[0]:.6f} {corner_pt[1]:.6f} {corner_pt[2]:.6f}\n")
                vidx += 1

        # Indices of the far plane corners (relative to start of frustum vertices)
        # Order: bottom-left, top-left, bottom-right, top-right (if dv=-0.5 -> bottom)
        # du=-0.5, dv=-0.5 -> idx 1 (bottom-left)
        # du=-0.5, dv= 0.5 -> idx 2 (top-left)
        # du= 0.5, dv=-0.5 -> idx 3 (bottom-right)
        # du= 0.5, dv= 0.5 -> idx 4 (top-right)
        bl = cam_v_start + 1
        tl = cam_v_start + 2
        br = cam_v_start + 3
        tr = cam_v_start + 4

        # Lines from camera origin to corners
        f.write(f"l {cam_v_start} {bl}\n")
        f.write(f"l {cam_v_start} {tl}\n")
        f.write(f"l {cam_v_start} {br}\n")
        f.write(f"l {cam_v_start} {tr}\n")
        # Lines for the far plane rectangle
        f.write(f"l {bl} {tl}\n")
        f.write(f"l {tl} {tr}\n")
        f.write(f"l {tr} {br}\n")
        f.write(f"l {br} {bl}\n")

        # Write intersection points (as 'v') and optionally lines from camera ('l')
        # Writing millions of points/lines can make the OBJ huge and slow to load.
        # Consider sampling points or only writing points, not lines.
        f.write('\n# Intersection points (sampled)\n')
        f.write(f"o intersection_points\n")
        H, W = pts.shape[:2]
        step = max(1, H // 64, W // 64) # Sample points to reduce file size
        point_v_start = vidx
        num_pts_written = 0
        for i in range(0, H, step):
            for j in range(0, W, step):
                p = pts[i, j]
                # Check if the point is valid (not NaN)
                if not np.isnan(p[0]):
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                    # Optional: line from camera to point (makes file huge)
                    # f.write(f"l {cam_v_start} {vidx}\n")
                    vidx += 1
                    num_pts_written +=1

        # Optionally group points if supported by viewer
        if num_pts_written > 0:
             f.write(f"g hit_points\n")
             f.write(f"p {' '.join(map(str, range(point_v_start, vidx)))}\n") # 'p' for point group


    print(f"    Saved scene, frustum, and {num_pts_written} sampled points to {filename}")

# --- Main Execution Flow --------------------------------------------------
def main():
    global _GPU_AVAILABLE # <--- 添加这一行

    total_t0 = _now()
    # --- Output Directory Setup ---
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outd = os.path.join('output', ts)
    os.makedirs(outd, exist_ok=True)
    print(f"Output directory: {outd}")

    # --- Scene Generation ---
    print('[1/6] Building scene...')
    t0 = _now()
    v0s, e1s, e2s, labels, colors = build_scene()
    num_triangles = len(v0s)
    log_step('Scene building', t0)
    if num_triangles == 0:
        print("Error: Scene construction resulted in 0 triangles. Exiting.")
        return
    print(f"    Scene contains {num_triangles} triangles.")

    # --- Camera Setup ---
    print('[2/6] Setting up camera...')
    # Use global scope variables carefully; passing as arguments is often clearer
    # For simplicity here, we keep the global approach from the original code
    # global cam_o, cam_dir, right, up, screen_w, screen_h
    t0 = _now()
    cam_o = np.array([-30, 20, -85], dtype=GEOMETRY_DTYPE) # Adjusted Z slightly
    cam_t = np.array([20, 10, 0], dtype=GEOMETRY_DTYPE)  # Look more towards center/lower
    cam_up_vec = np.array([0, 1, 0], dtype=GEOMETRY_DTYPE) # World up vector

    cam_dir = cam_t - cam_o
    cam_dir /= np.linalg.norm(cam_dir)

    right = np.cross(cam_dir, cam_up_vec)
    # Handle case where cam_dir is aligned with cam_up_vec (looking straight up/down)
    if np.linalg.norm(right) < 1e-6:
        # If looking straight up/down, 'right' could be arbitrary (e.g., world X)
        # Recompute 'up' accordingly
        print("Warning: Camera looking straight up/down. Adjusting right vector.")
        right = np.array([1, 0, 0], dtype=GEOMETRY_DTYPE) # Assume world X is right
        # Ensure cam_up is orthogonal to new cam_dir
        cam_dir_flat = np.array([cam_dir[0], 0, cam_dir[2]], dtype=GEOMETRY_DTYPE)
        if np.linalg.norm(cam_dir_flat) > 1e-6: # If not perfectly vertical
            cam_dir_flat /= np.linalg.norm(cam_dir_flat)
            right = np.cross(cam_dir, cam_up_vec) # Re-calculate based on non-vertical part
            right /= np.linalg.norm(right)
            up = np.cross(right, cam_dir) # Recalculate up based on right/dir
        else: # Perfectly vertical view
            up = np.cross(right, cam_dir) # Up will be orthogonal to right and dir
    else:
        right /= np.linalg.norm(right)
        up = np.cross(right, cam_dir) # Recalculate up to ensure orthogonality
        # No need to normalize 'up' if 'right' and 'cam_dir' are normalized unit vectors

    # Image dimensions
    W, H = 2048, 1024
    # Field of View (Vertical FOV)
    fov_degrees = 60.0
    fov_radians = np.deg2rad(fov_degrees)

    # Screen height in world space at distance 1 from camera
    # tan(fov_rad / 2) = (screen_h / 2) / 1 => screen_h = 2 * tan(fov_rad / 2)
    screen_h = GEOMETRY_DTYPE(2.0 * np.tan(fov_radians / 2.0))
    # Screen width based on aspect ratio
    aspect_ratio = W / H
    screen_w = GEOMETRY_DTYPE(screen_h * aspect_ratio)

    log_step('Camera setup', t0)
    print(f"    Resolution: {W}x{H}, FoV: {fov_degrees} deg")
    print(f"    Cam Pos: {cam_o}, Target: {cam_t}")
    print(f"    Cam Dir: {cam_dir}")
    print(f"    Cam Right: {right}")
    print(f"    Cam Up: {up}")

    # --- Ray Tracing ---
    print('[3/6] Ray tracing...')
    t0 = _now()
    rgb, depth, sem_lbl, pts = None, None, None, None # Initialize results

    if _GPU_AVAILABLE:
        print("    Using GPU (CUDA)...")
        try:
            # --- Prepare Data for GPU ---
            t_upload_start = _now()
            # Scene Geometry (already numpy arrays with correct dtype)
            d_v0s    = cuda.to_device(v0s)
            d_e1s    = cuda.to_device(e1s)
            d_e2s    = cuda.to_device(e2s)
            d_labels = cuda.to_device(labels)
            d_colors = cuda.to_device(colors)

            # Camera Parameters (copy individual vectors/scalars)
            d_cam_o   = cuda.to_device(cam_o)
            d_cam_dir = cuda.to_device(cam_dir)
            d_right   = cuda.to_device(right)
            d_up      = cuda.to_device(up)

            # Output Buffers (allocate on GPU)
            d_rgb    = cuda.device_array((H, W, 3), dtype=COLOR_DTYPE)
            d_depth  = cuda.device_array((H, W), dtype=DEPTH_DTYPE)
            d_sem    = cuda.device_array((H, W), dtype=LABEL_DTYPE)
            d_pts    = cuda.device_array((H, W, 3), dtype=POINT_DTYPE)
            # Initialize depth and points on GPU? Usually done implicitly or via kernel logic.
            # We handle non-hits within the kernel.
            log_step('GPU data upload', t_upload_start)

            # --- Kernel Launch Configuration ---
            # Try adjusting threads per block
            threads_per_block = (16, 16) # Common starting point (256 threads/block)
            # threads_per_block = (32, 32) # Max threads per block (1024), potentially faster if registers allow
            # threads_per_block = (8, 8)   # Smaller block size (64 threads/block)

            blocks_per_grid_x = (W + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid_y = (H + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x) # Order is (Y, X) or (i, j)

            print(f"    Launching CUDA kernel with {blocks_per_grid} blocks, {threads_per_block} threads/block")

            # --- Execute Kernel ---
            t_kernel_start = _now()
            raytrace_cuda_kernel[blocks_per_grid, threads_per_block](
                d_v0s, d_e1s, d_e2s, d_labels, d_colors, # Geometry
                d_cam_o, d_cam_dir, d_right, d_up,       # Camera
                GEOMETRY_DTYPE(screen_w), GEOMETRY_DTYPE(screen_h), # Screen params (pass as value)
                np.int32(W), np.int32(H),                # Dimensions (pass as value)
                d_rgb, d_depth, d_sem, d_pts)            # Output buffers
            cuda.synchronize() # Wait for kernel to finish before proceeding
            log_step('GPU kernel execution', t_kernel_start)

            # --- Download Results from GPU ---
            t_download_start = _now()
            rgb     = d_rgb.copy_to_host()
            depth   = d_depth.copy_to_host()
            sem_lbl = d_sem.copy_to_host()
            pts     = d_pts.copy_to_host()
            log_step('GPU data download', t_download_start)

        except Exception as e:
            print(f"\n---!! GPU execution failed: {e} !!---")
            print("---!! Falling back to CPU execution. !!---\n")
            _GPU_AVAILABLE = False # Prevent further GPU attempts

    # --- CPU Execution (Fallback or if no GPU) ---
    if not _GPU_AVAILABLE:
        print("    Using CPU (Numba JIT)...")
        t_cpu_start = _now()
        # Pass screen parameters and dimensions explicitly
        rgb, depth, sem_lbl, pts = raytrace_cpu(
            v0s, e1s, e2s, labels, colors,
            cam_o, cam_dir, right, up,
            screen_w, screen_h, W, H)
        log_step('CPU raytrace execution', t_cpu_start)

    log_step('Total ray tracing', t0) # Log total time for the section

    # --- Image Export ---
    print('[4/6] Saving output images...')
    t0 = _now()
    if rgb is not None:
        fn_view = os.path.join(outd, f'view_{ts}.png')
        Image.fromarray(rgb).save(fn_view)
        print(f"    Saved view: {fn_view}")
    else:
        print("    Skipping view saving (ray tracing failed?).")

    if depth is not None:
        fn_depth = os.path.join(outd, f'depth_{ts}.png')
        # Normalize depth map for visualization (handle potential inf values)
        valid_depth = depth[np.isfinite(depth)]
        if len(valid_depth) > 0:
            dmin, dmax = np.min(valid_depth), np.max(valid_depth)
            print(f"    Depth range (finite): {dmin:.2f} to {dmax:.2f}")
            # Clamp depth for visualization if max is too large
            vis_dmax = min(dmax, 300.0) # Clamp visualization max depth if needed
            # Normalize to 0-254, use 255 for infinite/no hit
            dmap = np.where(np.isfinite(depth),
                            np.clip((depth - dmin) / (vis_dmax - dmin + 1e-6) * 254, 0, 254),
                            255).astype(np.uint8)
            Image.fromarray(dmap, 'L').save(fn_depth)
            print(f"    Saved depth map: {fn_depth}")
        else:
            print("    Skipping depth saving (no valid depth values).")
            # Create a black image as placeholder?
            Image.new('L', (W, H), 0).save(fn_depth)


    if sem_lbl is not None:
        fn_sem = os.path.join(outd, f'semantic_{ts}.png')
        # Create color palette matching the scene build colors + background
        palette = {
            0: (0, 0, 0),            # Background/Default
            1: C[1].tolist(),        # Building
            2: C[2].tolist(),        # Road
            3: C[3].tolist(),        # Water
            4: C[4].tolist(),        # Skyscraper/Special
            5: C[5].tolist(),        # Tree
            6: C[6].tolist()         # Mountain
            # Add other labels if used
        }
        # Create RGB image from labels and palette
        sem_img = np.zeros_like(rgb, dtype=COLOR_DTYPE) # Use COLOR_DTYPE
        for label_id, color in palette.items():
            mask = (sem_lbl == label_id)
            sem_img[mask] = color
        Image.fromarray(sem_img).save(fn_sem)
        print(f"    Saved semantic map: {fn_sem}")
    else:
         print("    Skipping semantic map saving (ray tracing failed?).")

    log_step('Image export', t0)

    # --- OBJ Export ---
    print('[5/6] Saving combined OBJ file...')
    t0 = _now()
    fn_obj = os.path.join(outd, f'combined_{ts}.obj')
    # Determine a suitable 'far' distance for the frustum visualization
    far_dist = 150.0 # Default far distance
    if depth is not None:
        valid_depth = depth[np.isfinite(depth)]
        if len(valid_depth) > 0:
            far_dist = max(far_dist, np.max(valid_depth) * 1.1) # Extend beyond max hit

    if pts is not None:
         save_combined_obj(fn_obj, v0s, e1s, e2s,
                           cam_o, cam_dir, right, up, screen_w, screen_h, # Pass screen params
                           pts, far=far_dist)
    else:
         print("    Skipping OBJ saving (ray tracing failed?).")

    log_step('OBJ export', t0)

    # --- Completion ---
    print('[6/6] Finished.')
    total_time = _now() - total_t0
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Output saved to: {outd}")


if __name__ == '__main__':
    main()