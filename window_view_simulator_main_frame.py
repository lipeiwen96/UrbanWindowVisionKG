# window_view_simulator_main.py
import os
import json
import time
import uuid # 用于生成唯一 ID
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Polygon, MultiPolygon
from shapely.errors import WKTReadingError, ShapelyError
import trimesh
import mapbox_earcut
from modules.utils import log_time, load_json_data
from art_public_modules.art_data_exchange.data_exchange import ARTShapelyDataExchanger
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from modules.scene_structure import Scene, Object, MeshElement
from modules.renderer import Renderer, COLOR_DTYPE, GEOMETRY_DTYPE   # Import the new Renderer class
from modules.restore_elements import RestoreElements


# --- Constants and Paths (Keep as before) ---
RAW_MODEL_JSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "processed", "project", "map_model.json"))
SIMPLIFIED_MODEL_JSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "processed", "project", "project_simplified_only_crv.json"))
NORTH_MOUNTAINS_OBJ_PATH = os.path.abspath(os.path.join("library", "HK_map", "processed", "project", "北侧地形.obj"))
SOUTH_MOUNTAINS_OBJ_PATH = os.path.abspath(os.path.join("library", "HK_map", "processed", "project", "南侧地形.obj"))
OUTPUT_DIR = os.path.abspath(os.path.join("output", "processed_scene"))
RENDER_OUTPUT_DIR = os.path.abspath(os.path.join("output", "render_results")) # New dir for renders
CACHE_DIR = os.path.abspath(os.path.join("output", "cache"))  # 缓存目录
FLATTENED_DATA_NPZ_PATH = os.path.join(CACHE_DIR, "flattened_geometry.npz")
FLATTENED_METADATA_JSON_PATH = os.path.join(CACHE_DIR, "flattened_metadata.json")
PROCESSED_MODEL_JSON_PATH = os.path.abspath(os.path.join("library", "HK_map", "processed", "project", "processed_model.json")) # 重命名以便清晰


# --- 确保输出和缓存目录存在 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RENDER_OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def load_flattened_cache(npz_path: str, json_path: str) -> Optional[Dict[str, Any]]:
    """尝试从缓存文件加载展平后的几何数据。"""
    if os.path.exists(npz_path) and os.path.exists(json_path):
        print(f"  检测到缓存文件，尝试加载: {npz_path}, {json_path}")
        try:
            t0_load = time.perf_counter()
            with np.load(npz_path) as data:
                loaded_data = {key: data[key] for key in data}
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # JSON 加载后，字典的键可能变成字符串，需要转换回来（如果label_map/color_map的键是整数）
            # 注意：我们的 label_map/color_map 键是整数 label_id
            label_map_fixed = {int(k): v for k, v in metadata['label_map'].items()}
            # 确保 color_map 的值是 numpy 数组
            color_map_fixed = {int(k): np.array(v, dtype=COLOR_DTYPE) for k, v in metadata['color_map'].items()} # 颜色需要转回 numpy 数组

            loaded_data['label_map'] = label_map_fixed
            loaded_data['color_map'] = color_map_fixed

            log_time(f"成功加载缓存", t0_load)
            # 验证一下基本数据是否存在
            required_keys = ['v0s', 'e1s', 'e2s', 'normals', 'labels', 'colors', 'label_map', 'color_map']
            if all(key in loaded_data for key in required_keys):
                 return loaded_data
            else:
                 print("  缓存数据不完整，将重新生成。")
                 return None
        except Exception as e:
            print(f"  加载缓存失败: {e}。将重新生成。")
            # 加载失败时，最好把损坏的缓存文件删掉
            try: os.remove(npz_path)
            except OSError: pass
            try: os.remove(json_path)
            except OSError: pass
            return None
    else:
        print("  未找到缓存文件。")
        return None


def save_flattened_cache(data: Dict[str, Any], npz_path: str, json_path: str):
    """将展平后的几何数据保存到缓存文件。"""
    print(f"  保存 Flattened Geometry 缓存到: {npz_path}, {json_path}")
    try:
        t0_save = time.perf_counter()
        # 分离 NumPy 数组和元数据
        arrays_to_save = {
            'v0s': data['v0s'],
            'e1s': data['e1s'],
            'e2s': data['e2s'],
            'normals': data['normals'],
            'labels': data['labels'],
            'colors': data['colors'],
        }
        metadata_to_save = {
            # JSON 不能直接存 numpy 数组，需要先转 list
            'label_map': data['label_map'],
            # color_map 的值已经是 numpy 数组，需要转 list 存储
            'color_map': {k: v.tolist() for k, v in data['color_map'].items()} # 转换颜色为 list
        }

        # 保存 NPZ (压缩以节省空间)
        np.savez_compressed(npz_path, **arrays_to_save)
        # 保存 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, indent=4)

        log_time("成功保存缓存", t0_save)
    except Exception as e:
        print(f"  保存缓存失败: {e}")


if __name__ == "__main__":
    # RestoreElements.run(RAW_MODEL_JSON_PATH, SIMPLIFIED_MODEL_JSON_PATH, PROCESSED_MODEL_JSON_PATH)

    main_start_time = time.perf_counter()
    print("--- Window View Simulator 主程序 ---")

    # --- 步骤 1: 尝试从缓存加载 Flattened Geometry ---
    print(f"\n步骤 1: 尝试加载 Flattened Geometry 缓存...")
    flattened_data = load_flattened_cache(FLATTENED_DATA_NPZ_PATH, FLATTENED_METADATA_JSON_PATH)

    # --- 步骤 2: 如果缓存加载失败，则从源文件构建场景并 Flatten ---
    if flattened_data is None:
        print("\n步骤 2: 缓存未命中或无效，从源文件加载并处理场景...")
        scene_build_start = time.perf_counter()

        # 2a. 加载处理后的模型 JSON
        print(f"  -> 加载模型 JSON: {PROCESSED_MODEL_JSON_PATH}")
        if not os.path.exists(PROCESSED_MODEL_JSON_PATH):
            print(f"错误: 输入文件未找到 {PROCESSED_MODEL_JSON_PATH}")
            exit()
        try:
            map_processed_model = ARTShapelyDataExchanger.read_json_file(PROCESSED_MODEL_JSON_PATH)
            model_elements = map_processed_model.elements
            print(f"     找到 {len(model_elements)} 个元素。")
        except Exception as e:
            print(f"错误: 加载 JSON 文件失败: {e}")
            exit()

        # 2b. 创建三维场景并加载数据
        print("  -> 创建 Scene 对象并加载数据...")
        scene = Scene()
        WINDOW_METADATA_OUTPUT_PATH = os.path.join(CACHE_DIR, "window.json")
        TEST_WINDOW_AOI_WKT = "POLYGON ((839300 815600, 839300 823400, 831200 823400, 831200 815600, 839300 815600))"
        scene.load_elements(model_elements,
                            # --- 传递窗户生成参数 ---
                            generate_windows_flag=True,  # 启用窗户生成
                            window_aoi_wkt=TEST_WINDOW_AOI_WKT,  # 使用测试 AOI
                            window_metadata_output_path=WINDOW_METADATA_OUTPUT_PATH)  # 加载元素

        TREE_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "greening", "tree20250203_converted.json"))
        MORE_TREE_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "greening", "VIS_INV_TREE_CSDI_202503171731_converted.json"))
        MOREMORE_TREE_PATH = os.path.abspath(os.path.join("library", "HK_map", "row_map", "greening", "dataset_29_layer.json"))

        scene.plan_trees(TREE_PATH)
        scene.plan_trees_v2(MORE_TREE_PATH)
        scene.plan_trees_v3(MOREMORE_TREE_PATH)

        # 加载地形 OBJ
        if os.path.exists(NORTH_MOUNTAINS_OBJ_PATH):
            print(f"  -> 加载北侧地形: {NORTH_MOUNTAINS_OBJ_PATH}")
            scene.load_obj(NORTH_MOUNTAINS_OBJ_PATH, "MountainNorth")
        else:
            print(f"警告: 找不到北侧地形 OBJ: {NORTH_MOUNTAINS_OBJ_PATH}")
        if os.path.exists(SOUTH_MOUNTAINS_OBJ_PATH):
            print(f"  -> 加载南侧地形: {SOUTH_MOUNTAINS_OBJ_PATH}")
            scene.load_obj(SOUTH_MOUNTAINS_OBJ_PATH, "MountainSouth")
        else:
            print(f"警告: 找不到南侧地形 OBJ: {SOUTH_MOUNTAINS_OBJ_PATH}")

        log_time(
            f"完成场景创建 (共 {len(scene.objects)} 对象, {scene.total_vertices} 顶点, {scene.total_mesh_elements} 面)",
            scene_build_start)

        # 2c. 执行 Flatten Geometry
        print("\n步骤 3: 执行 Flatten Geometry...")
        flatten_start = time.perf_counter()
        if scene.objects:  # 仅当场景非空时才 flatten
            flattened_data = scene.flatten_geometry()
            log_time("完成几何体 Flattening", flatten_start)

            # 2d. 如果 Flatten 成功，保存缓存
            if flattened_data:
                save_flattened_cache(flattened_data, FLATTENED_DATA_NPZ_PATH, FLATTENED_METADATA_JSON_PATH)
            else:
                print("  错误: Flattening 失败，无法生成缓存。")
                # 如果 flatten 失败，flattened_data 仍然是 None
        else:
            print("  场景为空，跳过 Flattening。")
            # flattened_data 保持为 None

    else:  # 缓存加载成功
        print("\n步骤 2 & 3: 成功从缓存加载 Flattened Geometry，跳过场景加载和 Flattening。")
        # 注意：此时没有 `scene` 对象，只有 `flattened_data`

    # --- [NEW] Rendering Step ---
    if flattened_data:  # Only render if the scene has objects
        print("\n步骤 4: 创建 Renderer 并准备场景 (上传GPU)...")
        prepare_start = time.perf_counter()
        renderer = Renderer()  # Create instance
        if not renderer.prepare_scene(flattened_data):  # Prepare the scene
            print("错误: 场景准备失败，无法渲染。")
            exit()
        log_time("完成场景准备和GPU上传", prepare_start)

        # --- 步骤 5: 定义相机和渲染参数的基础模板 ---
        print("\n步骤 5: 定义相机和渲染参数模板...")
        # Example Camera (use your desired camera)
        camera_params = {
            "origin": [833870, 816620, 200],  # Viewpoint position (x, y, z) - Y is UP
            "target": [836000, 818000, 200],  # Look-at point (x, y, z)
            # "origin": [835900, 818283, 60],  # Viewpoint position (x, y, z) - Y is UP
            # "target": [836500, 819000, 60],  # Look-at point (x, y, z)
            # "origin": [834882, 819119, 30],  # Viewpoint position (x, y, z) - Y is UP
            # "target": [834403, 818948, 30],  # Look-at point (x, y, z)
            "up_vector": [0, 0, 1],  # World Up direction
            "fov_deg": 65.0,  # 垂直视场角 (degrees) - 增加一点以获得更宽广的视野
            "width": 3840,  # Output image width
            "height": 2160  # Output image height
        }

        # --- Define Rendering Settings ---
        base_render_params = {
            # --- 基本设置 ---
            "output_dir": RENDER_OUTPUT_DIR,
            "file_prefix": f"view_{camera_params['origin'][0]:.0f}_{camera_params['origin'][1]:.0f}_{camera_params['origin'][2]:.0f}",

            # --- 光照参数 ---
            # 太阳方向: 从哪个方向照射过来 (x, y, z) - 需要归一化
            # [-1, -1, -1] 表示从左后上方照射
            # [ 0,  0, -1] 表示从正上方照射
            "sun_direction": [0.577, -0.577, -0.577],  # 保持归一化 [-1,-1,-1] -> [~-0.577, ~-0.577, ~-0.577]
                                                      # [0, -1, -1] -> [0, -0.707, -0.707]
            # *** 修改默认光照参数以增强对比度 ***
            "sun_intensity": GEOMETRY_DTYPE(6.0),   # 稍微增加太阳光强度 (之前是 4.0)
            "ambient_light": GEOMETRY_DTYPE(0.1),  # 显著降低默认环境光 (之前是 0.1)
            "specular_color": [1.0, 1.0, 1.0],    # 高光颜色 (通常为白色)
            "specular_exponent": GEOMETRY_DTYPE(64.0), # 高光锐利度
            "ks": GEOMETRY_DTYPE(0.5),            # 镜面反射系数

            # --- 阴影参数 ---
            # 注意：shadow_bias 是参数扫描的关键，这里的默认值仅在不扫描时使用
            # renderer.py 中的实现使用 shadow_bias 来偏移阴影光线起点
            # 不需要 shadow_intensity 参数，阴影是通过光线追踪的有无来决定的
            "shadow_bias": GEOMETRY_DTYPE(4e-4),    # 阴影偏移，防止自遮挡。这是参数扫描的起点。

             # --- [新] 雾效参数 ---
             # 注意：雾效仅影响 RGB 图像，不影响深度图和语义图
            "fog_enabled": True,                   # 是否启用雾效 (True/False)
            "fog_color": [210, 225, 240],        # 雾的颜色 (RGB, 0-255) - 浅蓝色调
            "fog_density": GEOMETRY_DTYPE(0.00001), # 雾的浓度 (值越小雾越淡，指数衰减)

            # --- 其他渲染参数 ---
            "sky_color": [180, 210, 255],         # 天空颜色 (RGB, 0-255) - 更鲜艳的天空蓝
            # "max_distance": 10000.0              # 可选: 最大渲染距离

            # --- NEW Control Params ---
            "shading_mode": "simple_diffuse", # Choose 'phong' or 'simple_diffuse'
            "save_outputs": ["rgb", "params", "depth", "semantic", "points"] # List: "rgb", "depth", "semantic", "points", "params"
                                              # Default if omitted: ["rgb"]
        }

        result = renderer.render(camera_params, base_render_params)
        #
        # # ------------------------------------------------------------------
        # # 步骤 6：多参数网格 / 随机抽样大规模渲染
        # # ------------------------------------------------------------------
        # import random, math
        # print("\n步骤 6: 大规模参数扫查渲染…")
        # MAX_FRAMES = 1000  # 目标帧数
        # SAVE_MOD = 50  # 每隔多少帧把 RGB 之外的结果也落盘
        #
        # # ---------- 1. 定义候选参数范围 ----------
        # # 太阳方向：用球坐标 (azimuth°, elevation°) → 单位向量
        # AZIMUTH_DEGS = np.linspace(0, 330, 12)  # 0,30,60,…,330 (方位角)
        # ELEVATION_DEGS = np.linspace(-10, -70, 10)  # -10,-16.6,…,-70 (俯仰角, 负值表示在水平面以下)
        # # *** 阴影偏移 (Shadow Bias) 的扫描范围 ***
        # # 这是解决阴影问题的关键参数。当前范围 1e-4 到 5e-4。
        # # 如果扫描结果显示在 5e-4 时阴影仍然有明显悬浮 (Peter Panning)，
        # # 或者在 1e-4 时仍有严重自遮挡 (Shadow Acne)，则需要调整此范围。
        # # 例如，尝试更大的范围: np.geomspace(1e-4, 1e-3, 8) 或 np.geomspace(5e-5, 5e-4, 8)
        # BIAS_LIST = np.geomspace(1e-4, 5e-4, 8)  # 对数间隔 (保持不变，但强调其重要性)
        # # 环境光扫描范围
        # AMBIENT_LIST = [0.03, 0.05, 0.07, 0.1]  # 可自行增删 (保持不变)
        #
        # # 可选：打乱顺序随机抽样，避免集中失败
        # param_pool = []
        # for azi in AZIMUTH_DEGS:
        #     for ele in ELEVATION_DEGS:
        #         # 球坐标转方向向量 (Sun -> Scene) - 注意坐标系和角度定义
        #         # 假设: azimuth=0 沿 Y+ 轴, elevation=0 在 XY 平面, elevation=-90 沿 Z- 轴
        #         # 则: x = cos(el) * sin(az), y = cos(el) * cos(az), z = sin(el)
        #         az, el = np.deg2rad(azi), np.deg2rad(ele)
        #         # 计算指向光源的方向 (从场景指向太阳)
        #         dir_to_sun = np.array([
        #             math.cos(el) * math.sin(az),
        #             math.cos(el) * math.cos(az),
        #             math.sin(el)
        #         ], dtype=GEOMETRY_DTYPE)
        #         # 内核需要的是从太阳射向场景的方向
        #         sun_dir = -dir_to_sun # 取反
        #         # 确保归一化 (理论上已经是单位向量，但以防万一)
        #         norm = np.linalg.norm(sun_dir)
        #         if norm > 1e-6:
        #             sun_dir /= norm
        #         else: # 如果计算出零向量，则使用一个默认方向，例如垂直向下
        #             sun_dir = np.array([0, 0, -1], dtype=GEOMETRY_DTYPE)
        #
        #         for bias in BIAS_LIST:
        #             for amb in AMBIENT_LIST:
        #                 param_pool.append((sun_dir, bias, amb))
        #
        # random.shuffle(param_pool)  # 随机顺序
        # param_pool = param_pool[:MAX_FRAMES]  # 截取 1000 组
        #
        # # ---------- 2. 开始循环 ----------
        # all_success = 0
        # for idx, (sun_dir, bias_val, amb_val) in enumerate(param_pool, 1):
        #     t_iter = time.perf_counter()
        #     print(f"\n--- 迭代 {idx}/{len(param_pool)} | bias={bias_val:.1E} | "
        #           f"ambient={amb_val:.02f} | sun=({sun_dir[0]:+.2f},{sun_dir[1]:+.2f},{sun_dir[2]:+.2f})")
        #
        #     # ----- 复制基础参数并更新 -----
        #     render_p = base_render_params.copy()
        #     render_p.update({
        #         "sun_direction": sun_dir.tolist(),  # 必须 list 才能序列化
        #         "shadow_bias": GEOMETRY_DTYPE(bias_val), # 使用扫描值
        #         "ambient_light": GEOMETRY_DTYPE(amb_val), # 使用扫描值
        #         "shading_mode": "phong", # 或者根据需要设置为 "simple_diffuse"
        #         "file_prefix": f"sweep_{idx:04d}_bias{bias_val:.1E}_amb{amb_val:.2f}", # 文件名包含参数
        #         # 控制保存内容，减少磁盘占用，只在必要时保存深度等
        #         "save_outputs": ["rgb", "params"] if idx % SAVE_MOD else ["rgb", "params", "depth", "semantic"]
        #          # "save_outputs": ["rgb"] # 只保存 RGB 以加快速度，如果需要分析其他输出则取消注释上一行
        #     })
        #
        #     # ----- 渲染 -----
        #     result = renderer.render(camera_params, render_p)
        #     if result and result["rgb"] is not None:
        #         all_success += 1
        #         print(f"    ✓ 成功 (累计 {all_success})")
        #     else:
        #         print("    ✗ 失败")
        #
        #     log_time("本帧耗时", t_iter)
        #
        # print(f"\n--- 完成。共尝试 {len(param_pool)} 帧，成功 {all_success} 帧 ---")
        # print(f"--- 请务必检查 {RENDER_OUTPUT_DIR} 目录下的 sweep_*.png 图片 ---")
        # print(f"--- 特别关注不同 shadow_bias 值对阴影（地面和建筑表面）的影响 ---")

    # --- End ---
    main_end_time = time.perf_counter()
    print(f"\n--- 主程序完成 | 总耗时: {main_end_time - main_start_time:.2f}s ---")