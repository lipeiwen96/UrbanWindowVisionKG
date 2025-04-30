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
        scene.load_elements(model_elements)  # 加载元素

        TREE_PATH = r"E:\Code\HITSZ\UrbanWindowVisionKG\library\HK_map\row_map\greening\tree20250203_converted.json"
        MORE_TREE_PATH = r"E:\Code\HITSZ\UrbanWindowVisionKG\library\HK_map\row_map\greening\VIS_INV_TREE_CSDI_202503171731_converted.json"
        MOREMORE_TREE_PATH = r"E:\Code\HITSZ\UrbanWindowVisionKG\library\HK_map\row_map\greening\dataset_29_layer.json"
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
            # "origin": [833870, 816620, 200],  # Viewpoint position (x, y, z) - Y is UP
            # "target": [836000, 818000, 200],  # Look-at point (x, y, z)
            # "origin": [835900, 818283, 60],  # Viewpoint position (x, y, z) - Y is UP
            # "target": [836500, 819000, 60],  # Look-at point (x, y, z)
            "origin": [834882, 819119, 30],  # Viewpoint position (x, y, z) - Y is UP
            "target": [834403, 818948, 30],  # Look-at point (x, y, z)
            "up_vector": [0, 0, 1],  # World Up direction
            "fov_deg": 65.0,  # 垂直视场角 (degrees) - 增加一点以获得更宽广的视野
            "width": 1920,  # Output image width
            "height": 1080  # Output image height
        }

        # --- Define Rendering Settings ---
        base_render_params = {
            # --- 基本设置 ---
            "output_dir": RENDER_OUTPUT_DIR,
            "file_prefix": f"view_{camera_params['origin'][0]:.0f}_{camera_params['origin'][1]:.0f}_{camera_params['origin'][2]:.0f}",

            # --- 光照参数 ---
            # 太阳方向: 从哪个方向照射过来 (x, y, z)
            # [-1, -1, -1] 表示从左后上方照射
            # [ 0,  0, -1] 表示从正上方照射
            "sun_direction": [-0.3, 0.7, -0.25],  # 调整方向以获得更好的阴影效果
            "sun_intensity": GEOMETRY_DTYPE(10),   # 增加太阳光强度
            "ambient_light": GEOMETRY_DTYPE(0.2),   # 增加环境光亮度，减少纯黑区域
            "specular_color": [1.0, 1.0, 1.0],    # 高光颜色 (通常为白色)
            "specular_exponent": GEOMETRY_DTYPE(64.0), # 高光锐利度 (值越高，高光点越小越亮)
            "ks": GEOMETRY_DTYPE(0.5),            # 镜面反射系数 (高光强度)

            # --- 阴影参数 ---
            # "shadow_intensity": GEOMETRY_DTYPE(0.6), # 0.0 = 无阴影, 1.0 = 纯黑阴影 (可选，如果renderer支持)
            "shadow_bias": GEOMETRY_DTYPE(1e-4),    # 阴影偏移，防止自遮挡（可以微调）

             # --- [新] 雾效参数 ---
             # 注意：雾效仅影响 RGB 图像，不影响深度图和语义图
            "fog_enabled": True,                   # 是否启用雾效 (True/False)
            "fog_color": [210, 225, 240],        # 雾的颜色 (RGB, 0-255) - 浅蓝色调
            "fog_density": GEOMETRY_DTYPE(0.00003), # 雾的浓度 (值越小雾越淡，指数衰减)

            # --- 其他渲染参数 ---
            "sky_color": [180, 210, 255],         # 天空颜色 (RGB, 0-255) - 更鲜艳的天空蓝
            # "max_distance": 10000.0              # 可选: 最大渲染距离

            # --- NEW Control Params ---
            "shading_mode": "simple_diffuse", # Choose 'phong' or 'simple_diffuse'
            "save_outputs": ["rgb", "params", "depth", "semantic", "points"] # List: "rgb", "depth", "semantic", "points", "params"
                                              # Default if omitted: ["rgb"]
        }

        # --- 步骤 4: 循环渲染 (例如，测试不同的 shadow_bias) ---
        print("\n步骤 6: 开始循环渲染...")
        num_steps = 20  # How many steps to test
        start_bias = 1e-4  # Start just above the self-occlusion point
        end_bias = 1e-3  # End near the original default where shadows likely failed
        bias_values_to_test = np.linspace(start_bias, end_bias, num_steps, dtype=GEOMETRY_DTYPE)

        all_results = []  # Optional: Store results if needed

        # --- Select Mode for the Loop ---
        loop_shading_mode = "phong"  # <-- SET THIS to 'phong' to test shadows, or 'simple_diffuse' for the overlay effect

        for i, current_bias in enumerate(bias_values_to_test):
            loop_start = time.perf_counter()
            print(
                f"\n--- 开始渲染第 {i + 1}/{num_steps} 帧 | Shading: {loop_shading_mode} | Bias: {current_bias:.2E} ---")

            # Update render_params for this iteration
            current_render_params = base_render_params.copy()
            current_render_params["shadow_bias"] = current_bias
            current_render_params["shading_mode"] = loop_shading_mode
            # Unique file prefix for each iteration
            current_render_params["file_prefix"] = f"tune_bias_{i:03d}_val{current_bias:.1E}"
            # Decide what to save for tuning (maybe just RGB is enough?)
            current_render_params["save_outputs"] = ["rgb"]  # Save only RGB for faster tuning

            # Perform the render for this iteration
            # Note: We no longer pass flattened_data to render()
            render_results_i = renderer.render(camera_params, current_render_params)

            if render_results_i:
                print(f"    第 {i + 1} 帧渲染成功。")
                all_results.append(render_results_i)  # Store if needed
            else:
                print(f"    错误: 第 {i + 1} 帧渲染失败。")
                # Decide whether to continue or break the loop
                # break

            log_time(f"完成第 {i + 1} 帧渲染", loop_start)

            # --- [Optional] Post-loop processing ---
            # Example: Find best result, analyze all_results, etc.
        print("\n--- 循环渲染完成 ---")
        if all_results:
            print(f"共成功渲染 {len(all_results)} 帧。")
            # You can now examine the saved images in RENDER_OUTPUT_DIR

    else:
        print("\n场景数据为空，无法渲染。")

    # --- End ---
    main_end_time = time.perf_counter()
    print(f"\n--- 主程序完成 | 总耗时: {main_end_time - main_start_time:.2f}s ---")





