# scene_structure.py
from pathlib import Path
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
from modules.renderer import SKY_LABEL, LABEL_DTYPE  # Import SKY_LABEL if needed for default
from pyproj import Transformer # <-- Add pyproj import
from pyproj.exceptions import CRSError # <-- Add exception import
import random
from modules.window_generator import WindowGenerator, WindowMetadata, WINDOW_COLOR, DEFAULT_WINDOW_WIDTH_M, DEFAULT_WINDOW_SPACING_M # 导入所需


# --- 数据类型和颜色映射 (参考之前的定义) ---
GEOMETRY_DTYPE = np.float32
INDEX_DTYPE = np.int32
COLOR_DTYPE = np.uint8
Vec3 = Tuple[float, float, float]
TriIndices = Tuple[int, int, int]
research_box = (830000, 814400, 840500, 824600)
min_x, min_y, max_x, max_y = research_box
MAX_TREE_HEIGHT_M = 60.0    # 例如，限制最大高度为 60 米
MAX_TREE_SPREAD_M = 40.0    # 例如，限制最大树冠蔓延为 40 米
MAX_TREE_DBH_MM = 5000.0    # 例如，限制最大胸径为 5000 毫米 (5米)

# 颜色映射 (根据需要调整或扩展)
# 注意：这里的键是 element_type 字符串
DEFAULT_COLOR = np.array([128, 128, 128], dtype=COLOR_DTYPE) # 默认灰色
COLOR_MAP = {
    # --- 地面/景观 ---
    "Landscape": np.array([120, 190, 110], dtype=COLOR_DTYPE),  # 更鲜亮的草地绿
    "Base": np.array([80, 140, 210], dtype=COLOR_DTYPE),  # 水体/基底 - 更明亮的蓝
    "Water": np.array([80, 140, 210], dtype=COLOR_DTYPE),  # 水体 - 更明亮的蓝

    # --- 建筑/道路 ---
    "Building": np.array([205, 205, 205], dtype=COLOR_DTYPE),  # 通用建筑 - 中等偏亮灰色 (备用)
    "Building_T": np.array([235, 240, 245], dtype=COLOR_DTYPE),  # 塔楼 - 非常浅的冷灰色/近白色
    "Building_P": np.array([180, 180, 175], dtype=COLOR_DTYPE),  # 裙楼 - 稍暗、稍暖的灰色
    "ROAD": np.array([80, 80, 80], dtype=COLOR_DTYPE),  # 深灰色道路，增加对比

    # --- 特殊区域 (保持或微调) ---
    "LOT": np.array([210, 180, 140], dtype=COLOR_DTYPE),         # 地块 - 浅棕褐色/米色
    "GLA": np.array([170, 190, 90], dtype=COLOR_DTYPE),         # GLA - 稍亮的黄绿色

    # --- 其他元素 ---
    "Land_Boundary": np.array([208, 176, 151], dtype=COLOR_DTYPE),

    # --- 自然元素 ---
    "Mountain": np.array([100, 175, 95], dtype=COLOR_DTYPE),     # 基础山体 - 更鲜明的绿
    "MountainNorth": np.array([95, 170, 90], dtype=COLOR_DTYPE), # 北侧山体 - 略微调整
    "MountainSouth": np.array([110, 185, 105], dtype=COLOR_DTYPE),# 南侧山体 - 更亮绿
    "Tree": np.array([30, 180, 60], dtype=COLOR_DTYPE),
    # 非常鲜明的树木绿# --- 新增: 窗户颜色 ---
    "Window": WINDOW_COLOR if WindowGenerator else np.array([173, 216, 230], dtype=COLOR_DTYPE), # 如果导入失败，使用默认

    # --- 调试 ---
    "Unknown": np.array([255, 0, 255], dtype=COLOR_DTYPE),      # Magenta for testing
}

# --- Add Mappings for Flattening ---
# Map your object types (layers) to integer labels for the renderer
# Adjust these integer IDs as needed, but ensure they are consistent
# with how you might want to interpret the semantic output.
# Use -1 (or SKY_LABEL) for things you might want to ignore or treat differently.
TYPE_TO_LABEL_MAP = {
    # --- Ground/Landscape ---
    "Landscape": 0,
    "Base": 1,          # 水体/基底
    "Water": 1,         # 水体 (使用与Base相同的标签)
    # --- Buildings/Roads ---
    "Building": 2,      # 通用建筑标签 (备用)
    "Building_T": 9,     # 塔楼 - 新标签
    "Building_P": 10,    # 裙楼 - 新标签
    "ROAD": 3,
    # --- Special Areas ---
    "LOT": 4,           # 地块
    "GLA": 11,          # GLA - 分配新标签 (之前与LOT冲突)
    # --- Other Elements ---
    "Land_Boundary": 5, # 边界 (如果你想在语义图里区分它)
                        # 如果想忽略，可以设为 -1: "Land_Boundary": -1,
    # --- Natural Elements ---
    "Mountain": 6,      # 所有山体共享标签 6
    "MountainNorth": 6,
    "MountainSouth": 6,
    "Tree": 7,
    "Window": 8,
    # --- Fallback ---
    "Unknown": -1,      # 忽略未知类型
}
DEFAULT_LABEL = -1  # Label to use if type is not found in map


# ---------------------------------------------------------------------------
# MeshElement –– 细粒度三角面
# ---------------------------------------------------------------------------
@dataclass(order=True)
class MeshElement:
    # 恢复使用 default_factory 来自动生成 ID
    parent_object_id: str
    parent_object_type: str
    vertex_indices: TriIndices  # 指向本 Object.vertices 的索引
    color: np.ndarray
    mesh_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # --- ADDED FIELD ---
    label_id: Optional[int] = None # Allows overriding parent object's label

    def __hash__(self):
        # Hash 仍然基于 mesh_id
        return hash(self.mesh_id)


# ---------------------------------------------------------------------------
# Object –– 代表场景中的一个实体（建筑、地块、山体...）
# ---------------------------------------------------------------------------
@dataclass(order=True)
class Object:
    object_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    object_type: str = "Unknown"
    color: np.ndarray = field(default_factory=lambda: DEFAULT_COLOR.copy())

    # 顶点坐标 & Mesh
    vertices: List[Vec3] = field(default_factory=list)  # (N,3) - 使用列表存储元组
    mesh_elements: List[MeshElement] = field(default_factory=list)

    # --- Internal counters for logging ---
    _vertices_added_current_op: int = field(default=0, repr=False, init=False)
    _meshes_added_current_op: int = field(default=0, repr=False, init=False)

    def __post_init__(self):
        # Ensure mesh_id is unique per object instance if needed elsewhere,
        # though it's not directly used in the provided logic.
        self.mesh_id = str(uuid.uuid4())
        # Ensure color is a copy
        if not isinstance(self.color, np.ndarray):
             self.color = DEFAULT_COLOR.copy()

    # --------------------------- helpers ---------------------------
    def _add_vertex(self, v: Vec3) -> int:
        """将顶点加入 vertices，返回其索引。"""
        # 确保顶点是元组
        v_tuple = (float(v[0]), float(v[1]), float(v[2]))
        self.vertices.append(v_tuple)
        self._vertices_added_current_op += 1
        return len(self.vertices) - 1

    def _add_face(self, idx_a: int, idx_b: int, idx_c: int):
        """添加一个三角面。"""
        self.mesh_elements.append(
            MeshElement(
                # mesh_id 自动生成
                parent_object_id=self.object_id,
                parent_object_type=self.object_type,
                vertex_indices=(idx_a, idx_b, idx_c),
                color=self.color,
            )
        )
        self._meshes_added_current_op += 1

    def _process_poly(self, poly: Polygon, start_z: float, top_z: float):
        """
        使用 trimesh 对单个 Polygon 做拉伸并三角化。
        Returns:
            Tuple[int, int]: Number of vertices and faces added by this polygon.
        """
        verts_added_poly = 0
        meshes_added_poly = 0
        initial_vert_count_obj = len(self.vertices) # Vert count before this poly
        initial_mesh_count_obj = len(self.mesh_elements) # Mesh count before this poly

        # 1. 检查多边形有效性
        if not poly.is_valid:
            print(f"    - 🟡 Warning: Invalid polygon detected for obj '{self.object_id}', attempting buffer(0) fix...")
            poly = poly.buffer(0) # 尝试修复无效多边形
            if not poly.is_valid or poly.is_empty or not isinstance(poly, Polygon):
                 print(f"    - 🔴 Warning: Skipping invalid or empty polygon after buffer(0) for obj '{self.object_id}'.")
                 return verts_added_poly, meshes_added_poly # Return 0, 0

        # 2. 检查顶点数量 (需要至少3个不同点形成面积)
        #    注意：coords 列表最后一个点等于第一个点
        if poly.exterior is None or len(poly.exterior.coords) < 4:
            print(f"    - 🟡 Warning: Skipping polygon with < 3 distinct exterior coords for obj '{self.object_id}'.")
            return verts_added_poly, meshes_added_poly # Return 0, 0

        # 3. 检查高度
        height = top_z - start_z
        if height <= 1e-6: # 使用一个小的容差来比较浮点数
            print(f"    - 🟡 Warning: Skipping polygon with near-zero or negative height ({height:.4f}) for obj '{self.object_id}'.")
            # Optionally handle flat polygon triangulation here if needed
            return verts_added_poly, meshes_added_poly # Return 0, 0

        # 4. 使用 trimesh 进行挤出
        try:
            # 注意：trimesh 的挤出方向是 Z 轴正方向
            mesh = trimesh.creation.extrude_polygon(poly, height=height)
            # self.log_properties(mesh)

            # 检查挤出结果是否有效
            if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                print(f"    - 🟡 Warning: Trimesh extrusion resulted in an empty mesh for obj '{self.object_id}'. Skipping polygon.")
                return verts_added_poly, meshes_added_poly # Return 0, 0

            # 如果原始 start_z 不为 0，将网格向上平移
            if abs(start_z) > 1e-6: # 同样使用容差
                mesh.apply_translation([0, 0, start_z])

            # 5. 获取顶点和面，并添加到 Object
            vert_offset = len(self.vertices) # 获取当前的顶点数量作为偏移
            # 确保添加的是 Python 原生 float 元组
            new_vertices = mesh.vertices.astype(float).tolist()
            # Use internal _add_vertex and _add_face for consistency and counting
            vert_indices_map = {} # Map local mesh index to global object index
            for i, v in enumerate(new_vertices):
                global_idx = self._add_vertex(tuple(v))
                vert_indices_map[i] = global_idx

            new_faces = mesh.faces.astype(int).tolist()
            for tri in new_faces:
                a_local, b_local, c_local = tri
                # Map local indices to global indices before adding face
                a_global = vert_indices_map[a_local]
                b_global = vert_indices_map[b_local]
                c_global = vert_indices_map[c_local]
                self._add_face(a_global, b_global, c_global)

            # Calculate counts for *this specific polygon*
            verts_added_poly = len(self.vertices) - initial_vert_count_obj
            meshes_added_poly = len(self.mesh_elements) - initial_mesh_count_obj
            # print(f"      - Processed polygon part: Added {verts_added_poly} vertices, {meshes_added_poly} mesh elements.")

        except Exception as e:
            # 捕获 trimesh 可能抛出的任何异常
            print(f"    - 🔴 Error extruding polygon with trimesh for obj '{self.object_id}': {e}")
            # print(f"      Polygon WKT (first 100 chars): {poly.wkt[:100]}")
            # Reset counters for this polygon as it failed
            verts_added_poly = 0
            meshes_added_poly = 0

        return verts_added_poly, meshes_added_poly

    def _reset_op_counters(self):
        """Resets the counters for vertices/meshes added in the current operation."""
        self._vertices_added_current_op = 0
        self._meshes_added_current_op = 0

    # --------------------------- DataElement → Mesh ---------------------------
    def init_from_element(self, data_element: DataElement):
        """
        由 DataElement 构造网格。
        Returns:
            Tuple[int, int]: Number of vertices and faces added by this element.
        """
        self._reset_op_counters() # Reset counters for this specific element
        start_time = time.time()

        # 优先使用 DataElement 的 ID
        self.object_id = getattr(data_element, 'id', self.object_id)
        self.object_type = data_element.layer or "Unknown"
        # print(f"  ⏳ Initializing Object from DataElement: ID='{self.object_id}', Type='{self.object_type}'")

        geom = data_element.geometry

        # 1. 检查几何是否存在且非空
        if geom is None or geom.is_empty:
            # print(f"    - 🟡 Info: Skipping DataElement with empty geometry (ID: {self.object_id})")
            return 0, 0 # Return 0 added

        # 2. 检查几何类型是否支持
        if not isinstance(geom, (Polygon, MultiPolygon)):
            print(f"    - 🟡 Warning: Skipping unsupported geometry type: {geom.geom_type} (ID: {self.object_id})")
            return 0, 0 # Return 0 added

        # 3. 获取高度信息 (确保是 float)
        z0 = float(getattr(data_element, "start_height", 0.0) or 0.0)
        height_val = float(getattr(data_element, "height", 0.0) or 0.0)

        if self.object_type == "Land_Boundary":
            z0 = -5
            height_val = 3
        elif self.object_type == "ROAD":
            z0 = -5
            height_val = 5.2
        elif self.object_type == "Building":
            building_structure_type = data_element.custom_semantics.get("building_structure_type", None)
            if building_structure_type:
                if building_structure_type == "T":
                    self.object_type = "Building_T"
                else:
                    self.object_type = "Building_P"
        # print(f"    - Geometry Type: {geom.geom_type}, Start Z: {z0:.2f}, Height: {height_val:.2f}, Top Z: {z1:.2f}")

        # 存储源几何图形 (在类型确定后)
        if self.object_type in ["Building", "Building_T", "Building_P"]:
            self.source_geometry = geom

        z1 = z0 + height_val

        self.color = COLOR_MAP.get(self.object_type, DEFAULT_COLOR).copy()

        # 4. 处理几何 (Polygon 或 MultiPolygon)
        total_verts_added_element = 0
        total_meshes_added_element = 0

        if isinstance(geom, Polygon):
            verts_added, meshes_added = self._process_poly(geom, z0, z1)
            total_verts_added_element += verts_added
            total_meshes_added_element += meshes_added
        elif isinstance(geom, MultiPolygon):
            # print(f"    - Processing MultiPolygon with {len(geom.geoms)} parts...")
            for i, poly in enumerate(geom.geoms):
                 # print(f"      - Processing part {i+1}/{len(geom.geoms)}...")
                 verts_added, meshes_added = self._process_poly(poly, z0, z1) # 对每个部分应用相同的高度
                 total_verts_added_element += verts_added
                 total_meshes_added_element += meshes_added

        # 5. 结束日志
        elapsed = time.time() - start_time
        # Log only if something was actually added
        # if total_verts_added_element > 0 or total_meshes_added_element > 0:
        #     print(f"  ✅ Finished Object '{self.object_id}': Added {total_verts_added_element} vertices, {total_meshes_added_element} mesh elements. Time: {elapsed:.3f}s")
            # print(f"     Object '{self.object_id}' Total Now: Vertices={len(self.vertices)}, Mesh Elements={len(self.mesh_elements)}")

        # Return the counts for this specific element
        # Use the internal counters which were incremented by _add_vertex/_add_face
        return self._vertices_added_current_op, self._meshes_added_current_op

    # --------------------------- OBJ → Mesh ---------------------------
    def init_from_obj(self, obj_path: str | Path, obj_type: str):
        """
        读取 .obj（或任何 trimesh 支持的网格）并填充本对象的 vertices / mesh_elements。
        Returns:
            Tuple[int, int]: Number of vertices and faces added by this OBJ file.
        """
        self._reset_op_counters() # Reset counters for this specific OBJ
        start_time = time.time()
        obj_path = Path(obj_path)
        # self.object_id = obj_path.stem # 使用文件名（不含扩展名）作为 ID
        self.object_type = obj_type # 也用作类型，或根据需要修改
        # Attempt to get color, default to Unknown/DEFAULT_COLOR if type not in map
        self.color = COLOR_MAP.get(self.object_type, COLOR_MAP["Unknown"]).copy()
        print(f"  ⏳ Initializing Object from OBJ: Path='{obj_path}', ID='{self.object_id}', Type='{self.object_type}'")

        try:
            # 强制读取为单个网格，忽略材质和 UV
            mesh = trimesh.load_mesh(str(obj_path), force='mesh', process=False) # process=False 避免trimesh自动处理

            if mesh.is_empty or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                print(f"    - 🟡 Warning: Loaded empty mesh from {obj_path}")
                return 0, 0 # Return 0 added

            print(f"    - Loaded original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")
            print("      Original OBJ spatial info:")
            self.log_properties(mesh)  # Log original properties

            # 2. 定义并应用坐标轴变换矩阵
            #    目标: 将 OBJ (x, y, z) 映射到 Element (x', y', z')
            #    根据推断: x' = x,  y' = -z,  z' = y
            #    对应的 4x4 变换矩阵 M 使得 [x', y', z', 1] = M @ [x, y, z, 1].T
            #    (注意 trimesh 使用 M @ vertex 列向量 的形式)
            axis_swap_matrix = np.array([
                [1, 0, 0, 0],  # x' = 1*x + 0*y + 0*z
                [0, 0, -1, 0],  # y' = 0*x + 0*y - 1*z
                [0, 1, 0, 0],  # z' = 0*x + 1*y + 0*z
                [0, 0, 0, 1]
            ])
            print("\n    - Applying axis swap transformation...")
            mesh.apply_transform(axis_swap_matrix)
            print("      Axis-swapped OBJ spatial info:")
            self.log_properties(mesh)  # Log properties *after* axis swap

            # 索引偏移 (relative to this object's current vertex list)
            vert_offset = len(self.vertices)
            # 添加顶点 (确保是 float 元组) using internal method
            new_vertices = mesh.vertices.astype(float).tolist()
            vert_indices_map = {} # Map local mesh index to global object index
            for i, v in enumerate(new_vertices):
                global_idx = self._add_vertex(tuple(v))
                vert_indices_map[i] = global_idx

            # 添加面 (确保索引是 int) using internal method
            new_faces = mesh.faces.astype(int).tolist()
            for tri in new_faces:
                a_local, b_local, c_local = tri
                 # Map local indices to global indices before adding face
                a_global = vert_indices_map[a_local]
                b_global = vert_indices_map[b_local]
                c_global = vert_indices_map[c_local]
                self._add_face(a_global, b_global, c_global)

            verts_added = self._vertices_added_current_op
            meshes_added = self._meshes_added_current_op
            elapsed = time.time() - start_time
            print(f"  ✅ Finished Object '{self.object_id}' from OBJ: Added {verts_added} vertices, {meshes_added} mesh elements. Time: {elapsed:.3f}s")
            print(f"     Object '{self.object_id}' Total Now: Vertices={len(self.vertices)}, Mesh Elements={len(self.mesh_elements)}")
            self.log_properties(mesh)

            return verts_added, meshes_added

        except ValueError as ve: # 特别处理 trimesh 可能因格式问题抛出的 ValueError
             print(f"    - 🔴 Error loading OBJ (ValueError): {ve}. Check file format/content: {obj_path}")
             return 0, 0
        except Exception as e:
            print(f"    - 🔴 Error loading mesh from {obj_path}: {e}")
            return 0, 0

    def log_properties(self, mesh):
        # 从 trimesh 对象获取信息
        vertices = mesh.vertices
        if vertices.shape[0] == 0:
            print("警告：网格没有顶点。")
            return None

        vertex_count = vertices.shape[0]
        print(f"成功加载包含 {vertex_count} 个顶点的网格。")

        # 使用 trimesh 的内置属性计算边界框
        # bounds 结构是 [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        min_bounds = mesh.bounds[0]
        max_bounds = mesh.bounds[1]

        # 计算边界框的中心点（质心）
        # mesh.bounding_box.centroid 比直接用 bounds 计算更稳定
        center = mesh.bounding_box.centroid

        # 计算边界框的尺寸
        dimensions = max_bounds - min_bounds

        # 将信息存入字典
        spatial_info = {
            "min_bounds": tuple(min_bounds),  # 转为元组方便阅读
            "max_bounds": tuple(max_bounds),
            "center": tuple(center),
            "dimensions": tuple(dimensions),
            "vertex_count": vertex_count
        }

        print(spatial_info)

# ---------------------------------------------------------------------------
# Scene –– 容器：多个 Object + 可导出 OBJ
# ---------------------------------------------------------------------------
@dataclass(order=True)
class Scene:
    objects: List[Object] = field(default_factory=list)
    total_vertices: int = 0
    total_mesh_elements: int = 0
    # CRS transformer (initialized when needed)
    _transformer_4326_to_2326: Optional[Transformer] = field(default=None, init=False, repr=False)

    def _init_transformer(self):
        """Initializes the coordinate transformer."""
        if self._transformer_4326_to_2326 is None:
            try:
                # EPSG:4326 = WGS 84 (Lon/Lat)
                # EPSG:2326 = Hong Kong 1980 Grid System (Easting/Northing)
                # always_xy=True means input order is Lon, Lat and output order is Easting, Northing
                self._transformer_4326_to_2326 = Transformer.from_crs("epsg:4326", "epsg:2326", always_xy=True)
                print("  Initialized WGS84 (Lon/Lat) to HK80 (Easting/Northing) transformer.")
            except CRSError as e:
                print(f"  🔴 Error initializing pyproj transformer: {e}")
                print("     Coordinate transformation for trees will not work.")
                print("     Ensure pyproj is installed and CRS definitions are available.")

    def _create_tree_mesh(self, base_center_xy: Tuple[float, float], base_z: float, height: float, crown_radius: float, trunk_radius: float, trunk_height: float, segments: int = 10) -> Optional[trimesh.Trimesh]:
        """
        Helper to create combined mesh for a single tree using cone crown.
        Builds geometry along the Z-axis (scene's 'up').
        """
        try:
            cx, cy = base_center_xy # Scene X, Y

            # --- Create Trunk (Cylinder along Z) ---
            trunk_center_z = base_z + trunk_height / 2.0
            # Create cylinder centered at origin along Z
            trunk_mesh_origin = trimesh.creation.cylinder(
                radius=trunk_radius,
                height=trunk_height,
                sections=max(6, segments // 2)
            )
            # Move trunk to correct base position
            trunk_transform = trimesh.transformations.translation_matrix([cx, cy, trunk_center_z])
            trunk_mesh = trunk_mesh_origin.apply_transform(trunk_transform)

            # --- Create Crown (Cone along Z) ---
            crown_base_z = base_z + trunk_height
            apex_z = crown_base_z + (height - trunk_height) # Total height - trunk height = crown height
            apex_vertex = np.array([cx, cy, apex_z], dtype=GEOMETRY_DTYPE)

            # Create cone base vertices in XY plane at crown_base_z
            cone_base_verts = []
            for i in range(segments):
                angle = 2.0 * np.pi * i / segments
                # Use scene X and Y for the base circle
                x = cx + crown_radius * np.cos(angle)
                y = cy + crown_radius * np.sin(angle)
                cone_base_verts.append((x, y, crown_base_z))
            cone_base_verts_np = np.asarray(cone_base_verts, dtype=GEOMETRY_DTYPE)

            # Combine apex and base vertices for the cone
            cone_verts_np = np.vstack((apex_vertex.reshape(1, 3), cone_base_verts_np))
            num_cone_verts = len(cone_verts_np)

            # Create cone faces (triangles fanning out from apex)
            cone_faces = []
            apex_idx_local = 0 # Apex is the first vertex in cone_verts_np
            for i in range(segments):
                base_idx_local_1 = i + 1 # +1 because apex is index 0
                # Wrap around for the last face connecting back to the first base vertex
                base_idx_local_2 = (i + 1) % segments + 1
                cone_faces.append((apex_idx_local, base_idx_local_1, base_idx_local_2))

            # Create Trimesh object for the cone
            # Ensure faces are numpy array of correct type
            cone_mesh = trimesh.Trimesh(vertices=cone_verts_np, faces=np.array(cone_faces, dtype=INDEX_DTYPE))

            # --- Combine Trunk and Crown Meshes ---
            # Check if meshes are valid before concatenating
            if trunk_mesh is None or cone_mesh is None or trunk_mesh.is_empty or cone_mesh.is_empty:
                 print("    🔴 Warning: Trunk or Crown mesh is invalid/empty, cannot combine.")
                 # Return whichever part is valid, or None
                 return trunk_mesh if trunk_mesh and not trunk_mesh.is_empty else (cone_mesh if cone_mesh and not cone_mesh.is_empty else None)

            combined_mesh = trimesh.util.concatenate([trunk_mesh, cone_mesh])
            # Optional: Fix normals and winding after concatenation if needed
            combined_mesh.fix_normals()
            return combined_mesh

        except Exception as e:
            print(f"    🔴 Error creating cone tree geometry: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            return None

    # ──────────────────── 容器操作 ────────────────────
    def _add_if_valid(self, obj: Object, verts_added: int, meshes_added: int):
        """Adds the object if it has geometry and updates scene totals."""
        if obj.vertices and obj.mesh_elements and (verts_added > 0 or meshes_added > 0):
            self.objects.append(obj)
            self.total_vertices += verts_added
            self.total_mesh_elements += meshes_added
            return True # Indicate success
        else:
            # print(f"    - ℹ️ Info: Object '{obj.object_id}' resulted in no geometry, not added to scene.")
            return False # Indicate failure/skip

    # ──────────────────── 批量导入 DataElement ────────────────────
    def load_elements(self, elements: List[DataElement], report_interval: int = 500,
                      generate_windows_flag: bool = False,  # 是否启用窗户生成
                      window_aoi_wkt: Optional[str] = None,  # 窗户生成的AOI区域 (WKT字符串)
                      window_metadata_output_path: Optional[str | Path] = None,  # 窗户元数据输出路径
                      window_config: Optional[Dict[str, Any]] = None,  # 传递给WindowGenerator的配置
                      target_building_types: List[str] = ["Building_T", "Building_P"]  # 目标建筑类型
                      ):
        """Loads DataElements into the scene, creating Objects."""
        total_elements_to_process = len(elements)
        print(f"\n▶ Starting bulk import of DataElements...")
        print(f"  Total DataElements to process: {total_elements_to_process}")
        t0 = time.time()
        valid_objects_created = 0
        elements_processed = 0
        verts_added_phase = 0
        meshes_added_phase = 0
        # --- 用于存储符合窗户生成条件的建筑对象 ---
        buildings_for_window_gen: List[Object] = []
        aoi_polygon: Optional[Polygon] = None

        # --- 预处理窗户生成 AOI (如果启用) ---
        if generate_windows_flag:
            aoi_polygon = wkt_loads(window_aoi_wkt)
            print(f"  ✅ 已启用窗户生成，并成功解析 AOI Polygon。")

        # --- 第一阶段：加载所有元素并识别需要生成窗户的建筑 ---
        print("  ⏳ 正在加载元素并识别目标建筑...")
        objects_dict = {}  # 用于快速查找已创建的对象
        for i, ele in enumerate(elements, 1):
            elements_processed += 1
            obj = Object()
            verts_added, meshes_added = obj.init_from_element(ele)

            if self._add_if_valid(obj, verts_added, meshes_added):
                valid_objects_created += 1
                verts_added_phase += verts_added
                meshes_added_phase += meshes_added
                objects_dict[obj.object_id] = obj # 添加到查找字典

                # --- 检查是否为需要生成窗户的建筑，且在 AOI 内 ---
                if (generate_windows_flag and
                        aoi_polygon and  # 确保 AOI 有效
                        obj.object_type in target_building_types and
                        obj.source_geometry and  # 确保 source_geometry 已被设置
                        obj.source_geometry.intersects(aoi_polygon)):  # 使用 intersects 检查重叠
                    buildings_for_window_gen.append(obj)  # 收集符合条件的对象

            if i % report_interval == 0 or i == total_elements_to_process:
                elapsed = time.time() - t0
                print(f"\r  ● 元素加载进度: {i}/{total_elements_to_process}."
                      f" 有效对象: {valid_objects_created}."
                      f" 场景总计: {self.total_vertices} 顶点, {self.total_mesh_elements} 面。"
                      f" 已耗时: {elapsed:.1f}s", end="")

        elapsed_total = time.time() - t0
        print(f"✔ 完成 DataElement 导入阶段。")
        print(f"  此阶段总结:")
        print(f"    - 处理元素: {elements_processed}/{total_elements_to_process}")
        print(f"    - 创建有效对象: {valid_objects_created}")
        print(f"    - 添加几何: {verts_added_phase} 顶点, {meshes_added_phase} 面。")
        print(f"    - 发现符合窗户生成条件的建筑: {len(buildings_for_window_gen)} 个")
        print(f"    - 耗时: {elapsed_total:.2f}s\n")

        # --- 循环结束后，执行窗户生成 ---
        if generate_windows_flag and buildings_for_window_gen:
            print(f"\n▶ 开始为 {len(buildings_for_window_gen)} 个符合条件的建筑生成窗户...")
            t0_window_gen = time.time()

            # 准备窗户生成器的配置
            window_gen_params = window_config if window_config else {}
            # 如果路径有效，则传递给 save_metadata
            output_path_for_meta = Path(window_metadata_output_path) if window_metadata_output_path else None

            try:
                # 实例化 WindowGenerator (现在不依赖 scene_structure)
                generator = WindowGenerator(**window_gen_params)

                # 调用生成函数 (传入收集到的建筑列表和 AOI)
                # generate_windows 返回: new_window_objects, window_metadata, processed_building_ids
                window_metadata, windows_geometry_by_building = generator.generate_windows(
                    buildings_for_window_gen, aoi_polygon)
                log_time(f"窗户生成完成", t0_window_gen)
                print(f"  窗户生成总结:")

                log_time(f"窗户几何数据生成完成", t0_window_gen)  # 这里的时间可能不准，因为日志函数可能有问题
                print(f"  窗户生成总结:")
                print(f"    - 生成的元数据记录数: {len(window_metadata)}")
                total_geom_groups = sum(len(v) for v in windows_geometry_by_building.values())
                print(f"    - 生成的窗户几何数据组数: {total_geom_groups}")

                # --- 开始集成窗户几何数据 ---
                print("\n  ⏳ 开始将窗户几何数据集成到建筑对象中...")
                integration_start_time = time.time()
                verts_added_windows_total = 0
                meshes_added_windows_total = 0
                window_label_id = TYPE_TO_LABEL_MAP.get("Window", DEFAULT_LABEL)
                buildings_integrated_count = 0
                total_buildings_to_integrate = len(windows_geometry_by_building)

                # 使用 list(dict.items()) 复制键值对，以便在循环中安全删除
                integration_items = list(windows_geometry_by_building.items())


                for building_id, window_geometries in integration_items:
                    buildings_integrated_count += 1
                    target_building = objects_dict.get(building_id)
                    if not target_building:
                        print(f"    🟡 警告: 在集成期间未找到建筑 {building_id}，跳过。")
                        # 在删除前确保从原始字典中删除，即使处理失败
                        if building_id in windows_geometry_by_building:
                            del windows_geometry_by_building[building_id]
                        continue

                    num_windows_for_building = len(window_geometries)
                    verts_added_this_building = 0
                    meshes_added_this_building = 0
                    print(f"    -> 集成建筑 {buildings_integrated_count}/{total_buildings_to_integrate}: "
                          f"ID='{building_id}', 类型='{target_building.object_type}', "
                          f"窗户数={num_windows_for_building}")

                    integration_report_interval = max(1000, num_windows_for_building // 10)  # 每处理1000个或10%的窗户报告一次

                    for i, (win_vertices, win_faces) in enumerate(window_geometries):
                        if not win_vertices or not win_faces: continue

                        vert_map = {}
                        # 添加窗户顶点到建筑对象
                        for vert_i, vert_coords in enumerate(win_vertices):
                            global_idx = target_building._add_vertex(vert_coords)
                            vert_map[vert_i] = global_idx
                            verts_added_this_building += 1

                        # 添加窗户面片到建筑对象
                        for face in win_faces:
                            try:
                                a_local, b_local, c_local = face
                                a_global = vert_map[a_local]
                                b_global = vert_map[b_local]
                                c_global = vert_map[c_local]

                                # 创建 MeshElement，指定颜色和标签
                                window_mesh = MeshElement(
                                    parent_object_id=target_building.object_id,
                                    parent_object_type=target_building.object_type,
                                    vertex_indices=(a_global, b_global, c_global),
                                    color=WINDOW_COLOR.copy(),
                                    label_id=window_label_id
                                )
                                target_building.mesh_elements.append(window_mesh)
                                meshes_added_this_building += 1
                            except KeyError as ke:
                                # print(f"    🔴 错误: KeyError {ke} 在集成建筑 {building_id} 的面时。 Vert Map: {vert_map}, Face: {face}")
                                pass  # 减少输出
                            except Exception as e_int:
                                # print(f"    🔴 错误: 在集成建筑 {building_id} 的面时出错: {e_int}")
                                pass  # 减少输出

                        # 在集成单个建筑时报告进度
                        if (i + 1) % integration_report_interval == 0 or (i + 1) == num_windows_for_building:
                            current_verts = len(target_building.vertices)
                            current_meshes = len(target_building.mesh_elements)
                            print(f"\r       进度: 已集成 {i + 1}/{num_windows_for_building} 个窗户。"
                                  f" 当前建筑顶点: {current_verts}, 面片: {current_meshes}", end="")

                    print()  # 单个建筑集成完毕后换行
                    print(
                        f"       集成完毕: 共添加 {verts_added_this_building} 顶点, {meshes_added_this_building} 面片。")
                    verts_added_windows_total += verts_added_this_building
                    meshes_added_windows_total += meshes_added_this_building

                    # --- 关键优化：处理完一个建筑后，从字典中删除其几何数据以释放内存 ---
                    try:
                        del windows_geometry_by_building[building_id]
                        # print(f"       已释放建筑 {building_id} 的窗户几何数据内存。") # 可以取消注释用于调试
                    except KeyError:
                        print(f"    🟡 警告: 尝试删除已处理的建筑 {building_id} 数据时未找到键。")

                    # --- 所有建筑集成完毕 ---
                integration_elapsed = time.time() - integration_start_time
                print(f"  ✔ 完成所有窗户几何数据集成。")
                print(f"    - 集成阶段添加总计: {verts_added_windows_total} 顶点, {meshes_added_windows_total} 面片。")
                print(f"    - 集成阶段耗时: {integration_elapsed:.2f}s")

                # 更新场景总数
                self.total_vertices += verts_added_windows_total
                self.total_mesh_elements += meshes_added_windows_total
                print(f"    - 当前场景累计: {self.total_vertices} 顶点, {self.total_mesh_elements} 面。")

                # 保存元数据
                if window_metadata and output_path_for_meta:
                    WindowGenerator.save_metadata(window_metadata, output_path_for_meta)
                elif window_metadata:
                    print("  🟡 警告: 生成了窗户元数据，但未提供有效的输出路径。")

            except Exception as e_win_integration:
                print(f"  🔴 在窗户生成或集成过程中发生严重错误: {e_win_integration}")
                import traceback
                traceback.print_exc()

        elif generate_windows_flag:
            print("\n▶ 未找到符合窗户生成条件的建筑，跳过窗户生成和集成。")
        # --- 窗户生成逻辑结束 ---

    # ──────────────────── 单 OBJ 导入 ────────────────────
    def load_obj(self, path: str | Path, obj_type: str):
        """Loads a single OBJ file as one Object into the scene."""
        path = Path(path)
        print(f"▶ Importing OBJ file: {path} ...")
        t0 = time.time()
        obj = Object() # Create a new object for the OBJ
        verts_added, meshes_added = obj.init_from_obj(path, obj_type)

        if self._add_if_valid(obj, verts_added, meshes_added):
            elapsed = time.time() - t0
            print(f"✔ OBJ Import successful: '{obj.object_id}'")
            print(f"  Summary for OBJ Loading:")
            print(f"    - Added:   {verts_added} vertices, {meshes_added} mesh elements.")
            print(f"    - Time:    {elapsed:.2f}s")
            print(f"  Scene Totals Now: {self.total_vertices} vertices, {self.total_mesh_elements} mesh elements.\n")
        else:
             elapsed = time.time() - t0
             print(f"✖ OBJ Import failed or resulted in no geometry: '{path.name}'. Time: {elapsed:.2f}s\n")

    # --- [新增] plan_trees 方法 ---
    def plan_trees(self, tree_json_path: str | Path,
                   default_base_z: float = 0.0,
                   height_variation: float = 0.2, # <-- 新增：高度浮动比例 (±20%)
                   spread_variation: float = 0.15):
        print(f"\n▶ Planning trees from: {tree_json_path}...")
        t0_plan_trees = time.time()
        tree_json_path = Path(tree_json_path)

        if not tree_json_path.exists():
            print(f"  🔴 Error: Tree JSON file not found at '{tree_json_path}'")
            return

        # Initialize coordinate transformer if not already done
        self._init_transformer()
        if self._transformer_4326_to_2326 is None:
            print("  🔴 Error: Coordinate transformer not available. Cannot process trees.")
            return

        # Load JSON data
        try:
            with open(tree_json_path, 'r', encoding='utf-8') as f:
                tree_data = json.load(f)
            if not isinstance(tree_data, dict) or tree_data.get(
                    "type") != "FeatureCollection" or "features" not in tree_data:
                print(f"  🔴 Error: Invalid GeoJSON structure in '{tree_json_path}'")
                return
            features = tree_data["features"]
            print(f"  Found {len(features)} tree features in JSON.")
        except json.JSONDecodeError as e:
            print(f"  🔴 Error decoding JSON from '{tree_json_path}': {e}")
            return
        except IOError as e:
            print(f"  🔴 Error reading file '{tree_json_path}': {e}")
            return

        # Process each tree feature
        trees_added_count = 0
        trees_failed_count = 0
        total_verts_added = 0
        total_meshes_added = 0

        for i, feature in enumerate(features):
            try:
                geom = feature.get("geometry")
                props = feature.get("properties", {})

                if not geom or geom.get("type") != "Point" or "coordinates" not in geom:
                    # print(f"  🟡 Skipping feature {i+1}: Invalid or missing Point geometry.")
                    trees_failed_count += 1
                    continue

                # --- 1. Extract and Convert Coordinates ---
                lon, lat = geom["coordinates"]  # Assumes [Longitude, Latitude] order
                # Transform to HK80 (Easting, Northing which are Scene X, Y)
                scene_x, scene_y = self._transformer_4326_to_2326.transform(lon, lat)
                # TODO: Implement terrain height lookup here if possible
                # For now, use default_base_z
                base_z = default_base_z

                # --- 2. Extract Properties & ADD RANDOMIZATION ---
                height_m_prop = props.get("Height_M")
                crown_spread_m_prop = props.get("Crown_Spread_M")
                if height_m_prop is None or crown_spread_m_prop is None:
                    trees_failed_count += 1
                    continue  # Skip if required keys missing
                dbh_mm = float(props.get("DBH_MM", 100.0))

                height_m = float(height_m_prop)
                crown_spread_m = float(crown_spread_m_prop)

                # --- Randomize Height ---
                random_h_factor = 1.0 + random.uniform(-height_variation, height_variation)
                height_m = max(1.0, height_m * random_h_factor)  # Ensure min height 1m

                # --- Randomize Crown Spread ---
                random_cs_factor = 1.0 + random.uniform(-spread_variation, spread_variation)
                crown_spread_m = max(0.5, crown_spread_m * random_cs_factor)  # Ensure min spread 0.5m

                # --- End Randomization ---

                if not (min_x <= scene_x <= max_x and min_y <= scene_y <= max_y):
                    trees_failed_count += 1
                    continue  # 跳过此树

                if not (0 < height_m <= MAX_TREE_HEIGHT_M and
                        0 < crown_spread_m <= MAX_TREE_SPREAD_M and
                        0 < dbh_mm <= MAX_TREE_DBH_MM):
                    print(f"🟡 V1 Skipping {i+1}: Unrealistic dimensions from data (H={height_m:.1f}, Spread={crown_spread_m:.1f}, DBH={dbh_mm:.0f}). Exceeds limits.")
                    trees_failed_count += 1
                    continue

                crown_radius = crown_spread_m / 2.0
                trunk_radius = max(0.05, dbh_mm / 2000.0)  # Convert mm to m, ensure min radius
                # Heuristic for trunk height (e.g., 30% of total height, capped)
                trunk_height = min(height_m * 0.3, 2.0)  # Max 2m trunk height? Adjustable.
                trunk_height = max(0.1, trunk_height)  # Ensure min trunk height
                if trunk_height >= height_m:  # Ensure crown has some height
                    trunk_height = height_m * 0.5
                # Ensure crown radius is not excessively small compared to trunk
                crown_radius = max(crown_radius, trunk_radius * 2.0)

                # --- 3. Generate Tree Geometry using Helper ---
                tree_mesh = self._create_tree_mesh(
                    base_center_xy=(scene_x, scene_y), base_z=base_z, height=height_m,
                    crown_radius=crown_radius, trunk_radius=trunk_radius, trunk_height=trunk_height
                )

                if tree_mesh is None or tree_mesh.is_empty:
                    # print(f"  🟡 Skipping feature {i+1}: Failed to generate tree mesh geometry.")
                    trees_failed_count += 1
                    continue

                # --- 4. Create Scene Object for the Tree ---
                tree_obj = Object(
                    object_type="Tree",
                    color=COLOR_MAP["Tree"].copy()  # Get tree color
                )

                # --- 5. Add vertices and faces to the Object ---
                tree_obj._reset_op_counters()
                vert_indices_map = {}
                for vert_idx_local, vert_coords in enumerate(tree_mesh.vertices):
                    global_idx = tree_obj._add_vertex(tuple(vert_coords))
                    vert_indices_map[vert_idx_local] = global_idx

                for face in tree_mesh.faces:
                    try:
                        a_local, b_local, c_local = face
                        a_global = vert_indices_map[a_local]
                        b_global = vert_indices_map[b_local]
                        c_global = vert_indices_map[c_local]
                        tree_obj._add_face(a_global, b_global, c_global)
                    except KeyError as ke:
                        print(
                            f"    🔴 KeyError adding face for tree {tree_obj.object_id}: Index {ke} not in map. Skipping face.")
                    except Exception as face_e:
                        print(f"    🔴 Error adding face for tree {tree_obj.object_id}: {face_e}. Skipping face.")

                # --- 6. Add Tree Object to Scene ---
                verts_added = tree_obj._vertices_added_current_op
                meshes_added = tree_obj._meshes_added_current_op
                if self._add_if_valid(tree_obj, verts_added, meshes_added):
                    trees_added_count += 1
                    total_verts_added += verts_added
                    total_meshes_added += meshes_added
                else:
                    # print(f"  🟡 Skipping feature {i+1}: Tree object resulted in no addable geometry.")
                    trees_failed_count += 1

            except (TypeError, ValueError) as te:
                print(f"  🟡 Skipping feature {i + 1}: Error processing properties or coordinates - {te}")
                trees_failed_count += 1
            except Exception as e:
                print(f"  🔴 Skipping feature {i + 1}: Unexpected error - {e}")
                trees_failed_count += 1

        elapsed = time.time() - t0_plan_trees
        print(f"✔ Finished planning trees.")
        print(f"  Summary for Tree Planning:")
        print(f"    - Processed: {len(features)} features")
        print(f"    - Added:     {trees_added_count} valid Tree Objects")
        print(f"    - Skipped/Failed: {trees_failed_count}")
        print(f"    - Added:     {total_verts_added} vertices, {total_meshes_added} mesh elements in this phase.")
        print(f"    - Scene Totals Now: {self.total_vertices} vertices, {self.total_mesh_elements} mesh elements.")
        print(f"    - Time:      {elapsed:.2f}s\n")

    # --- [新增] plan_trees_v2 方法 (处理 EPSG:2326 坐标, 使用默认尺寸) ---
    def plan_trees_v2(
            self,
            tree_json_path: str | Path,
            default_base_z: float = 0.0,
            # 现在这些是随机化的中心参考值
            avg_height_m: float = 8.0,
            avg_crown_spread_m: float = 4.0,
            height_variation: float = 0.5,  # <-- V2 使用更大的浮动范围 (±30%)
            spread_variation: float = 0.25,  # <-- V2 树冠浮动范围 (±25%)
    ):
        """
        Reads tree data from a GeoJSON file (assumed EPSG:2326),
        generates simple tree geometry using defaults for height/spread,
        and adds them to the scene.
        """
        print(f"\n▶ Planning trees (V2 - EPSG:2326) from: {tree_json_path}...")
        t0_plan_trees = time.time()
        tree_json_path = Path(tree_json_path)

        if not tree_json_path.exists(): print(f"  🔴 Error: Tree JSON file not found at '{tree_json_path}'"); return

        # --- 不需要 Transformer ---

        # Load JSON data
        try:
            with open(tree_json_path, 'r', encoding='utf-8') as f:
                tree_data = json.load(f)
            if not isinstance(tree_data, dict) or tree_data.get(
                "type") != "FeatureCollection" or "features" not in tree_data: print(
                f"  🔴 Error: Invalid GeoJSON structure in '{tree_json_path}'"); return
            crs_info = tree_data.get("crs", {}).get("properties", {}).get("name")
            if crs_info and "EPSG:2326" not in crs_info: print(
                f"  🟡 Warning: GeoJSON CRS is '{crs_info}', but expected EPSG:2326. Coordinates treated as HK80.")
            features = tree_data["features"];
            print(f"  Found {len(features)} tree features in JSON.")
        except Exception as e:
            print(f"  🔴 Error reading/parsing JSON '{tree_json_path}': {e}"); return

        # Process each tree feature
        trees_added_count = 0
        trees_failed_count = 0
        total_verts_added = 0
        total_meshes_added = 0
        for i, feature in enumerate(features):
            try:
                geom = feature.get("geometry")
                props = feature.get("properties", {})
                if not geom or geom.get(
                    "type") != "Point" or "coordinates" not in geom: trees_failed_count += 1; continue

                # --- 1. Extract Coordinates (Direct HK80) ---
                scene_x, scene_y = geom["coordinates"]  # Direct Easting, Northing
                base_z = default_base_z  # TODO: Terrain height lookup

                if not (min_x <= scene_x <= max_x and min_y <= scene_y <= max_y):
                    trees_failed_count += 1
                    continue  # 跳过此树

                # --- 3. Extract DBH & GENERATE RANDOM H/CS ---
                dbh_mm = float(props.get("DBH", 100.0))  # Still get DBH

                # --- Randomize Height around avg_height_m ---
                random_h_factor = 1.0 + random.uniform(-height_variation, height_variation)
                height_m = max(1.5, avg_height_m * random_h_factor)  # Min height 1.5m for V2?

                # --- Randomize Crown Spread around avg_crown_spread_m ---
                random_cs_factor = 1.0 + random.uniform(-spread_variation, spread_variation)
                crown_spread_m = max(1.0, avg_crown_spread_m * random_cs_factor)  # Min spread 1.0m

                if not (0 < height_m <= MAX_TREE_HEIGHT_M and
                        0 < crown_spread_m <= MAX_TREE_SPREAD_M and
                        0 < dbh_mm <= MAX_TREE_DBH_MM):
                    print(f"🟡 V2 Skipping : Unrealistic dimensions from data (H={height_m:.1f}, Spread={crown_spread_m:.1f}, DBH={dbh_mm:.0f}). Exceeds limits.")
                    trees_failed_count += 1
                    continue

                # --- 3. Calculate Model Parameters ---
                # (Same calculation logic)
                crown_radius = crown_spread_m / 2.0;
                trunk_radius = max(0.05, dbh_mm / 2000.0)
                trunk_height = min(height_m * 0.3, 2.0);
                trunk_height = max(0.1, trunk_height)
                if trunk_height >= height_m:
                    trunk_height = height_m * 0.5
                crown_radius = max(crown_radius, trunk_radius * 1.5)

                # --- 4. Generate Tree Geometry ---
                tree_mesh = self._create_tree_mesh(
                    base_center_xy=(scene_x, scene_y), base_z=base_z, height=height_m,
                    crown_radius=crown_radius, trunk_radius=trunk_radius, trunk_height=trunk_height
                )
                if tree_mesh is None or tree_mesh.is_empty: trees_failed_count += 1; continue

                # --- 5. Create Scene Object ---
                tree_id_str = props.get('TREE_ID', f'v2_{uuid.uuid4()}')  # Use TREE_ID if available
                tree_obj = Object(object_type="Tree",
                                  color=COLOR_MAP["Tree"].copy())

                # --- 6. Add vertices and faces ---
                # (Same logic)
                tree_obj._reset_op_counters();
                vert_indices_map = {}
                for vert_idx_local, vert_coords in enumerate(tree_mesh.vertices): vert_indices_map[
                    vert_idx_local] = tree_obj._add_vertex(tuple(vert_coords))
                for face in tree_mesh.faces:
                    try:
                        a_local, b_local, c_local = face; tree_obj._add_face(vert_indices_map[a_local],
                                                                             vert_indices_map[b_local],
                                                                             vert_indices_map[c_local])
                    except Exception as face_e:
                        print(f"    🔴 Error adding face for tree {tree_obj.object_id}: {face_e}. Skipping face.")

                # --- 7. Add to Scene ---
                verts_added = tree_obj._vertices_added_current_op;
                meshes_added = tree_obj._meshes_added_current_op
                if self._add_if_valid(tree_obj, verts_added, meshes_added):
                    trees_added_count += 1; total_verts_added += verts_added; total_meshes_added += meshes_added
                else:
                    trees_failed_count += 1

            except (TypeError, ValueError) as te:
                print(
                    f"  🟡 Skipping feature {i + 1} (V2): Error processing properties or coordinates - {te}"); trees_failed_count += 1
            except Exception as e:
                print(f"  🔴 Skipping feature {i + 1} (V2): Unexpected error - {e}"); trees_failed_count += 1

        elapsed = time.time() - t0_plan_trees
        print(
            f"✔ Finished planning trees (V2). Added: {trees_added_count}, Skipped: {trees_failed_count}. Added {total_verts_added}v, {total_meshes_added}m. Time: {elapsed:.2f}s\n")

    def plan_trees_v3(
            self,
            tree_json_path: str | Path,
            default_base_z: float = 0.0,
            height_variation: float = 0.2, # <-- 新增：高度浮动比例 (±20%)
            spread_variation: float = 0.15 # <-- 新增：树冠浮动比例 (±15%)
    ):
        """ V3: HK80 coords, properties: HEIGHT, CROWN, DBH """
        print(f"\n▶ Planning trees (V3 - EPSG:2326, Specific Props) from: {tree_json_path}...")
        t0_plan_trees = time.time()
        tree_json_path = Path(tree_json_path)
        if not tree_json_path.exists(): print(f"🔴 Error: V3 JSON not found"); return
        try:
            with open(tree_json_path, 'r', encoding='utf-8') as f: tree_data = json.load(f)
            if not isinstance(tree_data, dict) or tree_data.get("type") != "FeatureCollection": print(f"🔴 Error: V3 Invalid GeoJSON"); return
            features = tree_data.get("features", []); print(f"  Found {len(features)} features.")
        except Exception as e: print(f"🔴 Error reading/parsing V3 JSON: {e}"); return

        trees_added_count = 0; trees_failed_count = 0; trees_filtered_out = 0
        total_verts_added = 0; total_meshes_added = 0
        filter_active = False
        if research_box:
            min_x, min_y, max_x, max_y = research_box
            filter_active = True
            print(f"  Applying filter: X=[{min_x}, {max_x}], Y=[{min_y}, {max_y}]")

        for i, feature in enumerate(features):
            tree_id_for_log = f'feature_{i+1}'
            try:
                geom = feature.get("geometry"); props = feature.get("properties", {})
                tree_id_for_log = props.get('TMCP_ID', props.get('OBJECTID', f'feature_{i+1}'))
                if not geom or geom.get("type") != "Point" or "coordinates" not in geom: trees_failed_count += 1; continue

                # --- 1. Extract & VALIDATE Coordinates (Direct HK80) ---
                coords = geom["coordinates"]
                scene_x: Optional[float] = None
                scene_y: Optional[float] = None
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    try:
                        # Attempt conversion, handle None or non-numeric
                        temp_x = float(coords[0]) if coords[0] is not None else None
                        temp_y = float(coords[1]) if coords[1] is not None else None
                        if temp_x is not None and temp_y is not None:
                             scene_x, scene_y = temp_x, temp_y
                    except (ValueError, TypeError):
                        pass # Keep scene_x/y as None if conversion fails

                if scene_x is None or scene_y is None:
                    print(f"🟡 V3 Skipping {tree_id_for_log}: Invalid/non-numeric coordinates: {coords}.")
                    trees_failed_count += 1; continue
                # --- End Coordinate Validation ---

                # --- 2. Filter by Research Box ---
                if filter_active:
                    # Comparison is now safe
                    if not (min_x <= scene_x <= max_x and min_y <= scene_y <= max_y):
                        trees_filtered_out += 1; continue

                base_z = default_base_z

                # --- 3. Extract Properties & ADD RANDOMIZATION ---
                height_prop = props.get("HEIGHT")
                crown_prop = props.get("CROWN")
                dbh_prop = props.get("DBH")
                if height_prop is None or crown_prop is None or dbh_prop is None:
                    trees_failed_count += 1
                    print(f"🟡 V3 Skipping {tree_id_for_log}: Missing required property.");
                    continue  # Skip if required keys missing

                height_m = float(height_prop)
                crown_spread_m = float(crown_prop)  # Assume CROWN is spread
                dbh_mm = float(dbh_prop)  # Assume DBH is mm

                # --- Randomize Height ---
                random_h_factor = 1.0 + random.uniform(-height_variation, height_variation)
                height_m = max(1.0, height_m * random_h_factor)  # Min height 1m

                # --- Randomize Crown Spread ---
                random_cs_factor = 1.0 + random.uniform(-spread_variation, spread_variation)
                crown_spread_m = max(0.5, crown_spread_m * random_cs_factor)  # Min spread 0.5m

                if not (0 < height_m <= MAX_TREE_HEIGHT_M and
                        0 < crown_spread_m <= MAX_TREE_SPREAD_M and
                        0 < dbh_mm <= MAX_TREE_DBH_MM):
                    print(f"🟡 V3 Skipping {tree_id_for_log}: Unrealistic dimensions from data (H={height_m:.1f}, Spread={crown_spread_m:.1f}, DBH={dbh_mm:.0f}). Exceeds limits.")
                    trees_failed_count += 1
                    continue
                # --- End Randomization ---

                # --- 4. Calculate Model Parameters ---
                # ... (保持不变) ...
                crown_radius = crown_spread_m / 2.0; trunk_radius = max(0.05, dbh_mm / 2000.0)
                trunk_height = min(height_m * 0.3, 2.0); trunk_height = max(0.1, trunk_height)
                if trunk_height >= height_m: trunk_height = height_m * 0.5
                crown_radius = max(crown_radius, trunk_radius * 1.5)

                # --- 5. Generate Tree Geometry ---
                tree_mesh = self._create_tree_mesh( # Simplified call
                    base_center_xy=(scene_x, scene_y), base_z=base_z, height=height_m,
                    crown_radius=crown_radius, trunk_radius=trunk_radius, trunk_height=trunk_height
                )
                if tree_mesh is None or tree_mesh.is_empty: trees_failed_count += 1; continue

                # --- 6. Create & Add Scene Object ---
                # ... (保持不变) ...
                tree_obj = Object(object_type="Tree", color=COLOR_MAP["Tree"].copy())
                tree_obj._reset_op_counters(); vert_map={};
                for vi, vc in enumerate(tree_mesh.vertices): vert_map[vi] = tree_obj._add_vertex(tuple(vc))
                for face in tree_mesh.faces:
                    try: tree_obj._add_face(vert_map[face[0]], vert_map[face[1]], vert_map[face[2]])
                    except Exception: pass
                verts_added=tree_obj._vertices_added_current_op; meshes_added=tree_obj._meshes_added_current_op
                if self._add_if_valid(tree_obj, verts_added, meshes_added): trees_added_count += 1; total_verts_added += verts_added; total_meshes_added += meshes_added
                else: trees_failed_count += 1

            except (TypeError, ValueError) as te: print(f"🟡 V3 Skipping {tree_id_for_log}: Type/Value Error - {te}"); trees_failed_count += 1
            except Exception as e: print(f"🔴 V3 Skipping {tree_id_for_log}: Unexpected Error - {e}"); trees_failed_count += 1

        elapsed = time.time() - t0_plan_trees
        print(f"✔ Finished planning trees (V3).")
        print(
            f"  Summary (V3): Added: {trees_added_count}, Filtered: {trees_filtered_out}, Failed/Skipped: {trees_failed_count}.")  # <-- 更新日志
        print(f"    Added {total_verts_added}v, {total_meshes_added}m. Time: {elapsed:.2f}s\n")

    # ──────────────────── 导出到 .obj ────────────────────
    def export_to_obj(self, out_path: str | Path):
        """将整个场景写入单个 OBJ 文件（不含材质）。"""
        start_time = time.time()
        out_path = Path(out_path)
        print(f"\n▶ Exporting scene to '{out_path}'...")

        lines: List[str] = ["# Generated by SceneStructure script\n"]
        lines.append(f"# Total Objects: {len(self.objects)}\n")
        lines.append(f"# Total Vertices: {self.total_vertices}\n")
        lines.append(f"# Total Faces (Mesh Elements): {self.total_mesh_elements}\n")
        lines.append("\n")

        vert_global_offset = 0
        exported_vertices = 0
        exported_faces = 0
        exported_objects = 0

        for i, obj in enumerate(self.objects):
            if not obj.vertices or not obj.mesh_elements:
                # This check should be redundant due to _add_if_valid, but kept for safety
                print(f"  - 🟡 Skipping object {i+1} ('{obj.object_id}') during export due to missing vertices or faces.")
                continue

            exported_objects += 1
            lines.append(f"o {obj.object_type}_{obj.object_id}\n") # 对象名称
            # 顶点
            object_vertex_count = len(obj.vertices)
            for vx, vy, vz in obj.vertices:
                lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
                exported_vertices += 1

            # 面 (注意 OBJ 索引从 1 开始)
            object_face_count = 0
            for mesh in obj.mesh_elements:
                # 确保 vertex_indices 中的索引是有效的 (relative to obj.vertices)
                if not all(0 <= idx < object_vertex_count for idx in mesh.vertex_indices):
                     print(f"    - 🔴 Error: Invalid local vertex index in mesh element {mesh.mesh_id} for object {obj.object_id}. Indices: {mesh.vertex_indices}, Max Index: {object_vertex_count-1}. Skipping face.")
                     continue # 跳过这个无效的面

                try:
                    # Apply global offset for OBJ format (1-based indexing)
                    a, b, c = (v_idx + 1 + vert_global_offset for v_idx in mesh.vertex_indices)
                    lines.append(f"f {a} {b} {c}\n")
                    exported_faces += 1
                    object_face_count += 1
                except Exception as e: # Catch any unexpected errors during formatting
                     print(f"    - 🔴 Error processing face for mesh {mesh.mesh_id} in object {obj.object_id}: {e}. Skipping face.")
                     continue

            # Update the global offset for the next object's vertices
            vert_global_offset += object_vertex_count
            # print(f"  - Exported object {i+1}: '{obj.object_id}' ({object_vertex_count} vertices, {object_face_count} faces)")

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            out_path.write_text("".join(lines), encoding="utf-8")
            elapsed = time.time() - start_time
            print(f"\n✔ Scene exported successfully to '{out_path}'")
            print(f"  Export Summary:")
            print(f"    - Objects written: {exported_objects}/{len(self.objects)}")
            # Verify final counts match scene totals
            print(f"    - Vertices written: {exported_vertices} (Scene total: {self.total_vertices})")
            print(f"    - Faces written:    {exported_faces} (Scene total: {self.total_mesh_elements})")
            if exported_vertices != self.total_vertices or exported_faces != self.total_mesh_elements:
                 print("    - 🟡 Warning: Written vertex/face count mismatch with scene totals. Check for errors during export.")
            print(f"  Export time: {elapsed:.3f}s")
        except IOError as e:
            print(f"✗ Error writing OBJ file to '{out_path}': {e}")
        except Exception as e_gen:
             print(f"✗ An unexpected error occurred during OBJ export: {e_gen}")

    # ──────────────────── Flatten Geometry for Renderer ────────────────────
    def flatten_geometry(self) -> Optional[Dict[str, Any]]:
        """
        Converts the scene's objects into flat NumPy arrays required by the Renderer.

        Returns:
            A dictionary containing:
            - v0s: (N, 3) float32 array of triangle vertex 0 positions.
            - e1s: (N, 3) float32 array of triangle edge 1 vectors (v1 - v0).
            - e2s: (N, 3) float32 array of triangle edge 2 vectors (v2 - v0).
            - normals: (N, 3) float32 array of face normals.
            - labels: (N,) int32 array of semantic labels per triangle.
            - colors: (N, 3) uint8 array of colors per triangle.
            - label_map: Dictionary mapping used label IDs to type names.
            - color_map: Dictionary mapping used label IDs to colors.
            Returns None if the scene is empty.
        """
        print("\n▶ 正在为渲染器展平场景几何体...")
        t0_flatten = time.time()

        if not self.objects:
            print("  场景为空，无法展平。")
            return None

        # 优化：预估大小并使用列表，最后转换为 NumPy 数组
        # 预估总面片数，可以稍微多估算一点
        estimated_triangles = self.total_mesh_elements + 1000  # 加一点余量
        all_v0s_list = []
        all_e1s_list = []
        all_e2s_list = []
        all_normals_list = []
        all_labels_list = []
        all_colors_list = []
        used_label_ids = set()

        processed_triangles = 0
        skipped_degenerate = 0
        skipped_ignored_label = 0

        total_objs = len(self.objects)
        flatten_report_interval = max(1, total_objs // 10)

        print("  ⏳ 开始处理对象...")
        for i, obj in enumerate(self.objects):
            num_obj_vertices = len(obj.vertices)
            if num_obj_vertices == 0:
                continue

            # 对当前对象的所有面片进行处理
            obj_mesh_elements = obj.mesh_elements
            obj_vertices = obj.vertices  # 减少属性访问次数
            obj_type = obj.object_type
            default_obj_label = TYPE_TO_LABEL_MAP.get(obj_type, DEFAULT_LABEL)

            for mesh_element in obj_mesh_elements:
                try:
                    # 确定标签 ID
                    label_id = mesh_element.label_id if mesh_element.label_id is not None else default_obj_label

                    # 跳过需要忽略的标签
                    if label_id == SKY_LABEL or label_id < 0:
                        skipped_ignored_label += 1
                        continue

                    # 获取顶点索引并检查有效性
                    idx_a, idx_b, idx_c = mesh_element.vertex_indices
                    if not (
                            0 <= idx_a < num_obj_vertices and 0 <= idx_b < num_obj_vertices and 0 <= idx_c < num_obj_vertices):
                        continue

                    # 获取顶点坐标
                    # 使用 try-except 捕获可能的无效索引错误 (理论上不应发生，但作为保险)
                    try:
                        v_a_tuple = obj_vertices[idx_a]
                        v_b_tuple = obj_vertices[idx_b]
                        v_c_tuple = obj_vertices[idx_c]
                    except IndexError:
                        continue  # 如果索引无效，跳过这个面

                    # 转换为 NumPy 数组进行计算
                    v_a = np.array(v_a_tuple, dtype=GEOMETRY_DTYPE)
                    v_b = np.array(v_b_tuple, dtype=GEOMETRY_DTYPE)
                    v_c = np.array(v_c_tuple, dtype=GEOMETRY_DTYPE)

                    # 计算 v0, e1, e2
                    v0 = v_a
                    e1 = v_b - v_a
                    e2 = v_c - v_a

                    # 计算法线并检查退化三角形
                    normal = np.cross(e1, e2)
                    norm_len_sq = np.dot(normal, normal)
                    if norm_len_sq < 1e-18:  # 使用更小的容差判断退化
                        skipped_degenerate += 1
                        continue

                    # 标准化法线
                    normal /= np.sqrt(norm_len_sq)

                    # 获取颜色
                    triangle_color = mesh_element.color

                    # 添加到列表
                    all_v0s_list.append(v0)
                    all_e1s_list.append(e1)
                    all_e2s_list.append(e2)
                    all_normals_list.append(normal)
                    all_labels_list.append(label_id)
                    all_colors_list.append(triangle_color)
                    used_label_ids.add(label_id)
                    processed_triangles += 1

                except Exception as e:
                    # print(f"  🔴 处理对象 '{obj.object_id}' 的面片 {mesh_element.mesh_id} 时发生未知错误: {e}")
                    pass  # 减少错误输出

            # 展平进度报告
            if (i + 1) % flatten_report_interval == 0 or (i + 1) == total_objs:
                print(f"\r  ● 展平进度: {i + 1}/{total_objs} 个对象...", end="")
        print()  # 换行

        if not all_v0s_list:
            print("  未处理任何有效的三角形用于展平。")
            return None

        print("  ⏳ 正在将列表转换为 NumPy 数组...")
        t_convert_start = time.time()
        # 最终将列表转换为 NumPy 数组
        result = {
            "v0s": np.array(all_v0s_list, dtype=GEOMETRY_DTYPE),
            "e1s": np.array(all_e1s_list, dtype=GEOMETRY_DTYPE),
            "e2s": np.array(all_e2s_list, dtype=GEOMETRY_DTYPE),
            "normals": np.array(all_normals_list, dtype=GEOMETRY_DTYPE),
            "labels": np.array(all_labels_list, dtype=LABEL_DTYPE),
            "colors": np.array(all_colors_list, dtype=COLOR_DTYPE),
        }
        t_convert_end = time.time()
        print(f"  ✔ NumPy 数组转换完成 (耗时: {t_convert_end - t_convert_start:.3f}s)")

        # --- 生成最终的标签和颜色映射 ---
        print("  ⏳ 正在生成标签和颜色映射...")
        final_label_map = {}
        final_color_map = {}
        reverse_type_map_full = {v: k for k, v in TYPE_TO_LABEL_MAP.items()}
        for label_id in used_label_ids:
            type_str = reverse_type_map_full.get(label_id, f"UnknownLabel_{label_id}")
            final_label_map[label_id] = type_str
            color = COLOR_MAP.get(type_str)
            if color is not None:
                final_color_map[label_id] = color  # <--- 修改点：直接存储 numpy 数组
            else:
                final_color_map[label_id] = DEFAULT_COLOR  # <--- 修改点：直接存储 numpy 数组

        result["label_map"] = final_label_map
        result["color_map"] = final_color_map
        print("  ✔ 标签和颜色映射生成完毕。")

        elapsed = time.time() - t0_flatten
        print(f"✔ 完成几何体展平: 处理了 {processed_triangles} 个有效三角形。")
        print(f"   (跳过 {skipped_degenerate} 个退化三角形, 跳过 {skipped_ignored_label} 个忽略标签的三角形)")
        print(f"   展平总耗时: {elapsed:.3f}s")

        # 打印最终数组形状以供检查
        # print("  展平后数组形状:")
        # for key, arr in result.items():
        #     if isinstance(arr, np.ndarray):
        #         print(f"    {key}: {arr.shape}")

        return result


if __name__ == "__main__":
    print("=====================================")
    print("  Scene Generation Script Started")
    print("=====================================")

    # --- 配置 ---
    # Use raw strings (r"...") or forward slashes for paths
    JSON_MODEL_PATH = r"D:\Code\UrbanWindowVisionKG\library\HK_map\processed\project\processed_model.json"
    OBJ_TERRAIN_PATH = r"D:\Code\UrbanWindowVisionKG\library\HK_map\processed\project\北侧地形.obj"
    TREE_PATH = r"D:\Code\UrbanWindowVisionKG\library\HK_map\row_map\greening\tree20250203_converted.json"
    MORE_TREE_PATH = r"D:\Code\UrbanWindowVisionKG\library\HK_map\row_map\greening\VIS_INV_TREE_CSDI_202503171731_converted.json"
    MOREMORE_TREE_PATH = r"D:\Code\UrbanWindowVisionKG\library\HK_map\row_map\greening\dataset_29_layer.json"
    OUTPUT_OBJ_PATH = "demo_scene_v4_detailed.obj" # Changed output name slightly
    ELEMENT_LIMIT = None # 设置为 None 处理所有元素，或设置为一个数字 (e.g., 1000) 来限制处理数量
    REPORT_INTERVAL = 1000 # 每处理 1000 个元素汇报一次

    # --- 准备工作 ---
    scene = Scene()
    # scene.plan_trees(TREE_PATH)
    # scene.plan_trees_v2(MORE_TREE_PATH)
    # scene.plan_trees_v3(MOREMORE_TREE_PATH)

    # --- 加载 DataModel ---
    print(f"\n▶ Loading DataModel from: {JSON_MODEL_PATH}")
    model = ARTShapelyDataExchanger.read_json_file(JSON_MODEL_PATH)
    elements_to_load = model.elements

    # --- 处理 DataElements ---
    WINDOW_METADATA_OUTPUT_PATH = "test_window.json"
    # --- 定义测试用AOI - --
    # !! 同样，这里的坐标需要根据你的 processed_model.json 中的建筑实际位置调整 !!
    TEST_AOI_MIN_X, TEST_AOI_MIN_Y = 830000, 814400
    TEST_AOI_MAX_X, TEST_AOI_MAX_Y = 840500, 824600

    TEST_WINDOW_AOI_WKT = f"POLYGON (({TEST_AOI_MIN_X} {TEST_AOI_MIN_Y}, {TEST_AOI_MAX_X} {TEST_AOI_MIN_Y}, {TEST_AOI_MAX_X} {TEST_AOI_MAX_Y}, {TEST_AOI_MIN_X} {TEST_AOI_MAX_Y}, {TEST_AOI_MIN_X} {TEST_AOI_MIN_Y}))"
    print(f"测试用窗户生成 AOI WKT: {TEST_WINDOW_AOI_WKT}")
    if elements_to_load: # Only proceed if elements were loaded
        scene.load_elements(elements_to_load, report_interval=REPORT_INTERVAL,
                            # --- 传递窗户生成参数 ---
                            generate_windows_flag=True,  # 启用窗户生成
                            window_aoi_wkt=TEST_WINDOW_AOI_WKT,  # 使用测试 AOI
                            window_metadata_output_path=WINDOW_METADATA_OUTPUT_PATH,  # 指定输出路径
                            # window_config={"floor_height_m": 3.1} # 可选：传递自定义配置
                            )
    else:
        print("ℹ️ No DataElements loaded, skipping element processing.")

    # --- 加载 OBJ 地形 ---
    # terrain_path = Path(OBJ_TERRAIN_PATH)
    # if terrain_path.exists():
    #     scene.load_obj(terrain_path, "MountainNorth")
    # else:
    #     print(f"🟡 Warning: OBJ terrain file not found at {OBJ_TERRAIN_PATH}, skipping.")

    # --- 导出场景 ---
    if scene.objects: # Only export if there are objects in the scene
        scene.export_to_obj(OUTPUT_OBJ_PATH)
    else:
        print("\nℹ️ Scene is empty, skipping OBJ export.")

    print("\n=====================================")
    print("  Scene Generation Script Finished")
    print("=====================================")

    # for obj in scene.objects:
    #     if obj.object_type == "MountainNorth":
    #         print(obj)
    #         print("MountainNorth")
    #         print(obj.mesh_elements[:10])
    #         break
    #
    # for obj in scene.objects:
    #     if obj.object_type == "Building":
    #         print(obj)
    #         print("Building")
    #         print(obj.mesh_elements[:10])
    #         break