# window_generator.py
import uuid
import json
from pathlib import Path
# 明确导入 typing 中的类型，避免与内置类型混淆
from typing import List, Dict, Tuple, Optional, Any, Union, Sequence
from dataclasses import dataclass, field
import numpy as np
# 导入必要的 Shapely 类型
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely.wkt import loads as wkt_loads
import time

# --- 在此模块中直接定义所需的类型别名和常量 ---
# 避免从 scene_structure 导入

# 定义 NumPy 数据类型常量
GEOMETRY_DTYPE = np.float32
INDEX_DTYPE = np.int32
COLOR_DTYPE = np.uint8

# 定义类型别名
Vec3 = Tuple[float, float, float]
TriIndices = Tuple[int, int, int]
PolygonLike = Union[Polygon, MultiPolygon] # 用于类型提示

# 定义窗口颜色
WINDOW_COLOR = np.array([173, 216, 230], dtype=COLOR_DTYPE) # 浅蓝色窗户

# --- 常量 ---
DEFAULT_FLOOR_HEIGHT_M = 3.0
DEFAULT_WINDOW_HEIGHT_M = 1.5
DEFAULT_WINDOW_WIDTH_M = 1.2
DEFAULT_WINDOW_SPACING_M = 0.8 # 窗户之间以及墙体末端的最小间距
# 修正计算：需要窗户宽度 + *两侧* 的间距
DEFAULT_MIN_WALL_LENGTH_M = DEFAULT_WINDOW_WIDTH_M + 2 * DEFAULT_WINDOW_SPACING_M
DEFAULT_SILL_HEIGHT_M = 0.9 # 从楼板到窗户底部的距离
DEFAULT_WINDOW_OFFSET_M = 0.05 # 窗户从墙面向外偏移的距离


# --- 元数据结构 ---
@dataclass
class WindowMetadata:
    """存储单个窗户的元数据信息"""
    window_id: str = field(default_factory=lambda: f"win_{uuid.uuid4()}")
    building_id: str = "unknown_building" # 所属建筑的ID
    building_type: str = "Unknown"        # 所属建筑的类型
    floor_number: int = -1                # 楼层编号 (从0开始)
    wall_segment_index: int = -1          # 在建筑轮廓上的墙体段索引
    window_index_on_wall: int = -1        # 在该墙体段上的窗户索引
    wall_start_coord: Tuple[float, float] = (0.0, 0.0) # 墙体段起点 (XY)
    wall_end_coord: Tuple[float, float] = (0.0, 0.0)   # 墙体段终点 (XY)
    wall_normal: Tuple[float, float] = (0.0, 0.0)      # 墙体外法线 (XY平面)
    window_center_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0) # 窗户中心点 (3D)
    window_width: float = 0.0             # 窗户宽度
    window_height: float = 0.0            # 窗户高度
    # 注意：根据之前的讨论，移除了 window_mesh_vertices 和 window_mesh_faces
    # 以减小元数据文件的大小。几何信息由 generate_windows 的第二个返回值提供。


# --- 窗户生成器类 ---
class WindowGenerator:
    """
    根据建筑信息在AOI内生成窗户的几何数据和元数据。
    可以选择只处理每个多边形部分最长的N条墙段。
    """
    def __init__(
        self,
        floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
        window_height_m: float = DEFAULT_WINDOW_HEIGHT_M,
        window_width_m: float = DEFAULT_WINDOW_WIDTH_M,
        window_spacing_m: float = DEFAULT_WINDOW_SPACING_M,
        min_wall_length_m: float = DEFAULT_MIN_WALL_LENGTH_M,
        sill_height_m: float = DEFAULT_SILL_HEIGHT_M,
        window_offset_m: float = DEFAULT_WINDOW_OFFSET_M,
        skip_ground_floor: bool = True,
        max_segments_per_polygon: int = 6 # 每个多边形部分最多处理的墙段数
    ):
        """
        初始化窗户生成器。

        参数:
            floor_height_m (float): 楼层高度 (米)。
            window_height_m (float): 窗户高度 (米)。
            window_width_m (float): 窗户宽度 (米)。
            window_spacing_m (float): 窗户间距 (米)。
            min_wall_length_m (float): 可生成窗户的最小墙长 (米)。
            sill_height_m (float): 窗台高度 (米)。
            window_offset_m (float): 窗户向外偏移距离 (米)。
            skip_ground_floor (bool): 是否跳过首层 (地面层) 不生成窗户。
            max_segments_per_polygon (int): 对每个建筑轮廓（或多边形部分），
                                           只处理长度排名前N的墙段。设为0或负数则处理所有符合长度的墙段。
        """
        self.floor_height = max(0.1, floor_height_m)
        self.window_height = max(0.1, window_height_m)
        self.window_width = max(0.1, window_width_m)
        self.window_spacing = max(0.0, window_spacing_m)
        # 如果提供的最小墙长不足以容纳一个窗户+两侧间距，则使用计算出的最小值
        calculated_min = self.window_width + 2 * self.window_spacing
        self.min_wall_length = max(calculated_min, min_wall_length_m)
        self.sill_height = max(0.0, sill_height_m)
        self.window_offset = window_offset_m
        self.skip_ground_floor = skip_ground_floor
        # 如果 max_segments_per_polygon <= 0，则设为一个非常大的数，相当于不限制
        self.max_segments_per_polygon = max_segments_per_polygon if max_segments_per_polygon > 0 else float('inf')

        # 检查窗户高度+窗台高度是否超过层高，如果超过则调整窗台高度
        if self.window_height + self.sill_height > self.floor_height:
            print(f"⚠️ 警告: 窗户高度 ({self.window_height}m) + 窗台高度 ({self.sill_height}m) "
                  f"超过层高 ({self.floor_height}m)。正在调整窗台高度。")
            self.sill_height = max(0.0, self.floor_height - self.window_height)
            print(f"   调整后窗台高度: {self.sill_height:.2f}m")

        print("\n--- 窗户生成器已初始化 ---")
        print(f"  层高: {self.floor_height:.2f}m, 窗高: {self.window_height:.2f}m, "
              f"窗宽: {self.window_width:.2f}m, 窗间距: {self.window_spacing:.2f}m")
        print(f"  最小墙长: {self.min_wall_length:.2f}m, 窗台高: {self.sill_height:.2f}m, "
              f"向外偏移: {self.window_offset:.2f}m")
        print(f"  跳过首层: {'是' if self.skip_ground_floor else '否'}")
        if self.max_segments_per_polygon != float('inf'):
            print(f"  每部分最多处理墙段数: {self.max_segments_per_polygon}")
        else:
            print("  每部分处理所有符合长度的墙段")
        print("------------------------------------\n")

    def _calculate_polygon_height(self, building_vertices: Sequence[Vec3]) -> Tuple[float, float]:
        """
        根据建筑对象的顶点列表估算最低Z值 (start_z) 和最高Z值 (top_z)。
        参数:
             building_vertices: 包含建筑顶点 (x, y, z) 元组的序列。
        返回:
             一个元组 (min_z, max_z)。如果顶点列表为空或无效，则返回 (0.0, 0.0)。
        """
        if not building_vertices:
            return 0.0, 0.0
        try:
            # 将顶点列表转换为 NumPy 数组进行计算
            verts_np = np.array(building_vertices, dtype=GEOMETRY_DTYPE)
            # 检查数组形状是否正确 (N, 3)
            if verts_np.ndim != 2 or verts_np.shape[1] != 3 or verts_np.shape[0] == 0:
                 print("  🟡 警告: 用于计算高度的顶点数据形状无效，将返回 (0, 0)。")
                 return 0.0, 0.0
            # 计算 Z 轴的最小值和最大值
            min_z = np.min(verts_np[:, 2])
            max_z = np.max(verts_np[:, 2])
            return float(min_z), float(max_z)
        except Exception as e:
            print(f"  🔴 错误: 计算建筑高度时出错: {e}")
            return 0.0, 0.0

    def _create_window_mesh_data(
        self,
        center_point_xy: np.ndarray,    # (2,) 数组，窗户在墙面上的XY中心点
        wall_normal_xy: np.ndarray,     # (2,) 数组，标准化的外向法向量 (XY平面)
        wall_direction_xy: np.ndarray,  # (2,) 数组，标准化的墙体方向向量 (XY平面)
        base_z: float                   # 窗户底部的Z坐标
    ) -> Tuple[List[Vec3], List[TriIndices]]:
        """为单个矩形窗户网格创建顶点和面索引。"""

        # 应用向外偏移量，得到窗户几何体在空间中的中心线位置
        offset_center_xy = center_point_xy + wall_normal_xy * self.window_offset

        # 计算窗户半宽度在墙体方向上的三维向量 (dx, dy, 0)
        half_width_offset_xy = wall_direction_xy * (self.window_width / 2.0)
        half_width_vec_3d = np.array([half_width_offset_xy[0], half_width_offset_xy[1], 0.0], dtype=GEOMETRY_DTYPE)

        # 计算窗户半高度的三维向量 (0, 0, dz)
        half_height_vec = np.array([0.0, 0.0, self.window_height / 2.0], dtype=GEOMETRY_DTYPE)

        # 计算窗户几何体在空间中的三维中心点
        # (偏移后的XY + Z方向中心)
        window_center_3d = np.array([offset_center_xy[0], offset_center_xy[1], base_z + self.window_height / 2.0], dtype=GEOMETRY_DTYPE)

        # 定义4个角点 (相对于 window_center_3d)
        # 顺序: 左上(TL), 右上(TR), 右下(BR), 左下(BL)
        # 保证从外侧看是逆时针顺序，以确保法线朝外
        p1 = window_center_3d - half_width_vec_3d + half_height_vec # 左上角 (Top Left)
        p2 = window_center_3d + half_width_vec_3d + half_height_vec # 右上角 (Top Right)
        p3 = window_center_3d + half_width_vec_3d - half_height_vec # 右下角 (Bottom Right)
        p4 = window_center_3d - half_width_vec_3d - half_height_vec # 左下角 (Bottom Left)

        # 顶点列表 (转换为元组)
        vertices: List[Vec3] = [
            tuple(p1.tolist()), tuple(p2.tolist()),
            tuple(p3.tolist()), tuple(p4.tolist())
        ]

        # 面索引列表 (使用顶点的本地索引, 0-based) - 保证逆时针
        # 三角面 1: p1, p4, p3 (索引 0, 3, 2) -> 左上, 左下, 右下
        # 三角面 2: p1, p3, p2 (索引 0, 2, 1) -> 左上, 右下, 右上
        faces: List[TriIndices] = [(0, 3, 2), (0, 2, 1)]

        return vertices, faces

    def generate_windows(
        self,
        # 使用 Any 避免导入 Object，调用者需保证传入的对象包含所需属性
        building_objects: List[Any],
        aoi_polygon: Polygon
    ) -> Tuple[List[Dict], Dict[str, List[Tuple[List[Vec3], List[TriIndices]]]]]:
        """
        主函数，为 AOI 内的建筑生成窗户几何数据和元数据。

        参数:
            building_objects: 包含建筑信息的对象列表。每个对象需要有:
                              - object_id (str): 建筑唯一标识符。
                              - object_type (str): 建筑类型字符串。
                              - vertices (List[Vec3]): 建筑的三维顶点列表。
                              - source_geometry (PolygonLike): 建筑的二维轮廓 (Shapely Polygon 或 MultiPolygon)。
            aoi_polygon: 定义感兴趣区域的 Shapely Polygon。

        返回:
            一个元组，包含:
            - window_metadata_list (List[Dict]): 包含所有生成窗户元数据的字典列表。
            - windows_geometry_by_building (Dict[str, List[Tuple[List[Vec3], List[TriIndices]]]]]):
              一个字典，键是 building_id，值是一个列表，列表中的每个元素代表一个窗户的几何数据
              (vertices, faces)，其中 vertices 是该窗户的顶点列表，faces 是该窗户的面索引列表。
        """
        window_metadata_list: List[Dict] = []
        windows_geometry_by_building: Dict[str, List[Tuple[List[Vec3], List[TriIndices]]]] = {}
        buildings_intersecting_aoi = 0 # 统计与AOI相交的建筑数量
        windows_generated_total = 0    # 统计生成的窗户几何数据组总数
        processed_building_ids = set() # 记录实际生成了窗户的建筑ID

        total_buildings_input = len(building_objects) # 输入的建筑总数
        buildings_processed_count = 0 # 初始化已处理建筑计数器
        start_time_generate = time.time() # 记录窗户生成开始时间

        print(f"🏢 开始为 AOI 内的建筑生成窗户...")
        aoi_bounds = aoi_polygon.bounds
        print(f"   AOI 范围: MinX={aoi_bounds[0]:.1f}, MinY={aoi_bounds[1]:.1f}, "
              f"MaxX={aoi_bounds[2]:.1f}, MaxY={aoi_bounds[3]:.1f}")

        for building_obj in building_objects:
            buildings_processed_count += 1
            building_id = getattr(building_obj, 'object_id', f'未知ID_{buildings_processed_count}')
            building_type = getattr(building_obj, 'object_type', '未知类型')

            # --- 0. 检查输入对象是否包含必要属性 ---
            if not hasattr(building_obj, 'source_geometry') or not hasattr(building_obj, 'vertices'):
                print(f"  🟡 警告: 跳过建筑 {building_id}，缺少 'source_geometry' 或 'vertices' 属性。")
                continue
            building_footprint: Optional[PolygonLike] = building_obj.source_geometry
            building_vertices: List[Vec3] = building_obj.vertices

            if building_footprint is None:
                 print(f"  🟡 警告: 跳过建筑 {building_id}，'source_geometry' 为空。")
                 continue
            if not isinstance(building_footprint, (Polygon, MultiPolygon)):
                print(f"  🟡 警告: 跳过建筑 {building_id}，'source_geometry' 类型不受支持 ({type(building_footprint)})。")
                continue

            # --- 1. 验证和修复建筑轮廓 ---
            if not building_footprint.is_valid:
                 # print(f"  ⏳ 尝试修复建筑 {building_id} 的无效轮廓...")
                 building_footprint = make_valid(building_footprint)
                 if not building_footprint.is_valid or building_footprint.is_empty:
                      print(f"  🔴 错误: 跳过建筑 {building_id}，轮廓修复失败或变为空。")
                      continue
                 # 修复后可能变成 MultiPolygon 或 GeometryCollection，需要重新检查
                 if not isinstance(building_footprint, (Polygon, MultiPolygon)):
                     print(f"  🔴 错误: 跳过建筑 {building_id}，轮廓修复后类型变为 {type(building_footprint)}，无法处理。")
                     continue

            # --- 2. 检查是否与 AOI 相交 ---
            try:
                if not aoi_polygon.intersects(building_footprint):
                    # print(f"  ⚪️ 信息: 建筑 {building_id} 不在 AOI 内，跳过。") # 可以取消注释以获取更详细日志
                    continue
            except Exception as e:
                print(f"  🔴 错误: 检查建筑 {building_id} 与 AOI 相交时出错: {e}")
                continue

            # 如果代码执行到这里，说明建筑与 AOI 相交
            buildings_intersecting_aoi += 1
            windows_generated_building = 0 # 重置当前建筑的窗户计数

            # --- 3. 估算楼层 ---
            start_z, top_z = self._calculate_polygon_height(building_vertices)
            building_height = top_z - start_z
            # print(f"   建筑 {building_id}: 高度={building_height:.2f}m (Z范围: {start_z:.2f} -> {top_z:.2f})") # 调试信息

            if building_height < self.floor_height:
                # print(f"  🟡 警告: 跳过建筑 {building_id}，高度 ({building_height:.2f}m) 不足一个楼层 ({self.floor_height:.2f}m)。")
                continue

            num_floors = int(np.floor(building_height / self.floor_height))
            start_floor_index = 1 if self.skip_ground_floor else 0 # 根据配置决定起始楼层索引
            if num_floors <= start_floor_index:
                # print(f"  🟡 警告: 跳过建筑 {building_id}，有效楼层数 ({num_floors - start_floor_index}) 不足。")
                continue
            # print(f"     估算楼层数: {num_floors}, 起始楼层索引: {start_floor_index}") # 调试信息


            # --- 4. 处理建筑轮廓的各个部分 (Polygon 或 MultiPolygon) ---
            polygons_to_process: List[Polygon] = []
            if isinstance(building_footprint, Polygon):
                 polygons_to_process = [building_footprint]
            elif isinstance(building_footprint, MultiPolygon):
                 # 仅处理 MultiPolygon 中的 Polygon 部分
                 polygons_to_process = [p for p in building_footprint.geoms if isinstance(p, Polygon)]

            for poly_index, poly in enumerate(polygons_to_process):
                # print(f"    处理多边形部分 {poly_index + 1}/{len(polygons_to_process)}...") # 调试信息
                # 获取外轮廓坐标，确保至少有3个不同点（4个坐标，首尾相同）
                if poly.exterior is None or len(poly.exterior.coords) < 4:
                    # print(f"     🟡 警告: 跳过多边形部分 {poly_index + 1}，外轮廓坐标不足。")
                    continue
                coords = list(poly.exterior.coords)

                # --- 4.1 选择 N 条最长墙段 ---
                segments_info: List[Dict[str, Any]] = []
                for i in range(len(coords) - 1): # 遍历每条边 (v1 -> v2)
                    v1 = np.array(coords[i])[:2]     # 获取 XY 坐标
                    v2 = np.array(coords[i+1])[:2]   # 获取 XY 坐标
                    wall_vector = v2 - v1
                    wall_length = np.linalg.norm(wall_vector)
                    # 预先过滤掉长度不满足最低要求的墙段
                    if wall_length >= self.min_wall_length:
                        segments_info.append({'index': i, 'length': wall_length, 'v1': v1, 'v2': v2})

                if not segments_info: # 如果没有符合长度的墙段
                    # print(f"     ⚪️ 信息: 多边形部分 {poly_index + 1} 没有符合最小长度要求的墙段。")
                    continue

                # 按长度降序排序
                segments_info.sort(key=lambda x: x['length'], reverse=True)

                # 选择前 N 条或所有符合条件的墙段
                num_segments_to_process = min(len(segments_info), int(self.max_segments_per_polygon)) # 确保是整数
                selected_segments = segments_info[:num_segments_to_process]
                # print(f"      选择了 {len(selected_segments)} 条最长墙段进行处理。") # 调试信息

                # --- 4.2 遍历选定的墙段生成窗户 ---
                for segment_data in selected_segments:
                    original_segment_index: int = segment_data['index']
                    v1: np.ndarray = segment_data['v1']
                    v2: np.ndarray = segment_data['v2']
                    wall_length: float = segment_data['length']

                    # 计算墙体方向和法线
                    wall_vector = v2 - v1
                    # 再次检查长度，避免除零错误 (虽然理论上已被过滤)
                    if wall_length < 1e-9: continue
                    wall_direction = wall_vector / wall_length
                    # 保证外法线：对于逆时针轮廓，(dy, -dx) 指向外部
                    normal = np.array([wall_direction[1], -wall_direction[0]])

                    # 计算该墙段可容纳的窗户数量
                    available_space = wall_length - 2 * self.window_spacing # 两端留空
                    if available_space < self.window_width:
                        num_windows_on_wall = 0
                    else:
                        # 计算一个 "窗户单元" (窗户+右侧间距) 的长度
                        unit_length = self.window_width + self.window_spacing
                        # 计算能放下多少个完整单元
                        num_full_units = int(np.floor(available_space / unit_length))
                        # 计算放下完整单元后剩余的空间
                        remaining_space = available_space - num_full_units * unit_length
                        # 如果剩余空间还能再放一个窗户 (不需要考虑间距了)
                        num_windows_on_wall = num_full_units + (1 if remaining_space >= self.window_width else 0)

                    if num_windows_on_wall <= 0: continue # 如果无法放置窗户，跳过此墙段

                    # 计算窗户居中排列的起始偏移量
                    total_window_width = num_windows_on_wall * self.window_width
                    # 总间距 = (窗户数 - 1) * 单个间距 (如果只有一个窗户，间距为0)
                    total_spacing_width = max(0, num_windows_on_wall - 1) * self.window_spacing
                    # 起始偏移 = (总墙长 - 所有窗户总宽 - 所有间距总宽) / 2
                    start_offset_along_wall = (wall_length - total_window_width - total_spacing_width) / 2.0

                    # --- 4.3 为每个楼层和每个窗户位置生成几何数据和元数据 ---
                    for floor_idx in range(start_floor_index, num_floors):
                        # 计算当前楼层窗户底部的 Z 坐标
                        window_base_z = start_z + floor_idx * self.floor_height + self.sill_height
                        # 检查窗户顶部是否会超过建筑顶部 (加一点容差)
                        if window_base_z + self.window_height > top_z + 1e-3:
                            # print(f"       楼层 {floor_idx}: 窗户顶部 ({window_base_z + self.window_height:.2f}m) "
                            #       f"超过建筑顶部 ({top_z:.2f}m)，停止此墙段更高楼层的生成。") # 调试信息
                            break # 后续楼层肯定也放不下

                        for win_idx_wall in range(num_windows_on_wall):
                            # 计算当前窗户中心点沿墙体方向的距离
                            dist_along_wall = (start_offset_along_wall +
                                               win_idx_wall * (self.window_width + self.window_spacing) +
                                               self.window_width / 2.0)
                            # 计算窗户中心点的 XY 坐标
                            window_center_xy = v1 + wall_direction * dist_along_wall

                            # 生成窗户的顶点和面索引
                            win_vertices, win_faces = self._create_window_mesh_data(
                                window_center_xy, normal, wall_direction, window_base_z
                            )

                            # --- 存储几何数据 ---
                            # 按 building_id 聚合
                            if building_id not in windows_geometry_by_building:
                                windows_geometry_by_building[building_id] = []
                            windows_geometry_by_building[building_id].append((win_vertices, win_faces))

                            windows_generated_total += 1
                            windows_generated_building += 1
                            processed_building_ids.add(building_id) # 记录这个建筑ID确实生成了窗户

                            # --- 创建并存储元数据 ---
                            # 计算用于元数据的3D中心点 (考虑了偏移)
                            meta_center_3d = tuple( (window_center_xy + normal * self.window_offset).tolist() +
                                                    [window_base_z + self.window_height / 2.0] )
                            meta = WindowMetadata(
                                building_id=building_id,
                                building_type=building_type,
                                floor_number=floor_idx,
                                wall_segment_index=original_segment_index, # 使用原始墙段索引
                                window_index_on_wall=win_idx_wall,
                                wall_start_coord=tuple(v1.tolist()),
                                wall_end_coord=tuple(v2.tolist()),
                                wall_normal=tuple(normal.tolist()),
                                window_center_3d=meta_center_3d,
                                window_width=self.window_width,
                                window_height=self.window_height,
                            )
                            window_metadata_list.append(vars(meta)) # 将 dataclass 转为字典存储

            # --- 建筑处理完毕后的进度报告 ---
            report_interval_buildings = 100 # 每处理100个建筑报告一次
            if buildings_processed_count % report_interval_buildings == 0 or buildings_processed_count == total_buildings_input:
                elapsed_time = time.time() - start_time_generate
                # 使用 '\r' 和 end="" 实现原地更新进度条效果 (如果终端支持)
                print(
                    f"\r      -> 进度: 已检查建筑 {buildings_processed_count}/{total_buildings_input}。"
                    f" 已生成窗户几何数据: {windows_generated_total} 组。"
                    f" 已耗时: {elapsed_time:.1f} 秒", end="")

        # --- 所有建筑处理完毕 ---
        print() # 确保进度报告后换行
        final_elapsed_time = time.time() - start_time_generate
        print(f"✔ 窗户生成流程完成。")
        print(f"   总共检查输入建筑数量: {total_buildings_input}")
        print(f"   与AOI相交的建筑数量: {buildings_intersecting_aoi}")
        print(f"   实际生成窗户的建筑数量: {len(processed_building_ids)}")
        print(f"   生成的窗户几何数据组总数 (顶点+面): {windows_generated_total}")
        print(f"   生成的窗户元数据记录数: {len(window_metadata_list)}")
        print(f"   窗户生成总耗时: {final_elapsed_time:.2f} 秒")

        # 返回元数据列表和按建筑ID组织的几何数据字典
        return window_metadata_list, windows_geometry_by_building

    @staticmethod
    def save_metadata(metadata_list: List[Dict], output_path: Path):
        """将收集到的窗户元数据列表保存到 JSON 文件。"""
        print(f"\n💾 正在保存窗户元数据到: {output_path}...")
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                # 使用 default=str 处理可能的非 JSON 序列化类型 (例如 NumPy 类型，如果元数据中意外包含)
                json.dump(metadata_list, f, indent=4, ensure_ascii=False, default=str)
            print(f"   ✔ 成功保存 {len(metadata_list)} 条窗户元数据记录。")
        except Exception as e:
            print(f"   🔴 错误: 保存元数据文件时出错: {e}")


# --- 测试执行块 ---
if __name__ == "__main__":
    print("\n--- 开始运行 WindowGenerator 测试 ---")

    # --- 定义研究区域盒子 (香港1980坐标系) ---
    research_box_coords = (830000, 814400, 840500, 824600)
    rb_min_x, rb_min_y, rb_max_x, rb_max_y = research_box_coords
    print(f"测试范围 (研究区域): X=[{rb_min_x}, {rb_max_x}], Y=[{rb_min_y}, {rb_max_y}]")

    # --- 辅助函数：创建简化的建筑数据对象 (模拟 scene_structure.Object 的接口) ---
    # 注意：这个辅助类只是为了测试，它模拟了 generate_windows 需要的属性
    @dataclass
    class MockBuildingData:
        object_id: str
        object_type: str
        vertices: List[Vec3]
        source_geometry: PolygonLike

    def create_mock_building(obj_id: str, obj_type: str, footprint_coords: List[Tuple[float, float]], start_z: float, height: float) -> Optional[MockBuildingData]:
        """创建一个包含 source_geometry 和模拟顶点列表的 MockBuildingData 对象。"""
        try:
            # 创建并验证轮廓多边形
            poly = Polygon(footprint_coords)
            if not poly.is_valid:
                poly = make_valid(poly)
            if not poly.is_valid or poly.is_empty or not isinstance(poly, Polygon):
                 print(f"错误: 为 {obj_id} 创建的源几何无效。")
                 return None
            # 确保逆时针顺序 (虽然对于窗口生成不强制，但保持一致性)
            if hasattr(poly.exterior, 'is_ccw') and not poly.exterior.is_ccw:
                 poly = Polygon(list(poly.exterior.coords)[::-1])

            # 模拟拉伸后的顶点 (仅用于高度计算测试)
            top_z = start_z + height
            exterior_coords = list(poly.exterior.coords)[:-1] # 排除闭合点
            if not exterior_coords: return None

            # 只创建几个关键顶点用于高度估算，不需要完整网格
            mock_vertices = [
                (exterior_coords[0][0], exterior_coords[0][1], start_z), # 底面一点
                (exterior_coords[1][0], exterior_coords[1][1], start_z), # 底面另一点
                (exterior_coords[0][0], exterior_coords[0][1], top_z),   # 顶面一点
            ]
            if len(exterior_coords) > 2:
                 mock_vertices.append((exterior_coords[2][0], exterior_coords[2][1], top_z)) # 顶面另一点

            return MockBuildingData(
                object_id=obj_id,
                object_type=obj_type,
                vertices=mock_vertices,
                source_geometry=poly
            )
        except Exception as e:
            print(f"错误: 创建模拟建筑 {obj_id} 时出错: {e}")
            return None

    # --- 定义示例建筑 ---
    sample_buildings_data: List[MockBuildingData] = []

    # 建筑 1 (在研究区域内，应被 AOI 覆盖)
    b1_coords = [(831000, 815000), (831050, 815000), (831050, 815030), (831000, 815030), (831000, 815000)] # 4 段
    bldg1 = create_mock_building("Bldg_A_测试", "Building_T", b1_coords, 0.0, 30.0)
    if bldg1: sample_buildings_data.append(bldg1)

    # 建筑 2 (在研究区域内，也应被 AOI 覆盖)
    b2_coords = [(831060, 815000), (831080, 815000), (831075, 815040), (831060, 815000)] # 3 段 (三角形)
    bldg2 = create_mock_building("Bldg_B_测试", "Building_P", b2_coords, 5.0, 15.0) # Z 坐标起始较高
    if bldg2: sample_buildings_data.append(bldg2)

    # 建筑 3 (在研究区域内，但可能在小的 AOI 之外)
    b3_coords = [(830500, 814500), (830550, 814500), (830550, 814550), (830500, 814550), (830500, 814500)] # 4 段
    bldg3 = create_mock_building("Bldg_C_测试", "Building_T", b3_coords, 0.0, 50.0)
    if bldg3: sample_buildings_data.append(bldg3)

    # 建筑 4 (更复杂的轮廓，用于测试墙段选择)
    b4_coords = [ # 8 段，包含短边
        (831000, 815100), (831040, 815100), (831045, 815105), (831050, 815100), # 短边
        (831080, 815100), (831080, 815140), (831000, 815140), (830995, 815135), # 另一短边
        (831000, 815100)
    ]
    bldg4 = create_mock_building("Bldg_D_复杂_测试", "Building_T", b4_coords, 0.0, 40.0)
    if bldg4: sample_buildings_data.append(bldg4)


    print(f"\n创建了 {len(sample_buildings_data)} 个模拟建筑数据对象用于测试。")
    if not sample_buildings_data:
        print("未能创建任何模拟建筑，测试终止。")
    else:
        # --- 定义测试 AOI (确保覆盖 A, B, D) ---
        aoi_wkt = "POLYGON ((830980 814980, 831090 814980, 831090 815150, 830980 815150, 830980 814980))"
        try:
            test_aoi_polygon = wkt_loads(aoi_wkt)
            if not test_aoi_polygon.is_valid: test_aoi_polygon = make_valid(test_aoi_polygon)
            print(f"测试 AOI 多边形已定义 (有效: {test_aoi_polygon.is_valid})。应与建筑 A, B, D 相交。")
        except Exception as e:
            print(f"错误: 创建测试 AOI 多边形时出错: {e}")
            test_aoi_polygon = None

        if test_aoi_polygon and test_aoi_polygon.is_valid:
            # --- 实例化生成器，限制处理墙段数 ---
            generator = WindowGenerator(
                floor_height_m=3.0,
                window_height_m=1.6,
                window_width_m=1.1,
                window_spacing_m=0.9,
                sill_height_m=0.8,
                skip_ground_floor=True,
                max_segments_per_polygon=4 # <<<--- 测试: 限制为4条最长墙段
            )

            # --- 运行生成 ---
            # 传入模拟建筑数据列表
            metadata, geometry_data = generator.generate_windows(
                sample_buildings_data, test_aoi_polygon
            )

            # --- 打印结果 ---
            print(f"\n--- 测试结果 ---")
            processed_building_ids = set(geometry_data.keys())
            print(f"实际生成窗户的建筑 ID: {processed_building_ids}")
            total_windows_generated = sum(len(geom_list) for geom_list in geometry_data.values())
            print(f"生成的窗户几何数据组数量: {total_windows_generated}")
            print(f"生成的元数据记录数量: {len(metadata)}")

            # 检查墙段过滤效果 (通过抽样元数据)
            print("\n抽样元数据 (前15条记录，显示墙段索引):")
            segments_processed_counts = {}
            for i, meta_item in enumerate(metadata[:15]):
                 bldg_id = meta_item.get('building_id')
                 wall_idx = meta_item.get('wall_segment_index')
                 print(f"  窗户 {i+1}: 建筑='{bldg_id}', 楼层={meta_item.get('floor_number')}, 墙段索引={wall_idx}")
                 # 记录每个建筑实际处理了哪些墙段索引
                 if bldg_id not in segments_processed_counts:
                      segments_processed_counts[bldg_id] = set()
                 segments_processed_counts[bldg_id].add(wall_idx)

            print("\n根据抽样元数据统计的每个建筑处理的唯一墙段索引:")
            for bldg_id, indices in segments_processed_counts.items():
                 print(f"  建筑 '{bldg_id}': {sorted(list(indices))} (数量: {len(indices)})")
            print(f"  (注意: 每个建筑处理的墙段数量应 <= {generator.max_segments_per_polygon})")

            # --- 保存元数据测试 ---
            test_metadata_path = Path("./test_window_metadata_独立版.json")
            WindowGenerator.save_metadata(metadata, test_metadata_path)

            # --- 可视化测试 (可选) ---
            print("\n--- 几何集合 (处理过的建筑轮廓 + 窗户中心点) ---")
            geometries_for_collection = []
            # 添加被处理建筑的轮廓
            processed_buildings_dict = {b.object_id: b.source_geometry for b in sample_buildings_data if b.object_id in processed_building_ids}
            for geom in processed_buildings_dict.values():
                 geometries_for_collection.append(geom)
            # 添加窗户中心点
            for meta_item in metadata:
                  center_3d = meta_item.get('window_center_3d')
                  if center_3d and len(center_3d) >= 2:
                       geometries_for_collection.append(Point(center_3d[0], center_3d[1]))

            if geometries_for_collection:
                  geometry_collection = GeometryCollection(geometries_for_collection)
                  print("WKT 输出 (可复制到 WKT 查看器):")
                  wkt_output = geometry_collection.wkt
                  max_wkt_len = 2000 # 限制输出长度
                  print(wkt_output[:max_wkt_len] + ('...' if len(wkt_output) > max_wkt_len else ''))
            else:
                  print("未能生成用于几何集合的可视化数据。")

    print("\n--- WindowGenerator 测试结束 ---")
