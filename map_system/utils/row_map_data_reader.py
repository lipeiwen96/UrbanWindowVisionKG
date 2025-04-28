"""
读取地图数据原生图形，转化为shapely格式
"""
from shapely.geometry import Polygon, Point, LineString, GeometryCollection, box, MultiPolygon


class MapGeoReader:
    @staticmethod
    def geo_reader(geo_type: str, geo_coords: list):
        geo = None
        if geo_type == "LineString":
            geo = MapGeoReader.read_linestring(geo_coords)
        elif geo_type == "MultiLineString":
            # 读取图形中暂无这个数据格式
            pass
        elif geo_type == "Polygon":
            geo = MapGeoReader.read_polygon(geo_coords)
        elif geo_type == "MultiPolygon":
            geo = MapGeoReader.read_multipolygon(geo_coords)
            return geo

        # 将无效的图形统一用None表示
        if geo is not None:
            if geo.is_empty:
                geo = None
            if not geo.is_valid:
                # print(geo)
                # print(geo.is_valid)
                # 对输入的有自交点的无效多边形，提取其外轮廓
                exterior_geo = MapGeoReader.get_extertior_polygon_of_self_crossing_geo(geo)
                if exterior_geo is not None:
                    if exterior_geo.is_empty:
                        geo = None
                    elif exterior_geo.is_valid:
                        geo = exterior_geo
                    else:
                        geo = None
                else:
                    geo = None

        return geo

    @staticmethod
    def read_linestring(geo_coords: list):
        pts = [(pt[0], pt[1]) for pt in geo_coords]
        linestring = LineString(pts)
        return linestring

    @staticmethod
    def read_polygon(geo_coords: list):
        """
        读取图形，图形有可能带有洞
        """
        geo_num = len(geo_coords)
        if geo_num == 1:
            return Polygon([(pt[0], pt[1]) for pt in geo_coords[0]])
        else:
            exterior = [(pt[0], pt[1]) for pt in geo_coords[0]]
            interior_list = []
            for coords in geo_coords[1:]:
                interior = [(pt[0], pt[1]) for pt in coords]
                interior_list.append(interior)
            return Polygon(shell=exterior, holes=interior_list)

    @staticmethod
    def read_multipolygon(geo_coords: list):
        polygon_list = []
        for single_geo_coords in geo_coords:
            polygon = MapGeoReader.read_polygon(single_geo_coords)
            polygon_list.append(polygon)
        return MultiPolygon(polygon_list)

    @staticmethod
    def get_extertior_polygon_of_self_crossing_geo(geo):
        """
        对输入的有自交点的无效多边形，提取其外轮廓
        """
        buffer_geo = geo.buffer(0)
        if isinstance(buffer_geo, MultiPolygon):
            # 过滤掉空的polygon
            # --- 修改下面这行 ---
            # non_empty_polygons = [polygon for polygon in buffer_geo if not polygon.is_empty] # 旧代码 (错误)
            non_empty_polygons = [polygon for polygon in buffer_geo.geoms if not polygon.is_empty]  # 新代码 (正确)
            # --- 修改结束 ---

            # 添加一个检查：如果过滤后列表为空怎么办？
            if not non_empty_polygons:
                # 这里需要处理 buffer(0) 产生的所有多边形都是空的情况
                # 你可以选择返回 None，或者引发一个特定的错误，或者返回一个空的 Polygon
                # 例如：
                # print(f"Warning: buffer(0) resulted in only empty polygons for original geometry.")
                return None  # 或者返回 Polygon()

            # 按面积排序（注意：当前是升序，[0]是最小的）
            non_empty_polygons = sorted(non_empty_polygons, key=lambda polygon: polygon.area)

            # --- 潜在逻辑问题 ---
            # 这里选择了面积最小的多边形 [0]。
            # 函数注释说“提取其外轮廓”，通常这意味着想要最大的那个部分。
            # 你确定需要选择最小的吗？如果需要最大的，应该用 non_empty_polygons[-1]
            buffer_geo = non_empty_polygons[0]  # 当前选择最小的
            # 如果需要最大的，可以改为: buffer_geo = non_empty_polygons[-1]
            # --- 潜在逻辑问题结束 ---

        # 确保 buffer_geo 现在是一个 Polygon (或者是原始有效的 Polygon/修复后变成的 Polygon)
        # 如果 buffer_geo 在上面处理后可能是 None，需要检查
        if buffer_geo is None or buffer_geo.is_empty:
            return None

        # 如果 buffer_geo 不是 Polygon 类型（虽然理论上 buffer(0) 后应该是），则无法获取 exterior
        if not isinstance(buffer_geo, Polygon):
            print(f"Warning/Error: buffer_geo is not a Polygon after processing: {type(buffer_geo)}")
            # 根据情况处理，可能返回 None 或 buffer_geo 本身？
            return None  # 或者返回 buffer_geo 如果可以接受非 Polygon

        # 提取外轮廓
        try:
            exterior_geo = Polygon(buffer_geo.exterior)
            # 再次检查结果是否有效且非空
            if exterior_geo.is_empty or not exterior_geo.is_valid:
                # print(f"Warning: Extracted exterior is empty or invalid.")
                return None  # 或者尝试其他修复？
            return exterior_geo
        except Exception as e:
            print(f"Error extracting exterior from buffer_geo: {e}")
            return None
