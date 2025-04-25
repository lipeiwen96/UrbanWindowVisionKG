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

        # 将无效的图形统一用None表示
        if geo is not None:
            if geo.is_empty:
                geo = None
            if not geo.is_valid:
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
            non_empty_polygons = [polygon for polygon in buffer_geo if not polygon.is_empty]
            non_empty_polygons = sorted(non_empty_polygons, key=lambda polygon: polygon.area)
            buffer_geo = non_empty_polygons[0]
        exterior_geo = Polygon(buffer_geo.exterior)
        return exterior_geo
