# map_structure.py
from dataclasses import field, dataclass
from typing import Optional
from shapely.geometry.base import BaseGeometry
from map_system.utils.row_map_data_reader import MapGeoReader


@dataclass
class MapBaseGeometry:
    # 图形数据
    geometry: Optional[BaseGeometry] = field(default=None) # 使用 None 作为默认值
    geom_type: str = field(default="")
    length: float = field(default=0)
    area: float = field(default=0)
    start_height: float = field(default=0)  # 起始高度
    height: float = field(default=0)  # 拉伸高度
    map_dict: dict = field(default=dict)
    object_id: int = field(default=0)
    row_data: dict = field(default=dict)


@dataclass
class MapRoadCenterLine(MapBaseGeometry):
    # 语义数据
    street_name_en: str = field(default="")
    street_name_tc: str = field(default="")
    last_update_date: str = field(default="")
    # 节点数据，暂时还用不到
    street_center_line_id: int = field(default=0)
    street_code: int = field(default=0)

    def init(self, row_data: dict):
        self.row_data = row_data
        geo = MapGeoReader.geo_reader(row_data["geometry"]["type"], row_data["geometry"]["coordinates"])
        if geo is not None:
            self.geometry = geo
            self.geom_type = geo.geom_type
            self.length = row_data['properties']['SHAPE_Length']
            self.street_name_en = row_data['properties']['STREET_NAME_EN'] if row_data['properties']['STREET_NAME_EN'] is not None else ""
            self.street_name_tc = row_data['properties']['STREET_NAME_TC'] if row_data['properties']['STREET_NAME_TC'] is not None else ""
            self.last_update_date = row_data['properties']['LAST_UPDATE_DATE']
            self.object_id = row_data['properties']['OBJECTID']
            self.street_center_line_id = row_data['properties']['STREET_CENTRELINE_ID']
            self.street_code = row_data['properties']['STREET_CODE']
        else:
            raise Exception(f"道路中心线数据读取失败，道路object_id: {row_data['properties']['OBJECTID']}, 道路坐标{row_data['geometry']['coordinates']}")


@dataclass
class MapBuilding(MapBaseGeometry):
    building_structure_id: str = field(default="")
    building_csuid: str = field(default="")
    building_structure_type: str = field(default="")
    official_building_name_tc: str = field(default="")
    official_building_name_en: str = field(default="")
    num_above_ground_storeys: int = field(default=0)
    category: str = field(default="")
    status: str = field(default="")

    def init(self, row_data: dict):
        self.row_data = row_data
        geo = MapGeoReader.geo_reader(row_data["geometry"]["type"], row_data["geometry"]["coordinates"])
        if geo is not None:
            self.geometry = geo
            self.geom_type = geo.geom_type
            self.length = row_data['properties']['SHAPE_Length'] if row_data['properties']['SHAPE_Length'] is not None else geo.length
            self.area = row_data['properties']['SHAPE_Area'] if row_data['properties']['SHAPE_Area'] is not None else geo.area
            self.object_id = row_data['properties']['OBJECTID']
            self.start_height = row_data['properties']['BASEHEIGHT'] if row_data['properties']['BASEHEIGHT'] is not None else -0.5
            # 这里补充一个建筑最低高度的处理
            # if self.start_height < -10:
            #     self.start_height = -10
            top_height = row_data['properties']['TOPHEIGHT'] if row_data['properties']['TOPHEIGHT'] is not None else 5
            self.height = round(top_height - self.start_height, 2)
        else:
            raise Exception(f"建筑数据读取失败，建筑object_id: {row_data['properties']['OBJECTID']}, 建筑坐标{row_data['geometry']['coordinates']}")

        p = row_data["properties"]  # 方便书写
        self.building_structure_id = p.get("BUILDINGSTRUCTUREID", "")
        self.building_csuid = p.get("BUILDINGCSUID", "")
        self.building_structure_type = p.get("BUILDINGSTRUCTURETYPE", "")
        self.category = p.get("CATEGORY", "")
        self.status = p.get("STATUS", "")
        self.official_building_name_en = p.get("OFFICIALBUILDINGNAMEEN", "")
        self.official_building_name_tc = p.get("OFFICIALBUILDINGNAMETC", "")
        self.num_above_ground_storeys = p.get("NUMABOVEGROUNDSTOREYS", 0)


@dataclass
class MapLot(MapBaseGeometry):
    # 数据源
    is_GLA: bool = field(default=False)
    # 地块类别
    lot_type: str = field(default="site")
    # lot信息
    lot_id: int = field(default=-1)
    lot_csu_id: int = field(default=0)
    # GLA信息
    gla_id: int = field(default=-1)
    gla_code: int = field(default=0)
    gla_number: int = field(default=0)

    def init(self, row_data: dict):
        self.row_data = row_data
        geo = MapGeoReader.geo_reader(row_data["geometry"]["type"], row_data["geometry"]["coordinates"])
        if geo is not None:
            self.geometry = geo
            self.geom_type = geo.geom_type
            self.length = row_data['properties']['SHAPE_Length']
            self.area = row_data['properties']['SHAPE_Area']
            self.object_id = row_data['properties']['OBJECTID']
            if "LOTID" in row_data['properties'].keys():
                self.lot_id = row_data['properties']['LOTID']
                self.lot_csu_id = row_data['properties']['LOTCSUID']
            else:
                self.is_GLA = True
                self.gla_id = row_data['properties']['GLAID']
                self.gla_code = row_data['properties']['GLACODE']
                self.gla_number = row_data['properties']['GLANUMBER']
        else:
            raise Exception(f"地块数据读取失败，地块object_id: {row_data['properties']['OBJECTID']}, 地块坐标{row_data['geometry']['coordinates']}")


@dataclass
class MapRoadPolygon(MapBaseGeometry):
    # 道路类型数据
    feat_type: str = field(default_factory=str)
    width: float = field(default=0)

    def init(self, row_data: dict):
        self.row_data = row_data
        geo = MapGeoReader.geo_reader(row_data["geometry"]["type"], row_data["geometry"]["coordinates"])
        if geo is not None:
            self.geometry = geo
            self.geom_type = geo.geom_type
            self.length = row_data['properties']['Shape_Length']
            self.area = row_data['properties']['Shape_Area']
            self.object_id = row_data['properties']['OBJECTID']
            self.feat_type = row_data['properties']['FEAT_TYPE']
        else:
            raise Exception(f"道路图形数据读取失败，道路object_id: {row_data['properties']['OBJECTID']}, 道路坐标{row_data['geometry']['coordinates']}")


@dataclass
class MapAdminBoundary(MapBaseGeometry):

    def init(self, row_data: dict):
        self.row_data = row_data
        geo = MapGeoReader.geo_reader(row_data["geometry"]["type"], row_data["geometry"]["coordinates"])
        if geo is not None:
            self.geometry = geo
            self.geom_type = geo.geom_type
        else:
            raise Exception(f"区域图形数据读取失败，边界id: {row_data['properties']['OBJECTID']}, 边界坐标{row_data['geometry']['coordinates']}")


@dataclass
class MapGreeningBoundary(MapBaseGeometry):
    def init(self, row_data: dict):
        self.row_data = row_data
        geo = MapGeoReader.geo_reader(row_data["geometry"]["type"], row_data["geometry"]["coordinates"])
        if geo is not None:
            self.geometry = geo
            self.geom_type = geo.geom_type
        else:
            raise Exception(f"绿化图形数据读取失败，绿化坐标{row_data['geometry']['coordinates']}")