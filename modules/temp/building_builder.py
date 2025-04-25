"""
Site Builder
读取dxf文件，创建项目周边信息
"""
import copy
from dataclasses import dataclass, field
from typing import AnyStr, List, Dict
import math
from shapely.geometry import Polygon, Point, LineString, MultiPoint, MultiPolygon, GeometryCollection, box
from shapely.ops import orient
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial


@dataclass(order=True)
class SiteIndex:
    site_area: float = field(default=float)
    site_classification: str = field(default="A")
    min_bulding_height: float = field(default=55.00)
    max_bulding_height: float = field(default=61.00)
    permitted_d_plot_ratio: float = field(default=0.00)  # 初始化时不进行计算
    permitted_nd_plot_ratio: float = field(default=0.00)  # 初始化时不进行计算
    permitted_total_plot_ratio: float = field(default=0.00)  # 初始化时不进行计算
    permitted_site_coverage: float = field(default=0.00)  # 初始化时不进行计算

    def export_to_dict(self):
        return {
            "site_area": self.site_area,
            "site_classification": self.site_classification,
            "min_bulding_height": self.min_bulding_height,
            "max_bulding_height": self.max_bulding_height,
            "permitted_d_plot_ratio": self.permitted_d_plot_ratio,
            "permitted_nd_plot_ratio": self.permitted_nd_plot_ratio,
            "permitted_total_plot_ratio": self.permitted_total_plot_ratio,
            "permitted_site_coverage": self.permitted_site_coverage,
        }


@dataclass(order=True)
class SiteBuilder:
    # 所有指标
    site_index: SiteIndex = field(default_factory=SiteIndex)
    # 前端显示用的模型
    model: DataModel = field(default_factory=DataModel)

    __base_height: float = field(default=20)
    __plot_height: float = field(default=0.65)
    __site_height: float = field(default=0.35)
    __inside_area_height: float = field(default=0.15)
    __surrounding_mul_rise_building_height: float = field(default=24)
    __surrounding_high_rise_building_height: float = field(default=80)

    def build(self, input_data_model: DataModel):
        self.model = DataModel(name="site")

        for element in input_data_model.elements:
            if element.layer == "Xkool_ProjectBoundary":
                gen_element = DataElement(geometry=element.geometry, layer=element.layer,
                                          material=DataElementMaterial(color="0xe5ebf1"),
                                          start_height=-(self.__base_height + self.__plot_height + self.__site_height),
                                          height=self.__base_height)
                self.model.insert_element(gen_element)
                self.site_index.site_area = round(element.geometry.area, 2)
            elif element.layer == "Xkool_PlotBoundary":
                gen_element = DataElement(geometry=element.geometry, layer=element.layer,
                                          material=DataElementMaterial(color="0xced4d6"),
                                          start_height=-(self.__plot_height + self.__site_height),
                                          height=self.__plot_height)
                self.model.insert_element(gen_element)
            elif element.layer == "Xkool_SiteBoundary":
                gen_element = DataElement(geometry=element.geometry, layer=element.layer,
                                          material=DataElementMaterial(color="0xc7e29f"),
                                          start_height=-self.__site_height,
                                          height=self.__site_height)
                self.model.insert_element(gen_element)
            elif element.layer == "Xkool_InsideUplacedArea":
                gen_element = DataElement(geometry=element.geometry, layer=element.layer,
                                          material=DataElementMaterial(color="0xbe2323"),
                                          start_height=self.__inside_area_height)
                self.model.insert_element(gen_element)
            elif element.layer == "Xkool_SurroundBuilding":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_SurroundBuilding",
                                          material=DataElementMaterial(color="0xbbbec2"),
                                          height=self.__surrounding_mul_rise_building_height)
                self.model.insert_element(gen_element)
            elif element.layer == "Xkool_SurroundBuildingHigh":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_SurroundBuilding",
                                          material=DataElementMaterial(color="0xbbbec2"),
                                          height=self.__surrounding_high_rise_building_height)
                self.model.insert_element(gen_element)

        self.model.renew()
        self.model.user_data = {"site_index": self.site_index.export_to_dict()}

