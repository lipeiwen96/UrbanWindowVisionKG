"""
Site Builder
读取dxf文件，创建项目周边信息
"""
import copy
from dataclasses import dataclass, field
from typing import AnyStr, List, Dict
from shapely import wkt, affinity
from shapely.geometry import Polygon
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial


# SITE指标
@dataclass(order=True)
class SiteIndex:
    site_mode: int = field(default=3)  # 上传地块的模式: 1 绘制地图； 2 上传地块 on Map； 3 上传地块
    site_area: float = field(default=0.0)
    site_classification: str = field(default="A")
    site_classification_info: str = field(default="Automatic land classification is not applicable, returns default value [A].")
    min_bulding_height: float = field(default=150.00)
    max_bulding_height: float = field(default=200.00)
    permitted_d_plot_ratio: float = field(default=0.00)  # 初始化时不进行计算
    permitted_nd_plot_ratio: float = field(default=0.00)  # 初始化时不进行计算
    permitted_total_plot_ratio: float = field(default=0.00)  # 初始化时不进行计算
    permitted_d_site_coverage: float = field(default=0.00)  # 初始化时不进行计算
    permitted_nd_site_coverage: float = field(default=0.00)  # 初始化时不进行计算

    @property
    def export_to_dict(self):
        return {
            "site_mode": self.site_mode,
            "site_area": self.site_area,
            "site_classification": self.site_classification,
            "site_classification_info": self.site_classification_info,
            "min_bulding_height": self.min_bulding_height,
            "max_bulding_height": self.max_bulding_height,
            "permitted_d_plot_ratio": self.permitted_d_plot_ratio,
            "permitted_nd_plot_ratio": self.permitted_nd_plot_ratio,
            "permitted_total_plot_ratio": self.permitted_total_plot_ratio,
            "permitted_d_site_coverage": self.permitted_d_site_coverage,
            "permitted_nd_site_coverage": self.permitted_nd_site_coverage,
        }


# 唯一个 创建场地模型的模块
@dataclass(order=True)
class SiteBuilder:
    site_name: str = field(default=str)
    site_id: str = field(default=str)
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
        """
        只是为了可视化，并不计算算法计算
        * 建一个场地三维模型，前端加载
        * 导出方案三维模型
        :param input_data_model:
        :return:
        """
        self.model = DataModel(name=self.site_name)
        self.model.id = self.site_id

        for element in input_data_model.elements:
            if element.layer == "Xkool_ProjectBoundary":
                gen_element = DataElement(geometry=element.geometry, layer=element.layer,
                                          material=DataElementMaterial(color="0xe5ebf1"),
                                          start_height=-(self.__base_height + self.__plot_height + self.__site_height),
                                          height=self.__base_height)
                self.model.insert_element(gen_element)
            elif element.layer == "Xkool_PlotBoundary":
                gen_element = DataElement(geometry=element.geometry, layer=element.layer,
                                          material=DataElementMaterial(color="0xced4d6"),
                                          start_height=-(self.__plot_height + self.__site_height),
                                          height=self.__plot_height)
                self.model.insert_element(gen_element)
            elif element.layer == "Xkool_SiteBoundary":
                self.site_index.site_area = round(element.geometry.area, 2)
                gen_element = DataElement(geometry=element.geometry, layer="Xkool_TargetSiteBoundary",
                                          material=DataElementMaterial(color="0xc7e29f"),
                                          start_height=-self.__site_height,
                                          height=self.__site_height)
                self.model.insert_element(gen_element)
            elif element.layer == "Xkool_SiteUnbuildableRegion":
                gen_element = DataElement(geometry=element.geometry, layer="Xkool_SiteUnbuildableRegion",
                                          material=DataElementMaterial(color="0x3F3F3F"),
                                          start_height=0, height=self.__inside_area_height)
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
            # TODO：道路/中心线

        self.model.renew()
        self.model.user_data = {"site_index": self.site_index.export_to_dict}


@dataclass(order=True)
class SiteBuilderBoundaryOnly:
    # 仅用于红线和不可建设区域的读取（模式2）
    site_name: str = field(default=str)
    site_id: str = field(default=str)
    # 所有指标
    site_index: SiteIndex = field(default_factory=SiteIndex)
    # 前端显示用的模型
    # model: DataModel = field(default_factory=DataModel)
    dict_to_front: Dict = field(default_factory=dict)

    __base_height: float = field(default=20)
    __plot_height: float = field(default=0.65)
    __site_height: float = field(default=0.35)
    __inside_area_height: float = field(default=0.15)
    __surrounding_mul_rise_building_height: float = field(default=24)
    __surrounding_high_rise_building_height: float = field(default=80)

    def build(self, input_data_model: DataModel):
        self.dict_to_front = {}
        self.model = DataModel(name=self.site_name)
        self.model.id = self.site_id

        # if "Xkool_SiteUnbuildableRegion" in input_data_model.layers:
        self.dict_to_front["Xkool_SiteUnbuildableRegion"] = []

        for element in input_data_model.elements:
            if element.layer == "Xkool_SiteBoundary":
                self.site_index.site_area = round(element.geometry.area, 2)
                self.dict_to_front["Xkool_SiteBoundary"] = wkt.dumps(element.geometry)

            elif element.layer == "Xkool_SiteUnbuildableRegion":
                self.dict_to_front["Xkool_SiteUnbuildableRegion"].append(wkt.dumps(element.geometry))

        # 找到bounding box的左下角点，移动所有图形的中点到原点
        x_min, y_min, x_max, y_max = wkt.loads(self.dict_to_front["Xkool_SiteBoundary"]).bounds

        move_x = -(x_min + x_max) / 2
        move_y = -(y_min + y_max) / 2

        self.dict_to_front["Xkool_SiteBoundary"] = self.translate(self.dict_to_front["Xkool_SiteBoundary"], move_x, move_y)
        for i in range(len(self.dict_to_front["Xkool_SiteUnbuildableRegion"])):
            self.dict_to_front["Xkool_SiteUnbuildableRegion"][i] = self.translate(self.dict_to_front["Xkool_SiteUnbuildableRegion"][i], move_x, move_y)

    def translate(self, poly, x, y):
        # 按照x,y移动多边形
        # print(poly)
        poly = wkt.loads(poly)
        poly = affinity.translate(poly, xoff=x, yoff=y)
        return wkt.dumps(poly)


