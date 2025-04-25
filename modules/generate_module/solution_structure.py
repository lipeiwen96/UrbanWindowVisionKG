"""
生成的方案结构
自动计算方案指标
自动加载间距框+三维模型
打包发给前端
"""
import uuid
from dataclasses import dataclass
from dataclasses import field

from algorithm_module.model.plan import Plan
from art_public_modules.art_data_structure.shapely.core_structure import DataElement
from art_public_modules.art_data_structure.shapely.core_structure import DataModel
from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial
from library.building_constant import BUILDING_TYPE_CONSTANT
from modules.regulation_module.building_structure import WindowDistanceBoxType
from modules.temp.closed_polygon_to_wall_segs import closed_polygon_to_2_walls
from modules.export_module.building_id import BuildingIdText


ACCURACY = 3  # 小数点位数


@dataclass(order=True)
class SolutionTemplate:
    # 　方案序号
    solution_id: int = field(default=0)
    # 前端显示用的模型
    model: DataModel = field(default_factory=DataModel)

    d_GFA: float = field(default=0)
    __nd_GFA: float = field(default=0)
    total_GFA: float = field(default=0)
    __d_PR: float = field(default=0)
    __nd_PR: float = field(default=0)
    total_PR: float = field(default=0)
    d_SC: float = field(default=0)
    __nd_SC: float = field(default=0)
    __FA: float = field(default=0)
    __max_building_height: float = field(default=0)
    tower_num: int = field(default=0)

    unit_num: int = field(default=0)
    studio_num: int = field(default=0)
    one_bed_num: int = field(default=0)
    two_bed_num: int = field(default=0)
    three_bed_num: int = field(default=0)

    input_dGFA: float = field(default=0)
    input_total_GFA: float = field(default=0)

    def create(self, plan: Plan, solution_id: int, solution_uuid: str = None):
        """
        建模！！！！
        :param plan:
        :param solution_id:
        :param solution_uuid:
        :return:
        """
        self.model = DataModel(name=f"Option_{solution_id}", id=solution_uuid or str(uuid.uuid4()))
        self.solution_id = solution_id
        self.__max_building_height = 0

        building_id = 1
        building_id_text_list = []
        for building in plan.buildings:
            total_unit_num = (BUILDING_TYPE_CONSTANT[building.building_name]["Studio_num"] +
                              BUILDING_TYPE_CONSTANT[building.building_name]["1B_num"] +
                              BUILDING_TYPE_CONSTANT[building.building_name]["2B_num"] +
                              BUILDING_TYPE_CONSTANT[building.building_name]["3B_num"]) * building.domestic_storey
            self.unit_num += total_unit_num
            self.studio_num += BUILDING_TYPE_CONSTANT[building.building_name]["Studio_num"] * building.domestic_storey
            self.one_bed_num += BUILDING_TYPE_CONSTANT[building.building_name]["1B_num"] * building.domestic_storey
            self.two_bed_num += BUILDING_TYPE_CONSTANT[building.building_name]["2B_num"] * building.domestic_storey
            self.three_bed_num += BUILDING_TYPE_CONSTANT[building.building_name]["3B_num"] * building.domestic_storey

            # 商业部分
            nd_element = DataElement(geometry=building.true_position_geometry,
                                     layer="Xkool_BuildingNDOutline",
                                     material=DataElementMaterial(color="0xffcc50"),
                                     start_height=0, height=building.non_domestic_height,
                                     custom_semantics={
                                         "group_id": building_id,
                                         "floor_height": building.non_domestic_floor_height,
                                         "storey": building.non_domestic_storey
                                     })
            self.model.insert_element(nd_element)
            # 住宅部分
            d_element = DataElement(geometry=building.true_position_geometry,
                                    layer="Xkool_BuildingDOutline",
                                    material=DataElementMaterial(color="0xb2b2b2"),
                                    start_height=building.non_domestic_height, height=building.domestic_floor_height * building.domestic_storey,
                                    custom_semantics={
                                        "group_id": building_id,
                                        "id_text": f"T{building_id}",
                                        "floor_height": building.domestic_floor_height,
                                        "storey": building.domestic_storey,
                                        "name": building.building_name,
                                        "building_total_floor_num": building.total_storey,
                                        "building_total_height": round(building.total_height, ACCURACY),
                                        "angle": building.angle,
                                        "nd_floor_height": building.non_domestic_floor_height,
                                        "nd_storey": building.non_domestic_storey
                                    })
            self.model.insert_element(d_element)
            # 生成女儿墙
            for wall in closed_polygon_to_2_walls(building.true_position_geometry, wall_thickness=0.2):
                d_element = DataElement(geometry=wall,
                                        layer="Xkool_BuildingTopWall",
                                        material=DataElementMaterial(color="0xb2b2b2"),
                                        start_height=building.total_height, height=1.3,
                                        custom_semantics={"group_id": building_id},
                                        )
                self.model.insert_element(d_element)

            # 楼层线
            # building_line = building.true_position_geometry.buffer(0.1, cap_style=2, join_style=2)
            # for i in range(building.non_domestic_storey):
            #     dl = DataElement(geometry=building_line, layer="Xkool_BuildingFloorLine",
            #                      start_height=building.non_domestic_floor_height * i,
            #                      material=DataElementMaterial(opacity=0.1, color="0x000000"),
            #                      height=0)
            #     self.model.insert_element(dl)
            # d_start_height = building.non_domestic_floor_height * building.non_domestic_storey
            # for i in range(building.domestic_storey):
            #     dl = DataElement(geometry=building_line, layer="Xkool_BuildingFloorLine",
            #                      start_height=building.domestic_floor_height * i + d_start_height,
            #                      material=DataElementMaterial(opacity=0.1, color="0x000000"),
            #                      height=0)
            #     self.model.insert_element(dl)

            # 楼间距框
            bd_element = DataElement(geometry=building.min_tower_distance_geometry,
                                     layer="Xkool_BuildingMinSeperation",
                                     material=DataElementMaterial(color="0xff0000", opacity=0.2),
                                     start_height=0,
                                     height=building.default_window_height,
                                     custom_semantics={"group_id": building_id},)
            self.model.insert_element(bd_element)
            # 三种窗间距框
            for window in building.window_list:
                if window.distance_box_type == WindowDistanceBoxType.inside:
                    bd_element = DataElement(geometry=window.inside_distance_box,
                                             layer="Xkool_HabitatRHP_Internal" if window.window_habitat_type else "Xkool_OtherRHP_Internal",
                                             material=DataElementMaterial(color="0x0000ff", opacity=0.6, outline_type="default"),  # 透明度0.2
                                             start_height=building.non_domestic_height + building.default_window_height,
                                             custom_semantics={"group_id": building_id},)
                    self.model.insert_element(bd_element)
                # elif window.distance_box_type == WindowDistanceBoxType.outside_fast_road:
                #     bd_element = DataElement(geometry=window.inside_distance_box,
                #                              layer="Xkool_HabitatRHP_ForReference" if window.window_habitat_type else "Xkool_OtherRHP_ForReference",
                #                              material=DataElementMaterial(color="0x0000ff", opacity=0.2, outline_type="dashed"),
                #                              start_height=building.non_domestic_height + building.default_window_height)
                #     self.model.insert_element(bd_element)
                elif window.distance_box_type == WindowDistanceBoxType.outside_normal_road:
                    # bd_element = DataElement(geometry=window.inside_distance_box,
                    #                          layer="Xkool_HabitatRHP_ForReference" if window.window_habitat_type else "Xkool_OtherRHP_ForReference",
                    #                          material=DataElementMaterial(color="0x0000ff", opacity=0.2, outline_type="dashed"),
                    #                          start_height=building.non_domestic_height + building.default_window_height)
                    # self.model.insert_element(bd_element)
                    bd_element = DataElement(geometry=window.outside_normal_road_distance_box,
                                             layer="Xkool_HabitatRHP_SmallerThan4.5mStreet" if window.window_habitat_type else "Xkool_OtherRHP_SmallerThan4.5mStreet",
                                             material=DataElementMaterial(color="0xff6c00", opacity=0.6, outline_type="default"),
                                             start_height=building.non_domestic_height + building.default_window_height,
                                             custom_semantics={"group_id": building_id},)
                    self.model.insert_element(bd_element)

            # 添加建筑编号信息
            building_id_text = BuildingIdText()
            building_id_text.init(id=building_id, point=building.true_position_geometry.centroid, geometry=building.true_position_geometry,
                                  d_floor_num=building.domestic_storey, d_floor_height=building.domestic_floor_height,
                                  nd_floor_num=building.non_domestic_storey, nd_floor_height=building.non_domestic_floor_height)
            building_id_text_list.append(building_id_text.to_dict)
            building_id += 1

            # 更新方案参数
            # 底面积
            self.__FA += building.first_floor_area
            # 商业总面积
            self.d_GFA += building.first_floor_area * building.domestic_storey
            self.__nd_GFA += building.first_floor_area * building.non_domestic_storey
            self.total_GFA += building.first_floor_area * (building.non_domestic_storey + building.domestic_storey)
            # 最高楼高
            self.__max_building_height = max(building.non_domestic_height + building.domestic_floor_height * building.domestic_storey,
                                             self.__max_building_height)
            # 建筑楼数
            self.tower_num += 1

        self.__d_PR = round(self.d_GFA / plan.site.site_area, ACCURACY)
        self.__nd_PR = round(self.__nd_GFA / plan.site.site_area, ACCURACY)
        self.total_PR = round(self.total_GFA / plan.site.site_area, ACCURACY)
        self.d_SC = round((self.__FA / plan.site.site_area) * 100, ACCURACY)
        self.__nd_SC = self.d_SC

        floor_num_str = ""
        for building in plan.buildings:
            floor_num_str += str(building.total_storey) + "+"
        print(f"【生成方案{solution_id}】: 共【{self.tower_num}栋楼】({floor_num_str}), 方案指标："
              f"GFA-【{round(self.total_GFA, ACCURACY)}】{round(self.d_GFA, ACCURACY)}(D)+{round(self.__nd_GFA, ACCURACY)}(ND),"
              f"PR-【{round(self.total_PR, ACCURACY)}】{round(self.__d_PR, ACCURACY)}(D)+{round(self.__nd_PR, ACCURACY)}(ND)  SC-【{round(self.d_SC, ACCURACY)}】")

        # 经济指标
        self.model.user_data = {
            "solution_index": self.solution_id,
            'site_area': round(plan.site.site_area),
            'GFA(ND)': round(self.__nd_GFA),
            'GFA(D)': round(self.d_GFA),
            'GFA(TOTAL)': round(self.total_GFA),
            'PR(D)': round(self.__d_PR, ACCURACY),
            'PR(ND)': round(self.__nd_PR, ACCURACY),
            'PR(TOTAL)': round(self.total_PR, ACCURACY),
            'SC(D)': round(self.d_SC, ACCURACY),
            'SC(ND)': round(self.__nd_SC, ACCURACY),
            'tower_num': self.tower_num,
            "unit_num": self.unit_num,
            "Studio_num": self.studio_num,
            "1B_num": self.one_bed_num,
            "2B_num": self.two_bed_num,
            "3B_num": self.three_bed_num,
            'max_bulding_height': round(self.__max_building_height, ACCURACY),
            'drawing': {
                "site_outline": plan.site.site_geometry.wkt,
                "site_unplaced_area_list": [area.wkt for area in plan.site.inside_unplaced_area_list],
                "building_list": [building.true_position_geometry.wkt for building in plan.buildings],
            },
            # 用于导出方案表格计算
            "input_index": {
                'site_area': plan.site.site_area,
                'Target_PR(TOTAL)': plan.site.site_plot_ratio,
                'Target_PR(D)': plan.site.input_dpr,
                'Target_PR(ND)': plan.site.input_ndpr,
                'Target_GFA(TOTAL)': round(plan.site.site_area * plan.site.site_plot_ratio, ACCURACY),
                'Target_GFA(D)': round(plan.site.site_area * plan.site.input_dpr, ACCURACY),
                'Target_GFA(ND)': round(plan.site.site_area * plan.site.input_ndpr, ACCURACY),
                'Target_SC(D)': round(plan.site.input_dsc, ACCURACY),
                'Target_SC(ND)': round(plan.site.input_ndsc, ACCURACY),
            },
            "input_building": {
                'building_type': plan.buildings[0].building_name,
                'floor_area': plan.buildings[0].first_floor_area,
                'floor_height(D)': plan.buildings[0].domestic_floor_height,
                'floor_height(ND)': plan.buildings[0].non_domestic_floor_height,
                'Nos_of_storey(D)': f"{plan.buildings[0].min_d_floor_number}~{plan.buildings[0].max_d_floor_number}",
                'Nos_of_storey(ND)': plan.buildings[0].non_domestic_storey,
                'min_building_height': plan.buildings[0].min_building_height,
                'max_building_height': plan.buildings[0].max_building_height,
            },
            "building_id_info": building_id_text_list
        }
        self.input_dGFA = round(plan.site.site_area * plan.site.input_dpr, ACCURACY)
        self.input_total_GFA = round(plan.site.site_area * plan.site.site_plot_ratio, ACCURACY)
