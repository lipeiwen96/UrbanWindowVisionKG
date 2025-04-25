"""
导出模型的总控制器
"""
import os
import shutil
import uuid
from dataclasses import field, dataclass
from typing import List
from art_public_modules.art_data_exchange.data_exchange import ARTShapelyDataExchanger
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from datetime import datetime
from art_public_modules.art_utils.common_utils.path_utils import PathUtils
from modules.input_module.site_builder import SiteBuilder
from modules.export_module.excel_exporter import ExcelExporter


@dataclass(order=True)
class SolutionExporter:
    # 输入信息
    input_project_name: str = field(default="")
    input_model: DataModel = field(default_factory=DataModel)

    file_name: str = field(default="")  # 每个文件的名字：Sol_PR{}_GFA{}_SC{}_TowerNum{}
    solution_name: str = field(default="")
    solution_dir: str = field(default="")  # 方案生成文件夹
    zip_dir: str = field(default="")  # 方案生成压缩包的路径
    sol_id: int = field(default=0)

    def setup(self, project_name: str, data_model: DataModel, clear_solution_dir: bool = True):
        self.input_project_name = project_name
        self.input_model = data_model
        print(self.input_model)
        sol_id = int(data_model.name.split("_")[-1])

        # 为这个方案建立一个独一无二的文件夹
        self.solution_name = f"Opt{sol_id}" + str(datetime.now().date()) + "-" + str(uuid.uuid4())
        self.solution_dir = os.path.abspath(os.path.join("files", "solutions", "generate_solution", self.solution_name))
        if os.path.exists(self.solution_dir):
            PathUtils.clean_all_files_in_dir(self.solution_dir)  # 存在文件夹，清空内部文件
        else:
            os.mkdir(self.solution_dir)  # 创建文件夹

        self.file_name = f"Opt{sol_id}-PR_{self.input_model.user_data['PR(TOTAL)']}-" \
                         f"GFA_{self.input_model.user_data['GFA(TOTAL)']}-" \
                         f"SC_{self.input_model.user_data['SC(D)']}-" \
                         f"TowerNum_{self.input_model.user_data['tower_num']}"

        # 先将场地模型和方案模型合并
        # 很重要 - 简化版的方案 + 场地信息
        self.__combine_site_and_solution_model()

        # 导出所有文件
        self.__export_mini_color_master()
        self.__export_dxf()
        self.__export_3dm()
        self.__export_excel()

        # 生成解压包
        self.zip_dir = os.path.abspath(os.path.join("files", "solutions", "solution_download", f"{self.solution_name}.zip"))  # 压缩包输出路径
        shutil.make_archive(self.zip_dir[:-4], 'zip', self.solution_dir)

        # 移除文件夹
        shutil.rmtree(self.solution_dir)

    def __combine_site_and_solution_model(self):
        # TODO MODE 0 / 1 / 2
        site_data_model = ARTShapelyDataExchanger.read_dxf_file(self.input_project_name)  # url - dxf / json
        # 创建项目场地
        site = SiteBuilder(site_id=str(uuid.uuid4()), site_name=self.input_project_name)
        site.build(site_data_model)

        for element in site.model.elements:
            self.input_model.insert_element(element)
        self.input_model.renew()
        self.input_model.name = self.file_name
        print(self.input_model.layers)

        # CAD中要的字
        self.input_model.user_data["building_text"] = []
        for element in self.input_model.elements:
            if element.layer == "Xkool_BuildingDOutline":
                name = f"{element.custom_semantics['id_text']}:{element.custom_semantics['building_total_floor_num']}F-{element.custom_semantics['building_total_height']}m"
                self.input_model.user_data["building_text"].append([name, [element.geometry.centroid.x, element.geometry.centroid.y],
                                                                    element.custom_semantics['angle']])
        self.input_model.delete_elements_by_layer_name("Xkool_BuildingTopWall")
        self.input_model.delete_elements_by_layer_name("Xkool_BuildingFloorLine")
        self.input_model.renew()

    def __export_mini_color_master(self):
        # 不做了
        pass

    def __export_dxf(self):
        ARTShapelyDataExchanger.write_dxf_file(self.solution_dir, self.input_model)

    def __export_3dm(self):
        # 修正模型的高度
        for element in self.input_model.elements:
            if element.layer == "Xkool_ProjectBoundary":
                element.start_height = -20
                element.height = 20
            elif element.layer == "Xkool_PlotBoundary":
                element.start_height = 0
                element.height = 0
            elif element.layer == "Xkool_TargetSiteBoundary":
                element.start_height = 0
                element.height = 0
            elif element.layer == "Xkool_SiteUnbuildableRegion":
                element.start_height = 0
                element.height = 0
            elif element.layer == "Xkool_BuildingMinSeperation":
                element.start_height = 2
                element.height = 0
            elif element.layer == "Xkool_BuildingDOutline":
                group_id = element.custom_semantics["group_id"] if "group_id" in element.custom_semantics.keys() else 0
                for i in range(element.custom_semantics["nd_storey"]):
                    dl = DataElement(geometry=element.geometry, layer="Xkool_BuildingFloorLine",
                                     start_height=element.custom_semantics["nd_floor_height"] * (i + 1),
                                     height=0,
                                     custom_semantics={"group_id": group_id},)
                    self.input_model.insert_element(dl)
                d_start_height = element.custom_semantics["nd_floor_height"] * element.custom_semantics["nd_storey"]
                for i in range(element.custom_semantics["storey"]):
                    dl = DataElement(geometry=element.geometry, layer="Xkool_BuildingFloorLine",
                                     start_height=element.custom_semantics["floor_height"] * (i + 1) + d_start_height,
                                     height=0,
                                     custom_semantics={"group_id": group_id},)
                    self.input_model.insert_element(dl)
        self.input_model.renew()
        rhino_path = os.path.join(self.solution_dir, self.file_name + ".3dm")
        ARTShapelyDataExchanger.write_rhino_file(rhino_path, self.input_model)

    def __export_excel(self):
        excel_exporter = ExcelExporter()

        excel_path = os.path.join(self.solution_dir, self.file_name + ".xlsx")
        excel_exporter.export(self.input_model, excel_path)









