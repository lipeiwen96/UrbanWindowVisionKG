from dataclasses import field, dataclass
from typing import Dict, Union, List
import os
from utils.excel_utils import ExcelMethods
from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
import xlsxwriter
from xlsxwriter.format import Format
from library.building_constant import BUILDING_TYPE_CONSTANT


@dataclass(order=True, unsafe_hash=True)
class ExcelExporter:
    data_model: DataModel = field(default_factory=DataModel)
    xlsx_path: str = field(default=str)

    def export(self, data_model: DataModel, firl_dir: str):
        self.data_model = data_model
        self.xlsx_path = firl_dir

        # 计算项目所有楼型
        self.building_type_dict = {}
        for ele in self.data_model.elements:
            if ele.layer == "Xkool_BuildingDOutline":
                if ele.custom_semantics["building_total_floor_num"] in self.building_type_dict.keys():
                    self.building_type_dict[ele.custom_semantics["building_total_floor_num"]]["num"] += 1
                else:
                    self.building_type_dict[ele.custom_semantics["building_total_floor_num"]] = {
                        "num": 1,
                        "total_height": ele.custom_semantics["building_total_height"],
                        "details": f": {ele.custom_semantics['nd_storey']}F(ND)+{ele.custom_semantics['storey']}F(D) "
                                   f"({BUILDING_TYPE_CONSTANT[ele.custom_semantics['name']]['flat_mix']})"
                    }

        self.__export_xlsx()

    def __export_xlsx(self):
        """
        绘制主要技术经济指标表
        """
        # 创建excel
        workbook = xlsxwriter.Workbook(self.xlsx_path)
        worksheet = workbook.add_worksheet('Option Index')

        # 定义表格单元格的全部样式
        self.__define_excel_cell_style(workbook)

        # STEP1 - 设置列宽
        worksheet.set_column('B:B', 10)
        worksheet.set_column('C:C', 30)
        worksheet.set_column('D:D', 12)
        worksheet.set_column('E:E', 16)
        # 设置行高
        worksheet.set_row(1, 20)  # 表名行

        # STEP2 - 表格的表头及列标题
        worksheet.merge_range('B2:E2', f'Option Main Technical and Economic Index Table', self.header_format)
        worksheet.merge_range('B3:E3', 'Input Information', self.input_title_format)
        worksheet.merge_range('B4:E4', '1.Input Index', self.default_format)
        # sitearea
        worksheet.write('B5', '1.1', self.default_format)
        worksheet.write('C5', 'Site Area', self.default_format)
        worksheet.write('D5', '㎡', self.default_format)
        worksheet.write('E5', round(self.data_model.user_data["input_index"]["site_area"]), self.default_format)
        # Target_PR(TOTAL)
        worksheet.write('B6', '1.2', self.default_format)
        worksheet.write('C6', 'Target PR(TOTAL)', self.default_format)
        worksheet.write('D6', '/', self.default_format)
        worksheet.write('E6', round(self.data_model.user_data["input_index"]["Target_PR(TOTAL)"], 2), self.default_format)
        # Target_PR(D)
        worksheet.write('B7', '1.3', self.default_format)
        worksheet.write('C7', 'Target PR(D)', self.default_format)
        worksheet.write('D7', '/', self.default_format)
        worksheet.write('E7', round(self.data_model.user_data["input_index"]["Target_PR(D)"], 2), self.default_format)
        # Target_PR(ND)
        worksheet.write('B8', '1.4', self.default_format)
        worksheet.write('C8', 'Target PR(ND)', self.default_format)
        worksheet.write('D8', '/', self.default_format)
        worksheet.write('E8', round(self.data_model.user_data["input_index"]["Target_PR(ND)"], 2), self.default_format)
        # Target_GFA(TOTAL)
        worksheet.write('B9', '1.5', self.default_format)
        worksheet.write('C9', 'Target GFA(TOTAL)', self.default_format)
        worksheet.write('D9', '㎡', self.default_format)
        worksheet.write('E9', round(self.data_model.user_data["input_index"]["Target_GFA(TOTAL)"]), self.default_format)
        # Target_GFA(D)
        worksheet.write('B10', '1.6', self.default_format)
        worksheet.write('C10', 'Target GFA(D)', self.default_format)
        worksheet.write('D10', '㎡', self.default_format)
        worksheet.write('E10', round(self.data_model.user_data["input_index"]["Target_GFA(D)"]), self.default_format)
        # Target_GFA(ND)
        worksheet.write('B11', '1.7', self.default_format)
        worksheet.write('C11', 'Target GFA(ND)', self.default_format)
        worksheet.write('D11', '㎡', self.default_format)
        worksheet.write('E11', round(self.data_model.user_data["input_index"]["Target_GFA(ND)"]), self.default_format)
        # Target_SC(D)
        worksheet.write('B12', '1.8', self.default_format)
        worksheet.write('C12', 'Target SC(D)', self.default_format)
        worksheet.write('D12', '%', self.default_format)
        worksheet.write('E12', round(self.data_model.user_data["input_index"]["Target_SC(D)"], 2), self.default_format)
        # Target_SC(ND)
        worksheet.write('B13', '1.9', self.default_format)
        worksheet.write('C13', 'Target SC(ND)', self.default_format)
        worksheet.write('D13', '%', self.default_format)
        worksheet.write('E13', round(self.data_model.user_data["input_index"]["Target_SC(ND)"], 2), self.default_format)

        # STEP2 - BuildingPlan
        worksheet.merge_range('B14:E14', '2.Input Building', self.default_format)
        # building type
        worksheet.write('B15', '2.1', self.default_format)
        worksheet.write('C15', 'Typical Floorplan Type Name', self.default_format)
        worksheet.write('D15', '/', self.default_format)
        worksheet.write('E15', self.data_model.user_data["input_building"]["building_type"], self.default_format)
        # Floor Area
        worksheet.write('B16', '2.2', self.default_format)
        worksheet.write('C16', 'Typical Floorplan Area', self.default_format)
        worksheet.write('D16', '㎡', self.default_format)
        worksheet.write('E16', round(self.data_model.user_data["input_building"]["floor_area"], 2), self.default_format)
        # Floor Height(D)
        worksheet.write('B17', '2.3', self.default_format)
        worksheet.write('C17', 'Floor Height(D)', self.default_format)
        worksheet.write('D17', 'm', self.default_format)
        worksheet.write('E17', round(self.data_model.user_data["input_building"]["floor_height(D)"], 2), self.default_format)
        # Floor Height(ND)
        worksheet.write('B18', '2.4', self.default_format)
        worksheet.write('C18', 'Floor Height(ND)', self.default_format)
        worksheet.write('D18', 'm', self.default_format)
        worksheet.write('E18', round(self.data_model.user_data["input_building"]["floor_height(ND)"], 2), self.default_format)
        # Nos_of_storey(D)
        # worksheet.write('B19', '2.5', self.default_format)
        # worksheet.write('C19', 'Nos. of storey(D)', self.default_format)
        # worksheet.write('D19', '/', self.default_format)
        # worksheet.write('E19', self.data_model.user_data["input_building"]["Nos_of_storey(D)"], self.default_format)
        # Nos_of_storey(ND)
        worksheet.write('B19', '2.5', self.default_format)
        worksheet.write('C19', 'Nos. of storey(ND)', self.default_format)
        worksheet.write('D19', '/', self.default_format)
        worksheet.write('E19', self.data_model.user_data["input_building"]["Nos_of_storey(ND)"], self.default_format)
        # MIN Height(D)
        worksheet.write('B20', '2.6', self.default_format)
        worksheet.write('C20', 'Min Building Height(D)', self.default_format)
        worksheet.write('D20', 'm', self.default_format)
        worksheet.write('E20', round(self.data_model.user_data["input_building"]["min_building_height"], 2), self.default_format)
        # MAX Height(ND)
        worksheet.write('B21', '2.7', self.default_format)
        worksheet.write('C21', 'Max Building Height(ND)', self.default_format)
        worksheet.write('D21', 'm', self.default_format)
        worksheet.write('E21', round(self.data_model.user_data["input_building"]["max_building_height"], 2), self.default_format)

        # Output
        worksheet.merge_range('B22:E22', 'Output Option Information', self.output_title_format)
        worksheet.merge_range('B23:E23', '3.Option Building Information', self.default_format)

        worksheet.write('B24', '', self.default_format)
        worksheet.write('C24', 'Nos. of storey(Total)', self.default_format)
        worksheet.write('D24', 'Height(m)', self.default_format)
        worksheet.write('E24', 'Tower Num', self.default_format)
        id = 0
        for key in self.building_type_dict.keys():
            worksheet.write(f'B{25+id}', f'{id+1}', self.default_format)
            worksheet.write(f'C{25+id}', f"{key}F"+self.building_type_dict[key]["details"], self.default_format)
            worksheet.write(f'D{25+id}', round(self.building_type_dict[key]["total_height"], 2), self.default_format)
            worksheet.write(f'E{25+id}', self.building_type_dict[key]["num"], self.default_format)
            id += 1

        worksheet.merge_range(f'B{25+id}:E{25+id}', '4.Option Index', self.default_format)
        # PR(TOTAL)
        worksheet.write(f'B{25+id+1}', '4.1', self.default_format)
        worksheet.write(f'C{25+id+1}', 'PR(TOTAL)', self.default_format)
        worksheet.write(f'D{25+id+1}', '/', self.default_format)
        worksheet.write(f'E{25+id+1}', round(self.data_model.user_data["PR(TOTAL)"], 2), self.default_format)
        # PR(D)
        worksheet.write(f'B{25+id+2}', '4.2', self.default_format)
        worksheet.write(f'C{25+id+2}', 'PR(D)', self.default_format)
        worksheet.write(f'D{25+id+2}', '/', self.default_format)
        worksheet.write(f'E{25+id+2}', round(self.data_model.user_data["PR(D)"], 2), self.default_format)
        # PR(ND)
        worksheet.write(f'B{25+id+3}', '4.3', self.default_format)
        worksheet.write(f'C{25+id+3}', 'PR(ND)', self.default_format)
        worksheet.write(f'D{25+id+3}', '/', self.default_format)
        worksheet.write(f'E{25+id+3}', round(self.data_model.user_data["PR(ND)"], 2), self.default_format)
        # GFA(TOTAL)
        worksheet.write(f'B{25+id+4}', '4.4', self.default_format)
        worksheet.write(f'C{25+id+4}', 'GFA(TOTAL)', self.default_format)
        worksheet.write(f'D{25+id+4}', '㎡', self.default_format)
        worksheet.write(f'E{25+id+4}', round(self.data_model.user_data["GFA(TOTAL)"]), self.default_format)
        # GFA(D)
        worksheet.write(f'B{25 + id + 5}', '4.5', self.default_format)
        worksheet.write(f'C{25 + id + 5}', 'GFA(D)', self.default_format)
        worksheet.write(f'D{25 + id + 5}', '㎡', self.default_format)
        worksheet.write(f'E{25 + id + 5}', round(self.data_model.user_data["GFA(D)"]), self.default_format)
        # GFA(ND)
        worksheet.write(f'B{25 + id + 6}', '4.6', self.default_format)
        worksheet.write(f'C{25 + id + 6}', 'GFA(ND)', self.default_format)
        worksheet.write(f'D{25 + id + 6}', '㎡', self.default_format)
        worksheet.write(f'E{25 + id + 6}', round(self.data_model.user_data["GFA(ND)"]), self.default_format)
        # SC(D)
        worksheet.write(f'B{25 + id + 7}', '4.7', self.default_format)
        worksheet.write(f'C{25 + id + 7}', 'SC(D)', self.default_format)
        worksheet.write(f'D{25 + id + 7}', '%', self.default_format)
        worksheet.write(f'E{25 + id + 7}', round(self.data_model.user_data["SC(D)"], 2), self.default_format)
        # SC(ND)
        worksheet.write(f'B{25 + id + 8}', '4.8', self.default_format)
        worksheet.write(f'C{25 + id + 8}', 'SC(ND)', self.default_format)
        worksheet.write(f'D{25 + id + 8}', '%', self.default_format)
        worksheet.write(f'E{25 + id + 8}', round(self.data_model.user_data["SC(ND)"], 2), self.default_format)

        # Nos. of Units
        worksheet.write(f'B{25 + id + 9}', '4.9', self.default_format)
        worksheet.write(f'C{25 + id + 9}', 'Nos. of Units', self.default_format)
        worksheet.write(f'D{25 + id + 9}', '/', self.default_format)
        worksheet.write(f'E{25 + id + 9}', self.data_model.user_data["unit_num"], self.default_format)
        # Nos. of Studio
        worksheet.write(f'B{25 + id + 10}', '4.10', self.default_format)
        worksheet.write(f'C{25 + id + 10}', 'Nos. of Studio', self.default_format)
        worksheet.write(f'D{25 + id + 10}', '/', self.default_format)
        worksheet.write(f'E{25 + id + 10}', self.data_model.user_data["Studio_num"], self.default_format)
        # Nos. of Units
        worksheet.write(f'B{25 + id + 11}', '4.11', self.default_format)
        worksheet.write(f'C{25 + id + 11}', 'Nos. of 1-Bedroom', self.default_format)
        worksheet.write(f'D{25 + id + 11}', '/', self.default_format)
        worksheet.write(f'E{25 + id + 11}', self.data_model.user_data["1B_num"], self.default_format)
        # Nos. of Units
        worksheet.write(f'B{25 + id + 12}', '4.12', self.default_format)
        worksheet.write(f'C{25 + id + 12}', 'Nos. of 2-Bedroom', self.default_format)
        worksheet.write(f'D{25 + id + 12}', '/', self.default_format)
        worksheet.write(f'E{25 + id + 12}', self.data_model.user_data["2B_num"], self.default_format)
        # Nos. of Units
        worksheet.write(f'B{25 + id + 13}', '4.13', self.default_format)
        worksheet.write(f'C{25 + id + 13}', 'Nos. of 3-Bedroom', self.default_format)
        worksheet.write(f'D{25 + id + 13}', '/', self.default_format)
        worksheet.write(f'E{25 + id + 13}', self.data_model.user_data["3B_num"], self.default_format)

        # 关闭excel
        workbook.close()
        print(f'成功将项目方案导出为Excel指标表，存储路径：{self.xlsx_path}')
        print("---------------------------------------------------------------")

    def __define_excel_cell_style(self, workbook):
        """
        定义表格单元格的全部样式
        """
        # 每个表格的表头样式——深紫色白字居中
        self.header_format = workbook.add_format({
            'bold': True,
            'color': 'white',  # 字体为白色
            'align': 'left',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#9575cd',  # deep purple04 最深
            'border': 1, })
        # 每个表格的列标题——紫色黑字居中
        self.input_title_format = workbook.add_format({
            'bold': True,
            'font_size': 12,  # 字体大小为9
            'align': 'left',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#e0a237',  # deep purple03 次深
            'border': 1, })
        self.output_title_format = workbook.add_format({
            'bold': True,
            # 'font_size': 10,  # 字体大小为9
            'align': 'left',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#5ecde5',  # deep purple03 次深
            'border': 1, })
        self.default_format = workbook.add_format({
            'font_size': 10,  # 字体大小为9
            'align': 'left',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'border': 1, })
        # 行标题——紫色加粗居中
        self.row_title_bord_format = workbook.add_format({
            'bold': True,
            'font_size': 9,  # 字体大小为9
            'align': 'left',  # 左对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#d1c4e9',  # deep purple02 次浅
            'border': 1, })
        self.row_title_bord_center_format = workbook.add_format({
            'bold': True,
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#d1c4e9',  # deep purple02 次浅
            'border': 1, })
        # 行标题——紫色居中
        self.row_title_format = workbook.add_format({
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#d1c4e9',  # deep purple02 次浅
            'border': 1})

        # 产品名称
        self.plant_product_name_format = workbook.add_format({
            'font_size': 10,  # 字体大小为11
            'align': 'left',  # 左对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#FDE9D9',  # 淡橙色
            'border': 1, })
        self.facility_product_name_format = workbook.add_format({
            'font_size': 10,  # 字体大小为11
            'align': 'left',  # 左对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#EBF1DE',  # 淡绿色
            'border': 1, })
        # 标准名称单元格：淡紫色荻黑字居中
        self.standard_name_format = workbook.add_format({
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#EDE7F6',  # 淡紫色
            'border': 1, })
        # 标准名称加粗单元格：淡紫色荻黑字加粗居中
        self.standard_name_bold_format = workbook.add_format({
            'bold': True,
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#EDE7F6',  # 淡紫色
            'border': 1, })
        # 固定参数单元格
        self.fixed_parameter_format = workbook.add_format({
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#D9D9D9',  # 浅灰色
            'border': 1, })
        self.fixed_parameter_percentage_format = workbook.add_format({
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#D9D9D9',  # 浅灰色
            'border': 1, })
        self.fixed_parameter_percentage_format.set_num_format(10)  # 百分数格式
        # 可变参数单元格-蓝底蓝字居中
        self.variable_parameter_format = workbook.add_format({
            'color': '#1976D2',  # 蓝色
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#DAEEF3',  # 淡蓝色
            'border': 1, })
        # 联动参数单元格-红底红字居中
        self.linkage_parameter_format = workbook.add_format({
            'color': 'red',
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#F7D6D5',  # 淡红色
            'border': 1, })
        self.linkage_parameter_percentage_format = workbook.add_format({
            'color': 'red',
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#F7D6D5',  # 淡红色
            'border': 1, })
        self.linkage_parameter_percentage_format.set_num_format(10)  # 百分数格式
        # 加强版的联动参数单元格-黄底红字居中
        self.linkage_parameter_strengthen_format = workbook.add_format({
            'color': 'red',
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#FFFF00',  # 黄色
            'border': 1, })
        self.linkage_parameter_percentage_strengthen_format = workbook.add_format({
            'color': 'red',
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#FFFF00',  # 黄色
            'border': 1, })
        self.linkage_parameter_percentage_strengthen_format.set_num_format(10)  # 百分数格式
        # 紫色版的联动参数单元格-紫底黑字居中
        self.linkage_parameter_purple_format = workbook.add_format({
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#d1c4e9',  # deep purple02 次浅
            'border': 1, })
        self.linkage_parameter_percentage_purple_format = workbook.add_format({
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#d1c4e9',  # deep purple02 次浅
            'border': 1, })
        self.linkage_parameter_percentage_purple_format.set_num_format(10)  # 百分数格式
        self.linkage_parameter_purple_bold_format = workbook.add_format({
            'bold': True,
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#d1c4e9',  # deep purple02 次浅
            'border': 1, })
        self.linkage_parameter_percentage_purple_bold_format = workbook.add_format({
            'bold': True,
            'font_size': 9,  # 字体大小为9
            'align': 'center',  # 居中对齐
            'valign': 'vcenter',
            'text_wrap': True,  # 自适应单元格宽度
            'bg_color': '#d1c4e9',  # deep purple02 次浅
            'border': 1, })
        self.linkage_parameter_percentage_purple_bold_format.set_num_format(10)  # 百分数格式