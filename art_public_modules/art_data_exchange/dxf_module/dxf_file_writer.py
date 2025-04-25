from typing import Dict, Tuple, List
import ezdxf
from shapely.geometry import MultiPolygon, MultiLineString, MultiPoint, Polygon, Point, LineString
from shapely import wkt
import art_public_modules.art_data_exchange.dxf_module.dxf_file_constant as constant
from art_public_modules.art_data_structure.shapely.core_structure import DataElement, DataModel


class DxfFileWriter:
    def __init__(self, file_path: str='.', file_name: str='default_dxf'):
        self.file_path = file_path
        self.file_name = file_name

    def _create_layer(self, data_model: DataModel, doc):
        """
        根据datamodel中的type创建图层
        :param data_model:
        :return:
        """
        all_elements = data_model.elements
        layers = set([_.layer for _ in all_elements])
        for name in layers:
            # 沛文修改 2020-8-18 根据字典修改保存图层的颜色
            if name in constant.DXF_LAYER_COLOR_DICT.keys():
                doc.layers.new(name=name, dxfattribs={'color': constant.DXF_LAYER_COLOR_DICT[name]})
            else:
                doc.layers.new(name)

    def _write_geometry_into_layer(self, msp, doc, data_model: DataModel):
        """
        将数据写入
        :param msp:
        :param data_model:
        :return:
        """
        all_elements = data_model.elements

        # 2023-7-16新增：创建group
        group_list = []
        if "building_id_info" in data_model.user_data.keys():
            group_num = len(data_model.user_data["building_id_info"])
            print(f"正在创建个CAD文件中{group_num}个GROUP")
            for i in range(group_num):
                group = doc.blocks.new(name=f"T{i + 1}")
                group_list.append(group)

        for element in all_elements:
            if isinstance(element.geometry, Point):
                group_id = self.check_in_group(element)
                if group_id == -1:
                    msp.add_point((element.geometry.x, element.geometry.y), dxfattribs={'layer': f'{element.layer}'})
                else:
                    group_list[group_id-1].add_point((element.geometry.x, element.geometry.y), dxfattribs={'layer': f'{element.layer}'})
            elif isinstance(element.geometry, LineString):
                group_id = self.check_in_group(element)
                if group_id == -1:
                    msp.add_polyline2d(list(element.geometry.coords), dxfattribs={'layer': f'{element.layer}'})
                else:
                    group_list[group_id-1].add_polyline2d(list(element.geometry.coords), dxfattribs={'layer': f'{element.layer}'})
            elif isinstance(element.geometry, Polygon):
                if element.is_hole == 'false':
                    group_id = self.check_in_group(element)
                    if group_id == -1:
                        msp.add_polyline2d(list(element.geometry.exterior.coords), dxfattribs={'layer': f'{element.layer}'})
                    else:
                        group_list[group_id - 1].add_polyline2d(list(element.geometry.exterior.coords), dxfattribs={'layer': f'{element.layer}'})
                else:
                    group_id = self.check_in_group(element)
                    if group_id == -1:
                        msp.add_polyline2d(list(element.exterior_geometry.exterior.coords), dxfattribs={'layer': f'{element.layer}'})
                        for _ in element.interior_geometry:
                            msp.add_polyline2d(list(_.exterior.coords), dxfattribs={'layer': f'{element.layer}'})
                    else:
                        group_list[group_id - 1].add_polyline2d(list(element.exterior_geometry.exterior.coords), dxfattribs={'layer': f'{element.layer}'})
                        for _ in element.interior_geometry:
                            group_list[group_id - 1].add_polyline2d(list(_.exterior.coords), dxfattribs={'layer': f'{element.layer}'})
            else:
                pass

        for i in range(len(group_list)):
            group_name = f"T{i+1}"
            msp.add_blockref(name=group_name, insert=(0,0))

    def check_in_group(self, element):
        if element.custom_semantics is not None:
            if "group_id" in element.custom_semantics.keys():
                group_id = element.custom_semantics["group_id"]
                return group_id
        return -1

    def _write_text_into_layer(self, msp, doc, data_model: DataModel):
        """
        写入文字标注到dxf文件
        """
        # 新建中文字体
        doc.styles.new("Chinese", dxfattribs={"font": "utils/simhei.ttf"})

        if "building_text" in data_model.user_data.keys():
            # 创建图层
            doc.layers.new(name="building_text", dxfattribs={'color': 9})
            # 创建文字
            for text in data_model.user_data["building_text"]:
                mtext = msp.add_mtext(text[0], dxfattribs={"style": "Chinese", 'layer': "annotation"}).set_location(
                    (text[1][0], text[1][1], 0),
                    rotation=text[2],
                    attachment_point=5)
                if "方案" in text[0]:
                    mtext.dxf.char_height = constant.TITLE_TEXT_HEIGHT
                else:
                    mtext.dxf.char_height = constant.NAME_TEXT_HEIGHT
        if "tree_centers" in data_model.user_data.keys():
            # 创建图层
            doc.layers.new(name="XKOOL_Gen_Tree", dxfattribs={'color': 95})
            for tree_center in data_model.user_data["tree_centers"]:
                msp.add_circle(center=tree_center, radius=4, dxfattribs={'layer': "XKOOL_Gen_Tree"})

    def write_dxf_file(self, data_model: DataModel):
        """
        根据输入的数据类型选择处理方式
        :param data_model:
        :return:
        """
        if not isinstance(data_model, DataModel):
            raise ValueError("輸入數據不是DataModel, 目前仅支持ART的DataModel")
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()
        # 创建图层
        self._create_layer(data_model=data_model, doc=doc)
        # 写入数据到dxf文件
        self._write_geometry_into_layer(msp=msp, doc=doc, data_model=data_model)
        # 写入文字标注到dxf文件
        self._write_text_into_layer(msp=msp, doc=doc, data_model=data_model)
        # 保存文件
        save_path = self.file_path+'/'+self.file_name+'.dxf'
        doc.saveas(save_path)
        print(f'成功写入{save_path}文件')


if __name__ == "__main__":
    polygon1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
    polygon2 = Polygon([(2, 2), (10, 2), (10, 10), (2, 10), (2, 2)])
    polygon3 = polygon1.difference(polygon2)

    ele1 = DataElement(geometry=polygon3, layer='1')
    ele2 = DataElement(geometry=polygon2, layer='2')
    ele3 = DataElement(geometry=polygon1, layer='3')
    data_model = DataModel(elements=[ele1, ele2, ele3])

    DxfFileWriter().write_dxf_file(data_model=data_model)
