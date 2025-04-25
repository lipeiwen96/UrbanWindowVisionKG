import json
from typing import List, Tuple, Dict, Union
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely import wkt

from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement


class JsonFileReader:
    @staticmethod
    def _read_json(file_path: str) -> Dict:
        """
        把json文件读取出来
        :param file_path:
        :return:
        """
        with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            data = json.load(f, strict=False)
        return data

    @staticmethod
    def _assemble_data_to_data_model(row_data:Dict)->Union[DataModel, None]:
        """
        把原始数据组装成data upload_model
        :param row_data:
        :return:
        """
        row_elements = row_data['model_elements']
        data_elements = []
        if len(row_elements)>0:
            for each in row_elements:
                cur_element = DataElement(geometry=Point(0,0), layer = each['element_type'], height=each['element_height'], start_height=each['element_start_height'])
                cur_element.is_hole = each['element_is_hole']
                if cur_element.is_hole == 'true':
                    cur_element.exterior_geometry = wkt.loads(each['element_exterior_geometry'])
                    cur_element.interior_geometry = [wkt.loads(_) for _ in each['element_interior_geometry']]
                    cur_element.geometry = cur_element.exterior_geometry.difference(MultiPolygon(cur_element.interior_geometry))
                elif cur_element.is_hole == 'false':
                    # cur_element.geometry = wkt.loads(each['element_exterior_geometry'])
                    cur_element.geometry = wkt.loads(each['element_geometry'])  # 沛文改 2022-0818
                else:
                    raise ('is_hole属性有误')

                cur_element.shadow = wkt.load(each['element_shadow']) if each['element_shadow']!="" else None
                cur_element.id = each['element_id']
                cur_element.geom_type = cur_element.geometry.geom_type

                cur_element.material.diffuse = each['material']['element_diffuse']
                cur_element.material.diffuse_opacity = each['material']['element_diffuse_opacity']
                cur_element.material.is_outline = each['material']['element_is_outline']
                cur_element.material.outline_color = each['material']['element_outline_color']
                cur_element.material.outline_opacity = each['material']['element_outline_opacity']
                cur_element.material.outline_width = each['material']['element_outline_width']
                cur_element.material.outline_type = each['material']['element_outline_type']
                cur_element.material.texture_pic = each['material']['element_texture_pic']
                cur_element.material.texture_scale = each['material']['element_texture_scale']
                cur_element.material.texture_rotate = each['material']['element_texture_rotate']
                cur_element.material.effect_type = each['material']['element_effect_type']

                cur_element.custom_semantics = each['element_custom_semantics']
                data_elements.append(cur_element)
            data_model = DataModel(elements=data_elements, name=row_data['model_name'])
            data_model.id = row_data['model_id']
            data_model.user_data = row_data["user_data"]

            return data_model
        else:
            return None

    def read_json_file(self, file_path: str) -> DataModel:
        """
        读取json化的数据，然后组装回去art 的data upload_model
        :param file_path:
        :return:
        """
        row_data = self._read_json(file_path=file_path)
        data_model = self._assemble_data_to_data_model(row_data=row_data)
        if data_model is not None:
            return data_model
        else:
            raise ('非法的数据，无法转换成data upload_model')


if __name__=="__main__":
    path="../../test_file/test_new2.json"

    data=JsonFileReader().read_json_file(file_path=path)
    print(data)