import json
from typing import List, Tuple, Dict, Union
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely import wkt

from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement


class HKJsonFileReader:
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
                cur_element = DataElement(geometry=wkt.loads(each['element_geometry']), layer=each['element_layer'], height=each['element_height'], start_height=each['element_start_height'])
                cur_element.id = each['element_id']
                cur_element.geom_type = cur_element.geometry.geom_type
                # material
                cur_element.material.opacity = each['material']['element_opacity']
                cur_element.material.color = each['material']['element_color']
                cur_element.material.outline_type = each['material']['element_outline_type']
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