import json
import os
from typing import List, Tuple, Dict
from shapely.geometry import Polygon, LineString, Point
from shapely import wkt

from art_public_modules.art_data_exchange.json_module.json_model_reader import JsonFileReader
from art_public_modules.art_data_structure.shapely.core_structure import DataModel


class HKJsonFileWriter:
    def __init__(self, file_path: str = None):
        self.file_path = file_path

    def _transform_data_model(self, data_model: DataModel)->Dict:
        """
        将data model转换成dict数据结构
        :param data_model:
        :return:
        """
        data_model_dict = {}
        data_model_dict['model_id'] = data_model.id
        data_model_dict['model_name'] = data_model.name
        data_model_dict['model_elements'] = []
        data_model_dict['model_layers'] = data_model.layers
        data_model_dict['model_user_data'] = data_model.user_data
        data_model_dict['model_version'] = data_model.version
        for element in data_model.elements:
            cur_element = {}
            cur_element['element_geometry'] = element.geometry.wkt
            cur_element['element_id'] = element.id
            cur_element['element_layer'] = element.layer
            # if isinstance(element.geometry, Polygon) and element.height == 0:
            #     cur_element['element_geom_type'] = "Linestring"
            # else:
            #     cur_element['element_geom_type'] = element.geom_type
            cur_element['element_geom_type'] = element.geom_type
            if element.height < 0:
                cur_element['element_height'] = - element.height
                cur_element['element_start_height'] = element.start_height + element.height
            else:
                cur_element['element_height'] = element.height
                cur_element['element_start_height'] = element.start_height
            material={}
            material['element_color'] = element.material.color
            material['element_opacity'] = element.material.opacity
            material['element_outline_type'] = element.material.outline_type
            cur_element['material'] = material
            cur_element['element_custom_semantics'] = element.custom_semantics
            data_model_dict['model_elements'].append(cur_element)
        return data_model_dict

    def _write_to_json(self, data:Dict):
        """
        写入json文件
        :param data:
        :return:
        """
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(self.file_path, 'w', encoding="utf-8") as f:
            f.write(json_str)

    def write_json_file(self, data_model: DataModel, file_path: str = None):
        """
        将data model写出成json文件，通常用过程存储
        :param file_path:
        :param data_model:
        :return:
        """
        if isinstance(data_model, DataModel):
            if file_path:
                self.file_path = file_path
            else:
                # 获得该模型的名字
                model_name = data_model.name
                self.file_path = self.file_path + model_name + '.json'
            data_model_dict = self._transform_data_model(data_model=data_model)
            self._write_to_json(data=data_model_dict)
            return data_model_dict
        else:
            raise ValueError('只能使用data upload_model')