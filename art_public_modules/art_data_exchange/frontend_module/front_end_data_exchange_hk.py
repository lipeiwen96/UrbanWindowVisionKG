from typing import Dict, Union
from shapely import wkt
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.base import BaseGeometry

from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from art_public_modules.art_utils.art_data_utils.art_shapely_data_ops import ARTShapelyDataOps


class HKFrontEndDataExchanger:
    def _convert_geometry_to_coord(self, object:BaseGeometry):
        """
        将shapely对象拆成坐标
        :param object:
        :return:
        """
        if isinstance(object, Point):
            return [object.x, object.y]
        elif isinstance(object, LineString):
            return [[each[0], each[1]] for each in list(object.coords)]
        elif isinstance(object, Polygon):
            return [[each[0], each[1]] for each in list(object.exterior.coords)]
        else:
            raise TypeError('只能转换shapely对象')

    def transform_to_front_end(self, data_model: DataModel, move_model_to_origin: bool = False) -> Dict:
        """
        把数据转换成字典格式发送给前端
        data model的中心点移动到0，0
        polygon，linestring以及point都转换成[x,y]格式
        :param data_model:
        :return:
        """
        data_model_center = data_model.envelope.centroid

        # 将模型的中心点移动至坐标原点 --> 非必要
        if move_model_to_origin:
            ARTShapelyDataOps.move_art_object(data_model, dx=data_model_center.x, dy=data_model_center.y)

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
            if isinstance(element.geometry, Polygon) and element.height == 0:
                cur_element['element_geom_type'] = "Linestring"
            else:
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

    def reassemble_data_model(self, data_model_dict:Dict)->DataModel:
        """
        将字典数据重新组装成data upload_model
        :param data_model_dict:
        :return:
        """
        all_data_elements = []
        for cur_element in data_model_dict['model_elements']:
            geometry = wkt.loads(cur_element['element_geometry'])
            new = DataElement(geometry=geometry, layer=cur_element['element_layer'],
                              height=cur_element['element_height'], start_height=cur_element['element_start_height'])
            new.geom_type = geometry.geom_type
            new.id = cur_element['element_id']
            new.material.color = cur_element['material']['element_color']
            new.material.opacity = cur_element['material']['element_opacity']
            new.material.outline_type = cur_element['material']['element_outline_type']
            new.custom_semantics=cur_element['element_custom_semantics']
            all_data_elements.append(new)
        data_model = DataModel(elements=all_data_elements, name=data_model_dict['model_name'], version=data_model_dict['model_version'],
                               id=data_model_dict['model_id'], layers=data_model_dict['model_layers'], user_data=data_model_dict['model_user_data'] )
        data_model.renew()
        return data_model