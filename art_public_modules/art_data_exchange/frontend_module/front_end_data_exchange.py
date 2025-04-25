from typing import Dict, Union

from shapely.geometry import Point, LineString, Polygon
from shapely.geometry.base import BaseGeometry

from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from art_public_modules.art_utils.art_data_utils.art_shapely_data_ops import ARTShapelyDataOps


class FrontEndDataExchanger:
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
            cur_element['element_geometry'] = self._convert_geometry_to_coord(object=element.geometry)
            cur_element['element_exterior_geometry'] = self._convert_geometry_to_coord(object=element.exterior_geometry)
            cur_element['element_interior_geometry'] = [self._convert_geometry_to_coord(object=_) for _ in element.interior_geometry]
            cur_element['element_is_hole'] = element.is_hole
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
            # material['element_diffuse_opacity'] = element.material.diffuse_opacity
            # material['element_is_outline'] = element.material.is_outline
            # material['element_outline_color'] = element.material.outline_color
            # material['element_outline_opacity'] = element.material.outline_opacity
            # material['element_outline_width'] = element.material.outline_width
            # material['element_outline_type'] = element.material.outline_type
            # material['element_texture_pic'] = element.material.texture_pic
            # material['element_texture_scale'] = element.material.texture_scale
            # material['element_texture_rotate'] = element.material.texture_rotate
            # material['element_effect_type'] = element.material.effect_type
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
            geometry = cur_element['element_exterior_geometry']
            if cur_element['element_geom_type'] == 'Polygon':
                geometry = Polygon(geometry)
            elif cur_element['element_geom_type'] == 'LineString':
                geometry = LineString(geometry)
            elif cur_element['element_geom_type'] == 'Point':
                geometry = Point(geometry)
            new = DataElement(geometry=geometry, type=cur_element['element_type'], height=cur_element['element_height'], start_height=cur_element['element_start_height'])
            new.is_hole = cur_element['element_is_hole']
            new.id = cur_element['element_id']
            new.geom_type = cur_element['element_geom_type']
            new.diffuse = cur_element['element_diffuse']
            new.diffuse_opacity = cur_element['element_diffuse_opacity']
            new.is_outline =cur_element['element_is_outline']
            new.outline_color = cur_element['element_outline_color']
            new.outline_opacity = cur_element['element_outline_opacity']
            new.outline_width = cur_element['element_outline_width']
            new.outline_type = cur_element['element_outline_type']
            new.texture_pic=cur_element['element_texture_pic']
            new.texture_scale=cur_element['element_texture_scale']
            new.texture_rotate=cur_element['element_texture_rotate']
            new.effect_type=cur_element['element_effect_type']
            new.custom_semantics=cur_element['element_custom_semantics']
            all_data_elements.append(new)
        data_model = DataModel(data_elements=all_data_elements, name=data_model_dict['model_name'])
        return data_model