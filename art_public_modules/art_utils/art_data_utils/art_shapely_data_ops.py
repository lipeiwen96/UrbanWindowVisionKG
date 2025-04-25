"""
针对art data structure 中的data element 以及data model的操作
"""
from typing import *
from matplotlib.patches import Polygon
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform, rotate
from copy import deepcopy

from art_public_modules.art_data_structure.shapely.core_structure import DataElement, DataModel


class ARTShapelyDataOps:
    @staticmethod
    def move_art_object(object: Union['DataElement', 'DataModel'], dx: float = 0, dy: float = 0):
        """
        给定一个ART物件，以及一个向量
        以该物件的envelope的中心点为基准移动到目标处
        :param object:
        :param dx: x轴移动距离
        :param dy: y轴移动距离
        :return:
        """
        if isinstance(object, (DataModel, DataElement)):
            if isinstance(object, DataElement):
                object.geometry = affine_transform(object.geometry, [1,0,0,1,-dx, -dy])
            else:
                for element in object.elements:
                    element.geometry =affine_transform(element.geometry, [1,0,0,1,-dx,-dy])
        else:
            raise TypeError('输入了错误的参数类型，请检查参数')

    @staticmethod
    def rotate_art_object(object: Union['DataElement', 'DataModel'], origin: Point = None, angle: float = 0):
        """
        给定一个ART物件以及一个角度
        对整个物件，以给定的点进行旋转
        :param object:
        :param angle:
        :return:
        """
        if isinstance(object, (DataModel, DataElement)) and (origin is None or isinstance(origin, Point)):
            if isinstance(object, DataElement):
                origin = origin if isinstance(origin, Point) else object.envelope.centroid
                object.geometry = rotate(object.geometry, angle=angle, origin=origin)
            else:
                for element in object.elements:
                    origin = origin if isinstance(origin, Point) else element.envelope.centroid
                    element.geometry = rotate(element.geometry, angle=angle, origin=origin)
                # TODO 可能需要update整个数据结构，测试的时候需要实验一下
                object.renew()
        else:
            raise TypeError('输入了错误的参数类型，请检查参数')

    @staticmethod
    def copy_data_element(object: DataElement, model: DataModel, is_merge: bool = False) -> DataElement:
        """
        复制一个data element，拥有唯一id，
        可选择是否自动并入对应的更新到对应的data model中
        :param object:
        :return:
        """
        new = deepcopy(object)
        new.refresh_id()
        if is_merge:
            model.insert_element(insert_object=new)
        return new