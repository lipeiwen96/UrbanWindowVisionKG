import uuid
from dataclasses import dataclass, field
from uuid import uuid1
from typing import *
from typing import AnyStr, List, Dict
import shapely.geometry.base
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from copy import deepcopy

from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial


@dataclass(order=True)
class DataElement:
    # 几何属性
    geometry: shapely.geometry.base.BaseGeometry
    shadow: shapely.geometry.base.BaseGeometry = field(default=None)
    # 特殊的几何属性，只会在data model初始化的时候初始化
    exterior_geometry: str = field(default=None)
    interior_geometry: List = field(default_factory=list)
    is_hole: str = field(default='false')  # 是否带有孔洞
    # 语义属性
    id: str = field(default=str)  # 物件ID
    geom_type: AnyStr = field(init=False)  # shapely对象类型
    layer: AnyStr = field(default=None)  # 类型名字，对渲染和文件的IO非常重要，在3dm文件输出中就是图层
    height: float = field(default=0.0)  # 高度
    start_height: float = field(default=0.0)  # L始高度
    # 材质
    material: DataElementMaterial = field(default=DataElementMaterial())
    # 自定义语义
    custom_semantics: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.geom_type = self.geometry.geom_type
        # 初始化的DataElement中的id都相同，因此需要重新进行初始化 2022-1-20改
        self.id = str(uuid.uuid4())

    @property
    def envelope(self) -> Polygon:
        """
        与坐标轴垂直的包围该Data element的外框
        :return:
        """
        return self.geometry.envelope

    def refresh_id(self):
        """
        更新id
        :return:
        """
        self.id = str(uuid1())


@dataclass(order=True)
class DataModel:
    version: AnyStr = field(default='sh')
    id: str = field(default=str(uuid.uuid4()))
    name: AnyStr = field(default=None)
    elements: List = field(default_factory=list)
    layers: List = field(default_factory=list)
    user_data: Dict = field(default_factory=dict)

    def __post_init__(self):
        self._init_all_elements()
        self._init_sp_geometry()
        self._init_model()
        # 更新图层
        self.layers = list(set([ele.layer for ele in self.elements]))

    def renew(self):
        """
        由于DataModel中的layers与self.model对应的字典有时无法匹配，因此需要将信息进行更新，使其匹配
        2020.1.20-改
        """
        self._init_all_elements()
        self._init_sp_geometry()
        self._init_model()
        # 更新图层
        self.layers = list(set([ele.layer for ele in self.elements]))

    @property
    def element_id_list(self):
        return [each.id for each in self.elements]

    @property
    def envelope(self) -> Polygon:
        """
        与坐标轴垂直的包围该Data model的外框
        :return:
        """
        return GeometryCollection([each.geometry for each in self.elements]).envelope

    @property
    def all_geometry(self) -> GeometryCollection:
        return GeometryCollection([each.geometry for each in self.elements])

    def _init_all_elements(self):
        """
        根据目前的所有的elements进行拍平
        :return:
        """
        flatten_elements = []
        for each in self.elements:
            if 'Multi' in each.geom_type:
                flatten_elements.extend(self._flatten_data_elements(data_element=each))
            else:
                flatten_elements.append(each)
        for each in flatten_elements:
            each.geom_type = each.geometry.geometryType()
        self.elements = flatten_elements

    def _init_sp_geometry(self):
        """
        用于处理每个几何对象对外输出的坐标，主要是需要针对带有孔洞的部分
        :return:
        """
        for each in self.elements:
            if isinstance(each.geometry, Point):
                each.exterior_geometry = each.geometry
            elif isinstance(each.geometry, LineString):
                each.exterior_geometry = each.geometry
            elif isinstance(each.geometry, Polygon):
                if len(list(each.geometry.interiors))>0:
                    each.exterior_geometry = Polygon(each.geometry.exterior)
                    each.interior_geometry = [Polygon(_) for _ in list(each.geometry.interiors)]
                    each.is_hole = 'true'
                else:
                    each.exterior_geometry = Polygon(each.geometry.exterior)

    def _init_model(self):
        if len(self.elements) > 0 or isinstance(self.elements, List):
            try:
                layer_list = list(set([each.layer for each in self.elements]))
                model = {}
                for name in layer_list:
                    model[f'{name}'] = []
                for each in self.elements:
                    model[f'{each.layer}'].append(each)
                self.model = model
            except Exception as error:
                # 传入的data element不是正确的data element
               print(f'发生错误{error}')
        else:
            raise ValueError("没有接手到任何DataElement的list")

    def _flatten_data_elements(self, data_element: DataElement) -> List[DataElement]:
        """
        拍平所有data elements中的mulit-geomentry
        拆出来的部分跟原有的部分的type是一致的
        :return:
        """
        res = []
        all_geometry = list(data_element.geometry.geoms)
        for g in all_geometry:
            cur_element = deepcopy(data_element)
            cur_element.geometry = g
            cur_element.geom_type = g.geom_type
            cur_element.id = str(uuid1())
            res.append(cur_element)
        return res

    def insert_element(self, insert_object: Union[DataElement, List[DataElement]]):
        """
        在现在的data model上插入一个data element，如果现在的data model中没有该layer，进行新增，如果有则在该layer上插入
        可以插入单个data element或data element的list
        如果该data element已存在于data model中就不会插入，通过id来识别是否存在该data element
        :param insert_object:
        :return:
        """
        if isinstance(insert_object, DataElement):
            if insert_object.layer not in self.layers:
                self.layers.append(insert_object.layer)
                self.model[f'{insert_object.layer}'] = []
            if 'Multi' in insert_object.geom_type:
                cur_elements = self._flatten_data_elements(data_element=insert_object)
                self.model[f'{insert_object.layer}'].extend(cur_elements)
                self.elements.extend(cur_elements)
            else:
                self.model[f'{insert_object.layer}'].append(insert_object)
                self.elements.append(insert_object)
        elif isinstance(insert_object, List):
            insert_type = list(filter(lambda each: each not in self.layers, list(set([each.layer for each in self.elements]))))
            self.layers.extend(insert_type)
            try:
                for element in insert_object:
                    if 'Multi' in element.geom_type:
                        cur_elements = self._flatten_data_elements(data_element=element)
                        self.elements.extend(cur_elements)
                    else:
                        self.elements.append(element)
            except Exception as error:
                print(f'发生错误:{error}')
        else:
            raise ValueError("要插入的数据类型有错误")

    def delete_elements_by_layer_name(self, layer_name: str):
        """
        删除对应类型名字的所有的data elements
        :param type_name:
        :return:
        """
        if layer_name in self.layers:
            self.elements = list(filter(lambda each:each.layer != layer_name, self.elements))
            self.layers.remove(layer_name)
        else:
            print(f'该data model不存在{layer_name}图层')

    def select_elements_by_layer(self, layer_name: str) -> List:
        """
        根据type名字返回该type下的所有data element
        :param type_name:
        :return:
        """
        if layer_name in self.layers:
            return list(filter(lambda each:each.layer==layer_name, self.elements))
        else:
            print(f'该data model不存在{layer_name}图层')

    def select_elements_by_geom_type(self, geom_type: str) -> List[DataElement]:
        """
        通过几何类型选择对应的elements
        :param geom_type:
        :return:
        """
        VAILD_GEOM_TYPE = ['Point','LineString','Polygon']
        if geom_type in VAILD_GEOM_TYPE:
            return list(filter(lambda each:each.geom_type == geom_type, self.elements))
        else:
            raise ValueError("不存在该几何类型")

    def elements_types(self) -> List:
        """
        返回所有的elements的type的名称
        :return:
        """
        if isinstance(self.model, Dict):
            return list(self.model.keys())
        else:
            raise ValueError("model属性中包含了错误的数据")