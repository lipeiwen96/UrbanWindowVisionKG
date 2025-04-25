"""
读取rhino 3dm文件
并封装成data upload_model
"""
import re
from typing import Dict, List, Union
import rhino3dm
from rhino3dm._rhino3dm import File3dm
from shapely.geometry import Polygon, LineString, Point

from art_public_modules.art_data_exchange.rhino_module.rhino_file_constant import READABLE_OBJ
from art_public_modules.art_data_structure.shapely.core_structure import DataElement, DataModel


class RhinoFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = None
        self._get_file_name()

    def _get_file_name(self):
        """
        获取当前文件名字，并用来作为data model的名字
        :return:
        """
        file_name = self.file_path.split("/")[-1]
        self.file_name = file_name.split('.')[0]

    def _export_file_layers(self, row_data: File3dm)->Dict:
        """
        用于读取3dm文件中的所有图层信息
        :param row_data:
        :return:
        """
        layer_index = [each.Index for each in row_data.Layers]
        layer_name = [each.Name for each in row_data.Layers]
        layer_info = dict(map(lambda index, name: (index, name), layer_index, layer_name))
        return layer_info

    def _catch_height(self, layer_name:str)->float:
        """
        从图层信息中捉取高度信息
        :param layer_name:
        :return:
        """
        num = re.findall(r'[-+]?\d*\.\d+|\d+', layer_name)
        return float(num[0]) if len(num) >0 else 0

    def _convert_rhino_obj_to_shapely_obj(self,obj: rhino3dm._rhino3dm.ObjectType) -> Union[None, Point, Polygon, LineString]:
        """
        将rhino的对象转换成shapely的几何对象
        :param obj:
        :return:
        """
        rhino_geometry = obj.Geometry
        try:
            # 处理 curve
            if str(rhino_geometry.ObjectType) == 'ObjectType.Curve':
                # 处理只有兩個點的短綫
                if rhino_geometry.SpanCount == 1:
                    # 两个点的直线->直线
                    shapely_geometry = LineString([(rhino_geometry.PointAtStart.X, rhino_geometry.PointAtStart.Y),
                                                   (rhino_geometry.PointAtEnd.X, rhino_geometry.PointAtEnd.Y)])
                    return shapely_geometry
                # 闭合曲线->polygon
                elif rhino_geometry.IsPolyline() and rhino_geometry.IsClosed:
                    polyline_points = rhino_geometry.TryGetPolyline()
                    shapely_geometry = Polygon([(point.X, point.Y) for point in polyline_points])
                    return shapely_geometry
                # 不闭合的多段线->多段线
                elif rhino_geometry.IsPolyline() and rhino_geometry.IsClosed == False:
                    polyline_points = [l for l in rhino_geometry.ToPolyline()]
                    shapely_geometry = LineString([(point.X, point.Y) for point in polyline_points])
                    return shapely_geometry
                else:
                    print([item for item in rhino_geometry])
                    print('该3dm文件中有无法转换成shapely的对象', rhino_geometry)
                    return None
            # # 处理短綫
            # if str(rhino_geometry.ObjectType) == 'ObjectType.Line':
            #     print(rhino_geometry.From, rhino_geometry.To)
            #     # 两个点的直线->直线
            #     shapely_geometry = LineString([rhino_geometry.From, rhino_geometry.To])
            #     return shapely_geometry
            # 处理点
            elif str(rhino_geometry.ObjectType) == 'ObjectType.Point3d':
                shapely_geometry = Point((rhino_geometry.X, rhino_geometry.Y))
                return shapely_geometry

            # 把TextDot转换成点，渲染的时候把文字渲染到对应的位置上
            elif str(rhino_geometry.ObjectType) == 'ObjectType.TextDot':
                shapely_geometry = Point(
                    (rhino_geometry.GetBoundingBox().Center.X, rhino_geometry.GetBoundingBox().Center.Y))
                return shapely_geometry

            # 把文字对象转换成点，渲染的时候把文字渲染到对应的位置上
            elif str(rhino_geometry.ObjectType) == 'ObjectType.Annotation':
                # 暂时无法处理这个东西
                pass

            # 处理点
            elif str(rhino_geometry.ObjectType) == 'ObjectType.Point':
                shapely_geometry = Point(
                    (rhino_geometry.Location.X, rhino_geometry.Location.Y))
                return shapely_geometry
            # TODO 增加更多类型对象的处理方式
            
        except:
            print('该3dm文件中有无法转换成shapely的对象')
            return None

    def _rhino_geometry_object_processor(self, row_data:File3dm, layers_info:Dict)->List[DataElement]:
        """
        将可以识别的3dm物件转换成art data element 对象
        :param row_data:
        :param layers_info:
        :return:
        """
        result = []
        rhino_objects = row_data.Objects
        valid_objects = list(filter(lambda each: str(each.Geometry.ObjectType) in READABLE_OBJ, rhino_objects))
        if len(valid_objects)>0:
            for each_obj in valid_objects:
                # 该物件所属的图层
                obj_layer = layers_info[each_obj.Attributes.LayerIndex]
                # 捉取该图层的高度信息
                obj_height = self._catch_height(layer_name=obj_layer)
                # 把该对象转换成shapely对象
                obj_geometry = self._convert_rhino_obj_to_shapely_obj(obj=each_obj)
                # 把当前对象变成data element
                cur_data_element = DataElement(geometry=obj_geometry, height=obj_height, layer=obj_layer)
                result.append(cur_data_element)
            return result
        else:
            raise SystemExit('没有能转换的rhino对象')

    def read_3dm_file(self):
        """
        读取3dm文件并组装成data upload_model
        :return:
        """
        # 尝试读取3dm文件
        row_data = rhino3dm.File3dm.Read(self.file_path)
        # 获取读取的3dm文件中的图层信息
        layers_info = self._export_file_layers(row_data=row_data)
        # 读取所有的几何对象并转换成data element TODO 需要增加中空物件的处理
        all_data_elements = self._rhino_geometry_object_processor(row_data=row_data, layers_info=layers_info)
        # 把所有的data element组合到data model中并返回
        return DataModel(elements=all_data_elements, name=self.file_name)


if __name__ == "__main__":
    TEST_FILE_PATH = '../../test_file/02.3dm'
    data_model = RhinoFileReader(file_path=TEST_FILE_PATH).read_3dm_file()
    print(data_model)
