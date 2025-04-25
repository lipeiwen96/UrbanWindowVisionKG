from random import randint
from typing import List, Tuple
import rhino3dm
from rhino3dm._rhino3dm import File3dm, Point3d, Layer, Group, ObjectAttributes, Curve, Polyline, Extrusion, Brep
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import orient

from art_public_modules.art_data_exchange.rhino_module.rhino_file_constant import WRITE_VERSION, RHINO_LAYER_COLOR_DICT
from art_public_modules.art_data_exchange.rhino_module.rhino_file_reader import RhinoFileReader
from art_public_modules.art_data_structure.shapely.core_structure import DataElement, DataModel


class RhinoFileWriter:
    def __init__(self, file_path:str):
        self.file_path = file_path
        self.doc = File3dm()

    def _add_layer_to_file(self, data_model:DataModel):
        """
        将data model中的type转变成图层
        :param data_model:
        :return:
        """
        layer_names = data_model.elements_types()
        if len(layer_names) > 0:
            for name in layer_names:
                cur_layer = Layer()
                cur_layer.Name = name
                cur_layer.Visible = True

                # 沛文修改 2020-8-18 根据字典修改保存图层的颜色
                if name in RHINO_LAYER_COLOR_DICT.keys():
                    cur_layer.Color = RHINO_LAYER_COLOR_DICT[name]
                else:
                    cur_layer.Color = (randint(0, 255), randint(0, 255), randint(0, 255), 255)

                self.doc.Layers.Add(cur_layer)
        else:
            cur_layer = Layer()
            cur_layer.Name = '0'
            cur_layer.Visible = True
            cur_layer.Color = (randint(0, 255), randint(0, 255), randint(0, 255), 255)
            self.doc.Layers.Add(cur_layer)

        if "solution_text" in data_model.user_data.keys():
            cur_layer = Layer()
            cur_layer.Name = '文字'
            cur_layer.Visible = True
            cur_layer.Color = (0, 0, 0, 255)
            self.doc.Layers.Add(cur_layer)

    def _transform_elements_to_rhino_objects(self, element: DataElement) -> Tuple[List[Point3d], List[List[Point3d]], List[Polyline], List[Extrusion]]:
        points = []
        lines = []
        polylines = []
        extrusions = []

        geometry = element.exterior_geometry
        interior = element.interior_geometry
        z = element.start_height
        height = element.height
        # 轉換點
        if isinstance(geometry, Point):
            point_object = Point3d(geometry.x, geometry.y, z)
            points.append(point_object)
        # 轉換短的多段綫
        elif isinstance(geometry, LineString) and len(list(geometry.coords)) == 2:
            short_line_points = [Point3d(coord[0], coord[1], z) for coord in list(geometry.coords)]
            lines.append(short_line_points)
        # 轉換多段綫
        elif isinstance(geometry, LineString) and len(list(geometry.coords)) > 2:
            line_points = [Point3d(coord[0], coord[1], z) for coord in list(geometry.coords)]
            line_object = Polyline(line_points)
            polylines.append(line_object)
        # 轉換多邊形
        elif isinstance(geometry, Polygon):
            polygon_points = [Point3d(coord[0], coord[1], z) for coord in list(orient(geometry).exterior.coords)]
            polygon_object = Polyline(polygon_points)
            polylines.append(polygon_object)
            # 挤出部分
            if height != 0:
                planar_curve = Curve.CreateControlPointCurve(polygon_object, degree=1)
                extrusion_object = Extrusion.Create(planar_curve, height=height, cap=True)
                extrusions.append(extrusion_object)
        # 内部需要扣掉的部分
        elif isinstance(interior, Polygon):
            polygon_points = [Point3d(coord[0], coord[1], z) for coord in list(orient(geometry).exterior.coords)]
            polygon_object = Polyline(polygon_points)
            polylines.append(polygon_object)
            # 挤出部分
            if height != 0:
                planar_curve = Curve.CreateControlPointCurve(polygon_object, degree=1)
                extrusion_object = Extrusion.Create(planar_curve, height=height, cap=True)
                extrusions.append(extrusion_object)

        return points, lines, polylines, extrusions

    def _data_structure_processor(self, data_model:DataModel):
        """
        转化data model为rhino内部格式
        :param data_model:
        :return:
        """
        # 添加图层
        self._add_layer_to_file(data_model=data_model)
        # 图层名
        layer_index = [each.Index for each in self.doc.Layers]
        layer_name = [each.Name for each in self.doc.Layers]
        layer_info = list(map(lambda index, name: (index, name), layer_index, layer_name))

        # 转换对象，匹配图层并写入到该3dm文件中
        for element in data_model.elements:
            layer_index = [item[0] for item in layer_info if item[1] == element.layer]
            # 转换物件
            id_list = []
            attribute = ObjectAttributes()
            points, lines, polylines, extrusions = self._transform_elements_to_rhino_objects(element=element)
            if points:
                if element.custom_semantics is None:
                    for point in points:
                        id = self.doc.Objects.AddPoint(point)
                        id_list.append(id)
            if lines:
                for line in lines:
                    id = self.doc.Objects.AddLine(line[0], line[1])
                    id_list.append(id)
            if polylines:
                for polyline in polylines:
                    id = self.doc.Objects.AddPolyline(polyline, attribute)
                    id_list.append(id)
            if extrusions:
                for extrusion in extrusions:
                    id = self.doc.Objects.AddExtrusion(extrusion, attribute)
                    id_list.append(id)

            # 匹配物件信息
            for object_id in id_list:
                cur_object = self.doc.Objects.FindId(str(object_id))
                cur_object.Attributes.LayerIndex = layer_index[0]


        # 创建树冠
        # print("Generating Tree... 正在生成景观树")
        # for element in data_model.elements:
        #     if element.custom_semantics:
        #         meshes = [element.custom_semantics["tree_geometry"]]
        #         tree_id = self.doc.Objects.AddMesh(meshes[0], ObjectAttributes())
        #         tree_obj = self.doc.Objects.FindId(str(tree_id))
        #         tree_obj.Attributes.LayerIndex = self.__find_layer_index_by_layer_name("XKOOL_Gen_Trees")
        # print("Generating Successfully！景观树生成成功！")



        # 创建文字标注
        if "solution_text" in data_model.user_data.keys():
            for text in data_model.user_data["solution_text"]:
                text_id = self.doc.Objects.AddTextDot(text[0],
                                                      Point3d(text[1][0],
                                                              text[1][1],
                                                              text[1][2]),
                                                      ObjectAttributes())
                text_obj = self.doc.Objects.FindId(str(text_id))
                text_obj.Attributes.LayerIndex = self.__find_layer_index_by_layer_name("文字")

    def __find_layer_index_by_layer_name(self, layer_name: str):
        # 所有的图层序号及图层名为:
        all_layer_index = [each.Index for each in self.doc.Layers]
        all_layer_name = [each.Name for each in self.doc.Layers]
        if layer_name in all_layer_name:
            layer_index = all_layer_name.index(layer_name)
            return all_layer_index[layer_index]
        else:
            layer_index = all_layer_name.index("Default")
            return all_layer_index[layer_index]

    def write_3dm_file(self, data_model:DataModel):
        """
        根据输入的数据类型选择处理方式
        :param data_model:
        :return:
        """
        if not isinstance(data_model, DataModel):
            raise ValueError("輸入數據不是DataModel, 目前仅支持ART的DataModel")

        self._data_structure_processor(data_model=data_model)
        self.doc.Write(self.file_path, version=WRITE_VERSION)


if __name__=="__main__":
    TEST_FILE_PATH = '../../test_file/02withouSPL.3dm'
    TEST_FILE_PATH2 = '../../test_file/new'
    data_model = RhinoFileReader(file_path=TEST_FILE_PATH).read_3dm_file()
    new=RhinoFileWriter(file_path=TEST_FILE_PATH2).write_3dm_file(data_model)
