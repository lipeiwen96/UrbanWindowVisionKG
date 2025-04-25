from typing import Union,Iterable
import ezdxf
import re
from ezdxf.document import Drawing
from shapely.geometry import Polygon, Point, LineString

import art_public_modules.art_data_exchange.dxf_module.dxf_file_constant as constant
from art_public_modules.art_utils.geometry_utils.shapely.shape_util import ShapelyShapeUtils
from art_public_modules.art_data_structure.shapely.core_structure import DataElement, DataModel


class DxfFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = None
        self._get_file_name()
        self.text_info_list = []  # 2022新增，用于存储模型中的一些参考点

    def _get_file_name(self):
        """
        获取当前文件名字，并用来作为data model的名字
        :return:
        """
        file_name = self.file_path.split("/")[-1]
        self.file_name = file_name.split('.')[0]

    def _convert_dxf_obj_to_shapely_obj(self, raw_entity) -> Union[Point, LineString, Polygon]:
        if raw_entity.dxftype() == "LWPOLYLINE":
            elevation = raw_entity.dxf.elevation
            vertices = [(p[0], p[1], elevation) for p in raw_entity.get_points()]
            n_vertices = len(vertices)
            bulges = [p[4] for p in raw_entity.get_points()]
            points = []
            for i, bulge in enumerate(bulges):
                if bulge == 0:
                    points.append(vertices[i])
                else:
                    # arc里已经拿掉了最后一个点
                    arc = ShapelyShapeUtils.make_arc_by_bulge(vertices[i], vertices[(i + 1) % n_vertices], bulge)
                    points.extend(arc)
                    # 因为arc中不包含其中一个端点，如果是最后一个vertex有bulge，则应该把另一个端点也加入points中
                    if i == len(bulges) - 1:
                        points.append(vertices[0])
            if len(points) > 2 and (raw_entity.closed or points[0] == points[-1]):
                geom = Polygon(points)
            else:
                geom = LineString(points)
            return geom
        elif raw_entity.dxftype() == "POLYLINE":
            _, _, elevation = raw_entity.dxf.elevation
            # Reset z to 0
            raw_entity_vertices = (raw_entity.vertices if isinstance(raw_entity.vertices, Iterable) else raw_entity.vertices())
            vertices = [(v.dxf.location[0], v.dxf.location[1], elevation) for v in raw_entity_vertices]
            n_vertices = len(vertices)
            bulges = [v.dxf.bulge for v in raw_entity_vertices]
            points = []
            for i, bulge in enumerate(bulges):
                if bulge == 0:
                    points.append(vertices[i])
                else:
                    # arc里已经拿掉了最后一个点
                    arc = ShapelyShapeUtils.make_arc_by_bulge(vertices[i], vertices[(i + 1) % n_vertices], bulge, elevation=elevation)
                    points.extend(arc)
            if len(points) > 2 and (raw_entity.is_closed or points[0] == points[-1]):
                geom = Polygon(points)
            else:
                geom = LineString(points)
            return geom
        elif raw_entity.dxftype() == "LINE":
            line = LineString([raw_entity.dxf.start, raw_entity.dxf.end])
            return line
        elif raw_entity.dxftype() == "CIRCLE":
            center = raw_entity.dxf.center
            radius = raw_entity.dxf.radius
            geom = Point(center).buffer(radius)
            return geom
        elif raw_entity.dxftype() == "ARC":
            # An arc at location center and radius from start_angle to end_angle, dxftype is ARC.
            # The arc goes from start_angle to end_angle in counter clockwise direction
            center = raw_entity.dxf.center
            radius = raw_entity.dxf.radius
            start_angle = raw_entity.dxf.start_angle
            end_angle = raw_entity.dxf.end_angle
            # end_angle比start angle大，保证arc是顺时针方向
            end_angle = end_angle + 360 if end_angle < start_angle else end_angle
            arc = ShapelyShapeUtils.make_arc_linestring(center, radius, start_angle, end_angle)
            return arc
        elif raw_entity.dxftype() == "ELLIPSE":
            """
            ELLIPSE 现在处理有点问题
            """
            raise NotImplementedError("ellipse is not supported")
        elif raw_entity.dxftype() == "SPLINE":
            control_points = raw_entity.control_points if hasattr(raw_entity, "control_points") else []
            if (len(control_points) == 0 and hasattr(raw_entity, "get_control_points") and callable(raw_entity.get_control_points)):
                control_points = raw_entity.get_control_points()
            """
            SPLINE 现在处理有点问题
            """
            points = control_points
            if len(points) > 2 and (raw_entity.closed or points[0] == points[-1]):
                geom = Polygon(points)
            else:
                geom = LineString(points)
            return geom
        elif raw_entity.dxftype() == "POINT":
            return Point((raw_entity.dxf.location[0], raw_entity.dxf.location[1]))
        elif raw_entity.dxftype() == "TEXT" or raw_entity.dxftype() == "MTEXT":
            # TODO：如果是文字标注，这里将文字标注以[text_info, point]的方式存储在data_model.user_dict{"text_info_list": []}中
            # 这里有问题，待优化
            # text = raw_entity.dxf.text
            # self.text_info_list.append()
            return Point((raw_entity.dxf.insert[0], raw_entity.dxf.insert[1]))
        else:
            # TODO(wl): using error report system when it is ready
            raise ValueError("Get enabled but not implemented type {}".format(raw_entity.dxftype()))

    def _get_file_objects(self, row_data:'Drawing')->DataModel:
        """
        将文件中的几何物件转换成shapely并转换成data structure
        :return:
        """
        objects = row_data.entities
        objects = list(filter(lambda object: object.DXFTYPE in constant.READABLE_OBJ, objects))
        data_elements = []
        for object in objects:
            if object.DXFTYPE != 'TEXT' and object.DXFTYPE != 'MTEXT':
                cur_object = self._convert_dxf_obj_to_shapely_obj(raw_entity=object)
            else:
                cur_object = self._convert_dxf_obj_to_shapely_obj(raw_entity=object)
            if isinstance(cur_object, Point):
                cur_object = Point([xy[0:2] for xy in list(cur_object.coords)])
            elif isinstance(cur_object, LineString):
                cur_object = LineString([xy[0:2] for xy in list(cur_object.coords)])
            elif isinstance(cur_object, Polygon):
                cur_object = Polygon([xy[0:2] for xy in list(cur_object.exterior.coords)])
            cur_object_height = re.findall(r'[-+]?\d*\.\d+|\d+', object.dxf.layer)
            cur_object_height = float(cur_object_height[0]) if len(cur_object_height)>0 else 0
            data_element = DataElement(geometry=cur_object, height=cur_object_height, layer=object.dxf.layer,shadow=cur_object)
            data_elements.append(data_element)
        return DataModel(elements=data_elements, name=self.file_name)

    def read_dxf_file(self):
        """
        读取dxf文件
        :return:
        """
        return self._get_file_objects(row_data=ezdxf.readfile(self.file_path))


if __name__=="__main__":
    test_file_path = r'C:\Users\hugo\Desktop\02.dxf'
    data_model = DxfFileReader(file_path=test_file_path).read_dxf_file()
    print(data_model)
