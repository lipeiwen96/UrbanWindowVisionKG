"""
Model Constructor
To construct 3D Standard-Level Model on Web front-end
@ Peiwen Li | ART
"""
from dataclasses import dataclass, field
import math
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import orient


from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement
from art_public_modules.art_utils.geometry_utils.shapely.geometry_utils import ShapelyLineUtils, ShapelyPointUtils
from modules.temp.closed_polygon_to_wall_segs import closed_polygon_to_2_walls
from modules.temp.window_rule import get_window_rectangle
from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial


def compute_length_width(box: Polygon):
    pt1 = Point(box.exterior.coords[0])
    pt2 = Point(box.exterior.coords[1])
    pt3 = Point(box.exterior.coords[2])
    line1 = pt2.distance(pt1)
    line2 = pt2.distance(pt3)
    return max(line1, line2), min(line1, line2)


def get_middle_line_pts(polygon: Polygon, pt_on_long_edge: bool = False):
    """
    获取一个四边形最长/最短的两条边的中点
    """
    # 所有坐标点
    all_coords = list(polygon.exterior.coords)
    all_lines = [LineString([all_coords[i], all_coords[i + 1]]) for i in range(len(all_coords) - 1)]
    # 重新排序
    all_lines = sorted(all_lines, key=lambda each: each.length)
    if pt_on_long_edge:
        return [all_lines[-1].centroid, all_lines[-2].centroid]
    else:
        return [all_lines[0].centroid, all_lines[1].centroid]


def get_extreme_edge(polygon: Polygon, max_edge: bool = True):
    all_coords = list(polygon.exterior.coords)
    all_lines = [LineString([all_coords[i], all_coords[i + 1]]) for i in range(len(all_coords) - 1)]
    # 重新排序
    all_lines = sorted(all_lines, key=lambda each: each.length)
    if max_edge:
        return all_lines[-1]
    else:
        return all_lines[0]


def scale_rectangle_inplace(rectangle: Polygon, scaled_lenth: float, scaled_width: float):
    """
    根据输入的矩形，保持中心点不变，缩放矩形的大小、比例，使缩放后的矩形长度为scaled_lenth，宽度为scaled_width
    """
    return create_rectangle_from_pt_inplace(input_pt=Point(rectangle.centroid),
                                                           rectangle=rectangle, lenth=scaled_lenth, width=scaled_width)


def create_rectangle_from_pt_inplace(input_pt: Point, rectangle: Polygon, lenth: float, width: float):
    """
    根据输入的矩形方向，以输入点为中心点，创建指定长宽的矩形
    """
    # 获取矩形长边
    long_edge = get_extreme_edge(rectangle, max_edge=True)
    state, k, b = ShapelyLineUtils.line_equation(Point(long_edge.coords[0]), Point(long_edge.coords[1]))  # 长边表达式
    # 开始缩放操作，首先计算中心点的偏移量
    if state == 0:
        # x = b
        return Polygon([Point(input_pt.x - width / 2, input_pt.y + lenth / 2),
                        Point(input_pt.x - width / 2, input_pt.y - lenth / 2),
                        Point(input_pt.x + width / 2, input_pt.y - lenth / 2),
                        Point(input_pt.x + width / 2, input_pt.y + lenth / 2),
                        Point(input_pt.x - width / 2, input_pt.y + lenth / 2)])
    elif k == 0:
        # y = b
        return Polygon([Point(input_pt.x - lenth / 2, input_pt.y + width / 2),
                        Point(input_pt.x - lenth / 2, input_pt.y - width / 2),
                        Point(input_pt.x + lenth / 2, input_pt.y - width / 2),
                        Point(input_pt.x + lenth / 2, input_pt.y + width / 2),
                        Point(input_pt.x - lenth / 2, input_pt.y + width / 2)])
    else:
        # 长边法向量
        v1, v2 = ShapelyLineUtils.compute_line_normal_vector(Point(long_edge.coords[0]), Point(long_edge.coords[1]))
        # 长边的偏移单位
        length_ratio = lenth / (2 * math.sqrt(1 + k * k))
        # 获取中心点延长边方向偏移后的两个参考点
        ref_pt1 = Point(input_pt.x + length_ratio, input_pt.y + k * length_ratio)
        ref_pt2 = Point(input_pt.x - length_ratio, input_pt.y - k * length_ratio)
        # 短边的偏移单位
        short_ratio = width / (2 * math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]))
        new_pt1 = Point(ref_pt1.x + v1[0] * short_ratio, ref_pt1.y + v1[1] * short_ratio)
        new_pt2 = Point(ref_pt1.x + v2[0] * short_ratio, ref_pt1.y + v2[1] * short_ratio)
        new_pt3 = Point(ref_pt2.x + v1[0] * short_ratio, ref_pt2.y + v1[1] * short_ratio)
        new_pt4 = Point(ref_pt2.x + v2[0] * short_ratio, ref_pt2.y + v2[1] * short_ratio)
        return Polygon([new_pt1, new_pt2, new_pt4, new_pt3, new_pt1])


@dataclass(order=True)
class ModelConstructor:
    floor_num: int = field(default=10)
    bottom_height: float = field(default=0.2)  # 楼板厚度
    start_height: float = field(default=0)  # 基准高度
    level_height: float = field(default=3)  # 模型高度
    # 门相关参数
    door_stick_thickness: float = field(default=0.08)  # 门框厚度
    door_max_width: float = field(default=1.0)  # 门板最大宽度
    door_height: float = field(default=1.9)
    # 窗户相关参数
    window_bottom_wall_height: float = field(default=1)  # 窗底部墙体高度
    window_stick_thickness: float = field(default=0.05)  # 窗框厚度
    window_stick_dis: float = field(default=0.6)  # 窗框杆件间隔
    window_model_height: float = field(default=1.2)  # 窗户模型的总高度
    window_top_wall_height: float = field(default=0.1)  # 窗顶部墙体高度
    # 栏杆相关参数
    railing_wall_height: float = field(default=0.2)  # 栏杆底部墙体高度
    railing_model_height: float = field(default=0.8)  # 栏杆高度
    # 女儿墙厚度
    topwall_width: float = field(default=0.2)

    # 标准层中的户型数量
    apartment_num: int = field(default=0)

    construct_model: DataModel = field(default_factory=DataModel)

    def construct_3D_model(self, input_data_model: DataModel):
        self.construct_model = DataModel()
        self.window_top_wall_height = self.level_height - self.window_bottom_wall_height - self.window_model_height - self.bottom_height
        self.windows_kitchen_office = []
        self.windows_habitat = []
        self.building_outline = Polygon()

        for element in input_data_model.elements:
            if element.layer == "xkool-wall":
                self.__construct_wall(element)
            elif element.layer == "xkool-window":
                self.__construct_window(element)
            elif element.layer == "xkool-door":
                self.__construct_door(element)
            elif element.layer == "xkool-railing":
                self.__construct_railing(element)
            elif element.layer == "xkool-outline":
                if isinstance(element.geometry, Polygon):
                    self.building_outline = element.geometry
                    # 楼板
                    for i in range(self.floor_num):
                        gen_element = DataElement(geometry=element.geometry,
                                                  layer="Generate_base",
                                                  start_height=self.start_height + i * self.level_height, height=self.bottom_height)
                        self.construct_model.insert_element(gen_element)
                    # 顶板
                    gen_element = DataElement(geometry=element.geometry,
                                              layer="Generate_base",
                                              start_height=self.start_height + self.floor_num * self.level_height, height=self.bottom_height)
                    self.construct_model.insert_element(gen_element)
                    # 女儿墙
                    # closed_polygon_to_2_walls
                    wall = closed_polygon_to_2_walls(element.geometry, self.topwall_width)
                    print(wall)
                    for w in wall:
                        print(w)
                        gen_element = DataElement(geometry=w,
                                                  layer="Generate_wall",
                                                  start_height=self.start_height + self.floor_num * self.level_height + self.bottom_height, height=1)
                        self.construct_model.insert_element(gen_element)
            elif element.layer == "xkool-core":
                # 顶板
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Generate_wall",
                                          start_height=self.start_height + self.floor_num * self.level_height,
                                          height=self.bottom_height + self.level_height)
                self.construct_model.insert_element(gen_element)
            elif element.layer == "xkool-win-habitat":
                # if isinstance(element.geometry, LineString):
                print(element.geometry)
                self.windows_habitat.append(element.geometry)
            elif element.layer == "xkool-win-kitchen":
                # if isinstance(element.geometry, LineString):
                print(element.geometry)
                self.windows_kitchen_office.append(element.geometry)
            elif element.layer == "site":
                # 顶板
                gen_element = DataElement(geometry=element.geometry,
                                          layer="site",
                                          start_height=0-self.bottom_height,
                                          height=self.bottom_height)
                self.construct_model.insert_element(gen_element)
                print("地块信息:", element.geometry)
            elif element.layer == "GIC":
                # 顶板
                gen_element = DataElement(geometry=element.geometry,
                                          layer="building",
                                          start_height=0,
                                          height=8 * self.level_height)
                self.construct_model.insert_element(gen_element)

            elif element.layer == "line":
                # 顶板
                gen_element = DataElement(geometry=element.geometry,
                                          layer="inside_road",
                                          start_height=0,
                                          height=0.3)
                self.construct_model.insert_element(gen_element)
                print("内部道路信息:", element.geometry)
            elif element.layer == "Xkool_ProjectBoundary":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_ProjectBoundary",
                                          material=DataElementMaterial(color="0xe5ebf1"),
                                          start_height=-21,
                                          height=20)
                self.construct_model.insert_element(gen_element)
            elif element.layer == "Xkool_PlotBoundary":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_PlotBoundary",
                                          material=DataElementMaterial(color="0xced4d6"),
                                          start_height=-1,
                                          height=0.65)
                self.construct_model.insert_element(gen_element)
            elif element.layer == "Xkool_SiteBoundary":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_SiteBoundary",
                                          material=DataElementMaterial(color="0xc7e29f"),
                                          start_height=-0.35,
                                          height=0.35)
                self.construct_model.insert_element(gen_element)
            elif element.layer == "Xkool_InsideUplacedArea":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_InsideUplacedArea",
                                          material=DataElementMaterial(color="0xbe2323"),
                                          start_height=0.15)
                self.construct_model.insert_element(gen_element)
            elif element.layer == "Xkool_SurroundBuilding":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_SurroundBuilding",
                                          material=DataElementMaterial(color="0xbbbec2"),
                                          start_height=0, height=24)
                self.construct_model.insert_element(gen_element)
            elif element.layer == "Xkool_SurroundBuildingHigh":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_SurroundBuilding",
                                          material=DataElementMaterial(color="0xbbbec2"),
                                          start_height=0, height=80)
                self.construct_model.insert_element(gen_element)
            elif element.layer == "Xkool_BuildingNDOutline":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_BuildingNDOutline",
                                          material=DataElementMaterial(color="0xffcc50"),
                                          start_height=0, height=10, custom_semantics={"floor_height": 5, "storey": 2})
                self.construct_model.insert_element(gen_element)
            elif element.layer == "Xkool_BuildingDOutline":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_BuildingDOutline",
                                          material=DataElementMaterial(color="0xffffff"),
                                          start_height=10, height=63, custom_semantics={"floor_height": 3.15, "storey": 20})
                self.construct_model.insert_element(gen_element)
            elif element.layer == "Xkool_BuildingDistance":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_BuildingDistance",
                                          material=DataElementMaterial(color="0x3c3cff"),
                                          start_height=12.5)
                self.construct_model.insert_element(gen_element)
            elif element.layer == "Xkool_WindowDistance":
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Xkool_WindowDistance",
                                          material=DataElementMaterial(color="0x3c3cff"),
                                          start_height=12.5)
                self.construct_model.insert_element(gen_element)
            # else:
            #     gen_element = DataElement(geometry=element.geometry,
            #                               layer="Line")
            #     self.construct_model.insert_element(gen_element)
            #     print(element.geometry)

        # print(self.windows_kitchen_office)
        # print(self.windows_habitat)
        print(self.building_outline)
        if len(self.windows_kitchen_office) > 0 or len( self.windows_habitat) > 0:
            self.total_height = self.floor_num * self.level_height - self.window_bottom_wall_height
            a, b = get_window_rectangle(self.windows_kitchen_office, self.windows_habitat, self.building_outline, self.total_height)
            for sa in a:
                gen_element = DataElement(geometry=sa,
                                          layer="dis")
                self.construct_model.insert_element(gen_element)
            for sb in b:
                gen_element = DataElement(geometry=sb,
                                          layer="dis")
                self.construct_model.insert_element(gen_element)

        self.construct_model.renew()

    def __construct_wall(self, element: DataElement):
        # 重建墙体模型
        if isinstance(element.geometry, Polygon):
            for i in range(self.floor_num):
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Generate_wall",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height,
                                          height=self.level_height - self.bottom_height)
                self.construct_model.insert_element(gen_element)
        else:
            print(f"---WARNING---检测到一个未闭合的墙体图形, 类型为：{element.geometry.geometryType()}, 请检查！")

    def __construct_window(self, element: DataElement):
        # 重建窗台模型
        if isinstance(element.geometry, Polygon):

            for i in range(self.floor_num):
                # 生成窗台墙
                # 01-窗台底部墙
                gen_element = DataElement(geometry=orient(element.geometry, sign=1),
                                          layer="Generate_wall",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height,
                                          height=self.window_bottom_wall_height)
                self.construct_model.insert_element(gen_element)
                # 02-窗台顶部墙
                gen_element = DataElement(geometry=orient(element.geometry, sign=1),
                                          layer="Generate_wall",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.window_model_height + self.window_bottom_wall_height,
                                          height=self.window_top_wall_height)
                self.construct_model.insert_element(gen_element)
                # 生成窗底板
                window_bottom = scale_rectangle_inplace(element.geometry,
                                                        scaled_lenth=get_extreme_edge(element.geometry, max_edge=True).length,
                                                        scaled_width=self.window_stick_thickness)
                gen_element = DataElement(geometry=orient(window_bottom, sign=1),
                                          layer="Generate_window_model",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.window_bottom_wall_height,
                                          height=self.window_stick_thickness)
                self.construct_model.insert_element(gen_element)
                # 生成窗玻璃 0216
                glass = window_bottom.buffer(-0.01)
                gen_element = DataElement(geometry=orient(glass, sign=1),
                                          layer="Generate_window_glass",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.window_bottom_wall_height + self.window_stick_thickness,
                                          height=self.window_model_height - 2 * self.window_stick_thickness)
                self.construct_model.insert_element(gen_element)
                # 生成窗顶板
                gen_element = DataElement(geometry=orient(window_bottom, sign=1),
                                          layer="Generate_window_model",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.window_bottom_wall_height + self.window_model_height - self.window_stick_thickness,
                                          height=self.window_stick_thickness)
                self.construct_model.insert_element(gen_element)
                # 生成窗杆件
                middle_line_pt1, middle_line_pt2 = get_middle_line_pts(window_bottom, pt_on_long_edge=False)
                start_pt = Point(LineString([middle_line_pt1, middle_line_pt2]).interpolate(self.window_stick_thickness / 2))
                end_pt = Point(LineString([middle_line_pt2, middle_line_pt1]).interpolate(self.window_stick_thickness / 2))
                gen_element = DataElement(geometry=orient(create_rectangle_from_pt_inplace(start_pt, window_bottom, self.window_stick_thickness, self.window_stick_thickness), sign=1),
                                          layer="Generate_window_model",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.window_bottom_wall_height + self.window_stick_thickness,
                                          height=self.window_model_height - 2 * self.window_stick_thickness)
                self.construct_model.insert_element(gen_element)
                gen_element = DataElement(geometry=orient(create_rectangle_from_pt_inplace(end_pt, window_bottom, self.window_stick_thickness, self.window_stick_thickness), sign=1),
                                          layer="Generate_window_model",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.window_bottom_wall_height + self.window_stick_thickness,
                                          height=self.window_model_height - 2 * self.window_stick_thickness)
                self.construct_model.insert_element(gen_element)
                # 生成分割杆件
                seg_num = int(start_pt.distance(end_pt) / self.window_stick_dis)
                if seg_num < 2:
                    pass
                else:
                    pt_list = ShapelyPointUtils.get_segment_pts(start_pt, end_pt, divide_seg_num=seg_num)
                    for pt in pt_list:
                        gen_element = DataElement(geometry=orient(create_rectangle_from_pt_inplace(pt, window_bottom, self.window_stick_thickness, self.window_stick_thickness), sign=1),
                                                  layer="Generate_window_model",
                                                  start_height=self.start_height + self.bottom_height + i * self.level_height + self.window_bottom_wall_height + self.window_stick_thickness,
                                                  height=self.window_model_height - 2 * self.window_stick_thickness)
                        self.construct_model.insert_element(gen_element)
        else:
            print(f"---WARNING---检测到一个未闭合的墙体图形, 类型为：{element.geometry.geometryType()}, 请检查！")

    def __construct_door(self, element: DataElement):
        # 重建门模型
        if isinstance(element.geometry, Polygon):
            for i in range(self.floor_num):
                # 门顶部墙
                gen_element = DataElement(geometry=orient(element.geometry, sign=1),
                                          layer="Generate_wall",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.door_height,
                                          height=self.level_height - self.bottom_height - self.door_height)
                self.construct_model.insert_element(gen_element)
                # 生成门定板
                window_bottom = scale_rectangle_inplace(element.geometry,
                                                        scaled_lenth=get_extreme_edge(element.geometry, max_edge=True).length,
                                                        scaled_width=self.window_stick_thickness)

                gen_element = DataElement(geometry=orient(window_bottom, sign=1),
                                          layer="Generate_window_model",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.door_height - self.door_stick_thickness,
                                          height=self.door_stick_thickness)
                self.construct_model.insert_element(gen_element)
                # 生成门玻璃 0216
                glass = window_bottom.buffer(-0.01)
                gen_element = DataElement(geometry=orient(glass, sign=1),
                                          layer="Generate_window_glass",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height,
                                          height=self.door_height)
                self.construct_model.insert_element(gen_element)
                # 生成窗顶板
                # 生成窗杆件
                middle_line_pt1, middle_line_pt2 = get_middle_line_pts(window_bottom, pt_on_long_edge=False)
                start_pt = Point(LineString([middle_line_pt1, middle_line_pt2]).interpolate(self.window_stick_thickness / 2))
                end_pt = Point(LineString([middle_line_pt2, middle_line_pt1]).interpolate(self.window_stick_thickness / 2))
                gen_element = DataElement(geometry=orient(create_rectangle_from_pt_inplace(start_pt, window_bottom, self.window_stick_thickness, self.window_stick_thickness), sign=1),
                                          layer="Generate_window_model",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height,
                                          height=self.door_height - self.window_stick_thickness)
                self.construct_model.insert_element(gen_element)
                gen_element = DataElement(geometry=orient(create_rectangle_from_pt_inplace(end_pt, window_bottom, self.window_stick_thickness, self.window_stick_thickness), sign=1),
                                          layer="Generate_window_model",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height,
                                          height=self.door_height - self.window_stick_thickness)
                self.construct_model.insert_element(gen_element)
                # 生成分割杆件
                seg_num = start_pt.distance(end_pt) / self.door_max_width
                if seg_num < 1.2:
                    pass
                else:
                    seg_num = math.ceil(seg_num)
                    pt_list = ShapelyPointUtils.get_segment_pts(start_pt, end_pt, divide_seg_num=seg_num)
                    for pt in pt_list:
                        gen_element = DataElement(geometry=orient(create_rectangle_from_pt_inplace(pt, window_bottom, self.window_stick_thickness, self.window_stick_thickness), sign=1),
                                                  layer="Generate_window_model",
                                                  start_height=self.start_height + self.bottom_height + i * self.level_height,
                                                  height=self.door_height - self.window_stick_thickness)
                        self.construct_model.insert_element(gen_element)
        else:
            print(f"---WARNING---检测到一个未闭合的墙体图形, 类型为：{element.geometry.geometryType()}, 请检查！")

    def __construct_railing(self, element: DataElement):
        # 重建扶手模型
        if isinstance(element.geometry, Polygon):
            for i in range(self.floor_num):
                # 生成扶手墙
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Generate_wall",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height,
                                          height=self.railing_wall_height)
                self.construct_model.insert_element(gen_element)
                # 生成扶手板
                gen_element = DataElement(geometry=element.geometry,
                                          layer="Generate_railing_model",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.railing_wall_height + self.railing_model_height,
                                          height=0.05)
                self.construct_model.insert_element(gen_element)

                # 生成若干扶手玻璃
                railing_polygon = scale_rectangle_inplace(element.geometry,
                                                          scaled_lenth=get_extreme_edge(element.geometry, max_edge=True).length,
                                                          scaled_width=get_extreme_edge(element.geometry, max_edge=False).length/4)
                gen_element = DataElement(geometry=railing_polygon,
                                          layer="Generate_railing_glass",
                                          start_height=self.start_height + self.bottom_height + i * self.level_height + self.railing_wall_height,
                                          height=self.railing_model_height)
                self.construct_model.insert_element(gen_element)
        else:
            print(f"---WARNING---检测到一个未闭合的墙体图形, 类型为：{element.geometry.geometryType()}, 请检查！")

