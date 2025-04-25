from dataclasses import dataclass, field
from typing import AnyStr, List, Dict
import math
from shapely.geometry import Polygon, Point, LineString, MultiPoint, MultiPolygon, GeometryCollection, box
from shapely.ops import orient
from shapely.ops import nearest_points
from shapely import wkt
from shapely.ops import unary_union


def get_window_rectangle(windows_kitchen_office: List[LineString], windows_habitat: List[LineString],
                            building_outline: Polygon, building_height: float):
    if len(windows_kitchen_office) == 0 and len(windows_habitat) == 0:
        return None

    # 绘制矩形
    windows_kitchen_office_rectangle = create_rectangle(windows_kitchen_office, building_outline, building_height, "KITCHEN_OFFICE")
    windows_habitat_rectangle = create_rectangle(windows_habitat, building_outline, building_height, "HABITAT")
    # rectangles = unary_union(windows_kitchen_office_rectangle + windows_habitat_rectangle)

    return windows_kitchen_office_rectangle, windows_habitat_rectangle


def create_rectangle(windows:List[LineString], building_outline: Polygon, building_height: float, window_type: str):
    windows_rectangle = []
    for wall in windows:
        # 获取墙体的长宽
        rectangle_long_side = building_height / 4
        if window_type == "HABITAT": rectangle_long_side = building_height * 1/3
        rectangle_short_side = 2.3
        if wall.length > 2.3: wall_short_side = wall.length


        # 获取墙体的中心点和法向量, 并创建矩形
        outside_position, outside_position_pt= find_the_wall_direction(building_outline,wall)
        if outside_position == "DOWN" or outside_position == "LEFT": rectangle_long_side = - rectangle_long_side
        if outside_position == "TOP" or outside_position == "DOWN":
            start_pt = Point(outside_position_pt[0] - rectangle_short_side / 2, outside_position_pt[1])
            end_pt = Point(outside_position_pt[0] + rectangle_short_side / 2, outside_position_pt[1])
            end_pt_2 = Point(outside_position_pt[0] + rectangle_short_side / 2,
                             outside_position_pt[1] + rectangle_long_side)
            start_pt_2 = Point(outside_position_pt[0] - rectangle_short_side / 2,
                               outside_position_pt[1] + rectangle_long_side)
            windows_rectangle.append(Polygon([start_pt, end_pt, end_pt_2, start_pt_2]))

        if outside_position == "LEFT" or outside_position == "RIGHT":
            start_pt = Point(outside_position_pt[0] , outside_position_pt[1] - rectangle_short_side / 2)
            end_pt = Point(outside_position_pt[0] , outside_position_pt[1] + rectangle_short_side / 2)
            end_pt_2 = Point(outside_position_pt[0] + rectangle_long_side,
                             outside_position_pt[1] + rectangle_short_side / 2)
            start_pt_2 = Point(outside_position_pt[0] + rectangle_long_side,
                               outside_position_pt[1] - rectangle_short_side / 2)
            windows_rectangle.append(Polygon([start_pt, end_pt, end_pt_2, start_pt_2]))

    # for i in windows_rectangle:
    #     print(i)

    return windows_rectangle


def find_the_wall_direction(contourline_polygon: Polygon, wall: LineString):
    # 墙体为横向，则 wall_horizental 为True, 否则为False
    wall_horizental = True
    outside_position = ""
    outside_position_pt = []
    ID_FONT_SIZE = 0.2
    wall_pt1 = wall.coords[0]
    wall_pt2 = wall.coords[-1]

    centroid_x = (wall_pt1[0] + wall_pt2[0]) / 2
    centroid_y = (wall_pt1[1] + wall_pt2[1]) / 2

    # 首先要判断墙是横的还是竖的
    x_dis = abs(wall_pt1[0] - wall_pt2[0])
    y_dis = abs(wall_pt1[1] - wall_pt2[1])
    if x_dis > y_dis:
        # 墙体为横向
        wall_horizental = True
    else:
        # 墙体为竖的
        wall_horizental = False

    # 再找到墙体的外部放序号的点位
    if wall_horizental:
        # 墙体为横向
        if contourline_polygon.contains(Point(centroid_x, centroid_y - ID_FONT_SIZE * 2)):
            # 墙体的上方为外部点
            outside_position = "TOP"
            outside_position_pt = [centroid_x, centroid_y + ID_FONT_SIZE * 0.6]
        else:
            # 墙体的下方为外部点
            outside_position = "DOWN"
            outside_position_pt = [centroid_x, centroid_y - ID_FONT_SIZE * 1.4]
    else:
        # 墙体为横向
        if contourline_polygon.contains(Point(centroid_x - ID_FONT_SIZE * 2, centroid_y)):
            # 墙体的右边为外部点
            outside_position = "RIGHT"
            outside_position_pt = [centroid_x + ID_FONT_SIZE * 0.8, centroid_y]
        else:
            # 墙体的左边为外部点
            outside_position = "LEFT"
            outside_position_pt = [centroid_x - ID_FONT_SIZE * 0.8, centroid_y]

    return outside_position, outside_position_pt


if __name__ == '__main__':
    # 建筑外轮廓
    building_outline = wkt.loads("POLYGON ((1.858071414075206 15.60219463611252, 1.858071414074078 12.07719463611445, 2.05807141407399 12.07719463611405, 2.05807141407399 11.5771946361143, 4.133071414074551 11.57719463611141, 4.133071414075248 12.07719463611147, 4.533071414076395 12.07719463611087, 4.533071414076395 10.17719463611182, 6.433071414077945 10.17719463611181, 6.433071414078757 10.67719463610888, 9.433072339707582 10.67719463611055, 9.43307233976707 10.17719463610861, 8.933072339709817 10.17719463610861, 8.933072339709817 6.327194636111999, 9.433072339733831 6.32719463611203, 9.433072339712409 3.627194636109081, 11.60807233971213 3.627194636109154, 11.60807233971215 4.127194636093964, 12.23307233971213 4.127194636093964, 12.23307233971498 2.877194636107999, 15.03307233971213 2.877194636108309, 15.03307233971397 8.952194636099328, 16.58307233971227 8.95219463609924, 16.58307233971846 2.877194636105144, 19.38307233970958 2.877194636107999, 19.38307233974398 4.12719463609975, 20.0080723397126 4.127194636109198, 20.00807233971243 3.627194636109154, 22.18307233971232 3.627194636109081, 22.18307233972081 6.327194636111874, 22.68307233971608 6.327194636111999, 22.68307233971608 10.17719463610861, 22.18307233971598 10.17719463610861, 22.18307233971535 10.67719463606877, 25.18307141407768 10.67719463606877, 25.18307141407768 10.17719463610881, 27.08307326535063 10.17719463611182, 27.08307326535364 12.07719463611206, 27.48307141408038 12.0771946361114, 27.48307141408107 11.57719463611147, 29.55807326535127 11.57719463611147, 29.55807141408163 12.07719463611405, 29.75807141408514 12.07719463611432, 29.75807141407917 15.60219463611511, 24.78307326535171 15.60219463610832, 24.78307326535171 16.85219463610994, 20.35807141408499 16.85219463607514, 20.3580714140859 17.77719463609427, 11.2580714140758 17.77719463613582, 11.2580714140773 16.85219463610829, 6.833071414072788 16.85219463610994, 6.833071414072788 15.60219463611511, 1.858071414075206 15.60219463611252))")
    # 建筑高度
    building_height = 100
    # 厨房办公窗户
    data_kitchen = '''LINESTRING (9.533072339712408 3.677194636109079, 11.50807233971213 3.677194636109099)
    LINESTRING (9.48307233971515 3.727194636109078, 9.483072339714649 4.627194636109041)
    LINESTRING (12.88307141407323 4.189694636107264, 14.80807233971228 4.18969463610703)'''
    lines_kitchen = data_kitchen.strip().split('\n')
    windows_kitchen_office = [wkt.loads(line) for line in lines_kitchen]

    data_habitat = '''LINESTRING (15.03307233971397 9.014694636099282, 16.58307233971223 9.014694636100117)
    LINESTRING (12.6520583584225 17.55219463607492, 14.9520583584225 17.55219463607016)
    LINESTRING (16.67706020969185 17.55219463607492, 18.97706020969185 17.55219463607016)'''
    lines_habitat = data_habitat.strip().split('\n')
    windows_habitat = [wkt.loads(line) for line in lines_habitat]

    get_window_rectangle(windows_kitchen_office, windows_habitat, building_outline, building_height)

    # poly1 = wkt.loads("POLYGON ((520 350, 640 350, 640 230, 520 230, 520 350))")
    # poly2 = wkt.loads("POLYGON ((450 420, 560 420, 560 310, 450 310, 450 420))")
    #
    # print(poly1.intersects(poly2))





