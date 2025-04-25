"""
李沛文 | ART | XKOOL
2021.05.06

[ART TOOLS]
ConvexHUllRegion
将任意的凸多边形转变为(state, k, b, d)表示的区间表达式
其中state表示是否存在斜率，y = kx + b， d表示边界对应的方向
"""
from shapely import wkt
from shapely.geometry import Polygon, Point
from dataclasses import field, dataclass
from typing import List


@dataclass(order=True, unsafe_hash=True)
class ConvexHUllRegion:
    vertex_list: List[Point] = field(default_factory=list)
    region_expression: List[tuple] = field(default_factory=list)

    def __line_expression(self, pt1: Point, pt2: Point, pt3: Point) -> tuple:
        """
        将输入的Pt1与Pt2转化为线性表达式，并使用pt3与线性表达式的位置关系判断边界的方向
        并返回(state, k, b, d)，其中state表示是否存在斜率，y = kx + b， d表示边界对应的方向
        d: True表示区域值大于线性表达式的值（即左下边界）
           False表示区域值小于线性表达式的值（即右上边界）
        """
        (state, k, b) = self.__line_equation(pt1, pt2)
        # 方向d
        d = False

        if state:
            # 存在斜率, y = kx + b形式
            if pt3.y > k * pt3.x + b:
                d = True
        else:
            # 不存在斜率, x = b形式
            if pt3.x > b:
                d = True

        return (state, k, b, d)

    # 两点转化为直线方程
    def __line_equation(self, pt1: Point, pt2: Point) -> tuple:
        # 返回(state, k , b)
        #  若两点x坐标相等，则不存在斜率，state = 0, 表达式为 x = b
        if pt1.x == pt2.x:
            state = 0
            k = 0
            b = pt1.x
        #  若两点y坐标相等，存在斜率，state = 1, 表达式为 y = b
        elif pt1.y == pt2.y:
            state = 1
            k = 0
            b = pt1.y
        #  其余情况，均存在斜率，state = 1, 表达式为 y = kx + b
        else:
            state = 1
            k = (pt2.y - pt1.y) / (pt2.x - pt1.x)
            b = pt1.y - pt1.x * k

        return (state, k, b)

    # 将输入的多边形转化为一组线性表达式
    def points_to_region_expression(self, pt1: Point, pt2: Point, pt3: Point):
        self.region_expression.append(self.__line_expression(pt1, pt2, pt3))

    def create_polygon_region(self, input_convex_hull: Polygon):
        # 计算输入多边形的所有顶点（首尾不重复）
        self.vertex_list = [Point(coord) for coord in input_convex_hull.boundary.coords][:-1]
        vertex_num = len(self.vertex_list)

        if vertex_num == 3:
            self.points_to_region_expression(self.vertex_list[0], self.vertex_list[1], self.vertex_list[2])
            self.points_to_region_expression(self.vertex_list[1], self.vertex_list[2], self.vertex_list[0])
            self.points_to_region_expression(self.vertex_list[2], self.vertex_list[0], self.vertex_list[1])

        else:
            for i in range(0, vertex_num - 2):
                self.points_to_region_expression(self.vertex_list[i], self.vertex_list[i + 1], self.vertex_list[i + 2])
            self.points_to_region_expression(self.vertex_list[-2], self.vertex_list[-1], self.vertex_list[0])
            self.points_to_region_expression(self.vertex_list[-1], self.vertex_list[0], self.vertex_list[1])

        return self.region_expression


if __name__ == "__main__":
    # 按照wkt的方式可以快速读入polygon
    input_polygon = wkt.loads("POLYGON ((-220 550, 680 550, 680 120, -220 120, -220 550))")

    polygon_region = ConvexHUllRegion()
    region_expression =polygon_region.create_polygon_region(input_polygon.convex_hull)

    print(region_expression)