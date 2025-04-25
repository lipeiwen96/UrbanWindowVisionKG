import math
import numpy as np
from shapely.affinity import scale, rotate
from shapely.geometry import Point, LineString, Polygon


class ShapelyShapeUtils:
    """
    关于圆弧、曲线等形状的一些操作
    """
    @staticmethod
    def make_arc_linestring(center, radius, start_angle, end_angle, num_segments=16):
        theta = np.radians(np.linspace(start_angle, end_angle, num_segments))
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        return LineString(np.column_stack([x, y]))

    @staticmethod
    def make_ellipse(center, major_axis, ratio, start_param, end_param):
        major_length = math.sqrt((center[0] - major_axis[0]) ** 2 + (center[1] - major_axis[1]) ** 2)
        minor_length = major_length * ratio

        degrees = math.degrees(math.asin((major_axis[1] - center[1]) / major_length))

        ellipse = (center, (major_length, minor_length), degrees)

        # Let create a circle of radius 1 around center point:
        circ = Point(ellipse[0]).buffer(1)

        # Let create the ellipse along x and y:
        ell = scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))

        # Let rotate the ellipse (clockwise, x axis pointing right):
        ellv = rotate(ell, 90 - ellipse[2])

        # If one need to rotate it clockwise along an upward pointing x axis:
        # elrv = rotate(ell, 90 - ellipse[2])
        if end_param - start_param - 2 * math.pi < 0.001:
            return ellv
        else:
            raise NotImplementedError("not implemented")

    @staticmethod
    def make_arc_by_bulge(point_a, point_b, bulge, elevation=0):
        """
        根据一个bulge值，获得arc
        Borrow 小哥哥之前写的，不过只接受 (x, y, z) 的点tuple，而不是shapely的Point
        :param point_a:
        :param point_b:
        :param bulge:
        :return:
        """
        sign = np.sign(bulge)
        line = LineString([point_a, point_b])
        c = line.length
        if c == 0:
            # 两点重合
            return []
        s = abs(c / 2 * bulge)
        # 根据bulge计算圆弧的半径
        r = abs(((c / 2) ** 2 + s ** 2) / (2 * s))

        line_centroid = line.centroid
        line_center_x, line_center_y = line_centroid.x, line_centroid.y
        # 直线单位向量
        unit_vector_of_line = ((point_b[0] - point_a[0]) / c, (point_b[1] - point_a[1]) / c)
        vx, vy = unit_vector_of_line
        # 如果bulge为正，则圆心位于前进方向的左边（圆弧位于后面）， 否则相反
        vertical_unit_vector = - sign * vy, sign * vx
        dx, dy = vertical_unit_vector
        step = abs(r) - abs(s)
        dx, dy = dx * step, dy * step
        # 沿着方位向量移动n步，求得圆心
        center_x, center_y = (line_center_x + dx, line_center_y + dy)

        # 计算角度范围
        start_theta = ShapelyShapeUtils.get_angle_inside_a_circle(point_a[0] - center_x, point_a[1] - center_y)
        end_theta = ShapelyShapeUtils.get_angle_inside_a_circle(point_b[0] - center_x, point_b[1] - center_y)
        if bulge > 0:
            # bulge 为正时，逆时针，end_theta必须大于start_theta，例如start_theta=150,end_theta=-150，实际上end_theta应为210
            if end_theta < start_theta:
                end_theta += 360
        else:
            # 顺时针
            if end_theta > start_theta:
                end_theta -= 360
        # 生成角度等差数列
        theta_list = np.linspace(start_theta, end_theta, num=16)
        # 生成点的数列
        point_list = [(center_x + abs(r) * math.cos(math.radians(theta)),
                       center_y + abs(r) * math.sin(math.radians(theta)),
                       elevation) for theta in theta_list]
        # 取除了最后一个点外的所有点返回（最后一个点在下一个线段计算，以免重复计算）
        return point_list[:-1]

    @staticmethod
    def get_angle_inside_a_circle(dx, dy):
        theta = math.degrees(math.atan2(dy, dx))
        return theta

    @staticmethod
    def generate_corners_for_polygon(polygon: Polygon, turning_dis: float = 7):
        output_polygon = polygon.buffer(0)  # 消除异常的边缘
        output_polygon = output_polygon.buffer(-turning_dis)
        output_polygon = output_polygon.buffer(turning_dis, cap_style=1, join_style=1)  # 倒圆角
        return output_polygon

    # @staticmethod
    # def shapely_rectangle_to_dxf_xyb_format(box: Polygon, radius):
    #     """
    #     为输入的矩形按照输入的半径进行倒圆角操作, 并转化为dxf所需的xyb格式
    #     """
    #     # 校正精度
    #     correction_box = GeometryCorrection.correct_one_precision(box, tolerance=2)
    #     points_list = correction_box.exterior.coords
    #     points_xyb_list = []
    #
    #     # 计算膨胀值b = vertical_edge / bevel_edge
    #     bevel_edge = radius * math.sqrt(2) / 2
    #     vertical_edge = radius - bevel_edge
    #     b = vertical_edge / bevel_edge
    #
    #     # 输入矩形的各顶点
    #     point_1 = points_list[0]
    #     point_2 = points_list[1]
    #     point_3 = points_list[2]
    #     point_4 = points_list[3]
    #
    #     start_pt_1, end_pt_1 = GeometryMethods.get_edge_point(start_pt=Point(point_1), end_pt=Point(point_2), radius=radius)
    #     points_xyb_list.append([start_pt_1[0], start_pt_1[1], 0])  # xyb格式
    #     points_xyb_list.append([end_pt_1[0], end_pt_1[1], b])  # xyb格式
    #
    #     start_pt_2, end_pt_2 = GeometryMethods.get_edge_point(start_pt=Point(point_2), end_pt=Point(point_3), radius=radius)
    #     points_xyb_list.append([start_pt_2[0], start_pt_2[1], 0])  # xyb格式
    #     points_xyb_list.append([end_pt_2[0], end_pt_2[1], b])  # xyb格式
    #
    #     start_pt_3, end_pt_3 = GeometryMethods.get_edge_point(start_pt=Point(point_3), end_pt=Point(point_4), radius=radius)
    #     points_xyb_list.append([start_pt_3[0], start_pt_3[1], 0])  # xyb格式
    #     points_xyb_list.append([end_pt_3[0], end_pt_3[1], b])  # xyb格式
    #
    #     start_pt_4, end_pt_4 = GeometryMethods.get_edge_point(start_pt=Point(point_4), end_pt=Point(point_1), radius=radius)
    #     points_xyb_list.append([start_pt_4[0], start_pt_4[1], 0])  # xyb格式
    #     points_xyb_list.append([end_pt_4[0], end_pt_4[1], b])  # xyb格式
    #     points_xyb_list.append([start_pt_1[0], start_pt_1[1], 0])  # xyb格式
    #
    #     return points_xyb_list