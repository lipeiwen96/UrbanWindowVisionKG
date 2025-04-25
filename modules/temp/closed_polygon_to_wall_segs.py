import shapely.geometry.polygon
from shapely.geometry import Polygon, LineString, Point
from shapely import wkt
from shapely import geometry


def closed_polygon_to_wall_segs(wall_outline_raw:Polygon,  wall_thickness:float):
    """
    每条线段生成一段墙体
    """
    wall_outline = shapely.geometry.polygon.orient(wall_outline_raw, sign=-1.0)
    lines = []
    # 将多边形的点转换为线段
    for i in range(len(wall_outline.exterior.coords) - 1):
        lines.append([wall_outline.exterior.coords[i][0],
                      wall_outline.exterior.coords[i][1],
                      wall_outline.exterior.coords[i + 1][0],
                      wall_outline.exterior.coords[i + 1][1]])
    res = []
    for line in lines:
        # line的第二个点应缩短一个墙体的厚度
        # 计算向量
        pt_1 = [line[0], line[1]]
        vector = [line[2]- line[0], line[3] - line[1]]
        # 计算向量长度 = 线的原始长度 - 墙体厚度
        vec_length  = get_vector_length(vector) - abs(wall_thickness)
        # 单位化向量
        vec_n = unitize(vector)
        # 重新计算线的第二个点
        pt_2 = [line[0] + vec_n[0] * vec_length,
                line[1] + vec_n[1] * vec_length]
        # 重新生成线
        line = LineString([pt_1, pt_2])
        # 生成墙体
        wall_segment = line.buffer(-abs(wall_thickness), single_sided=True, cap_style=2, join_style=2)
        res.append(wall_segment)
    return res


def closed_polygon_to_2_walls(wall_outline_raw:Polygon,  wall_thickness:float):
    """
    将多段线打断为两段，分别生成墙体
    """

    # 将多段线打断为两段，分别生成墙体
    # 切换方向
    wall_outline = shapely.geometry.polygon.orient(wall_outline_raw, sign=-1.0)
    # 找到第一条边的中点
    pt1 = get_midpoint(wall_outline.exterior.coords[0], wall_outline.exterior.coords[1])
    #找到中间一条边的中点
    mid_index = int(len(wall_outline.exterior.coords)/2)
    pt2 = get_midpoint(wall_outline.exterior.coords[mid_index], wall_outline.exterior.coords[mid_index + 1])
    line_1 = []
    for i in range(mid_index+1):
        if i == 0:
            line_1.append(pt1)
        else:
            line_1.append(wall_outline.exterior.coords[i])
    line_1.append(pt2)
    line_1_ls = LineString(line_1)
    wall_1 = line_1_ls.buffer(-abs(wall_thickness), single_sided=True, cap_style=2, join_style=2)
    line_2 = []
    line_2.append(pt2)
    for i in range(mid_index+1, len(wall_outline.exterior.coords)):
        line_2.append(wall_outline.exterior.coords[i])
    line_2.append(pt1)
    line_2_ls = LineString(line_2)
    wall_2 = line_2_ls.buffer(-abs(wall_thickness), single_sided=True, cap_style=2, join_style=2)
    return [wall_1, wall_2]


def get_midpoint(pt1:list, pt2:list):
    """
    计算两点的中点
    """
    return [(pt1[0] + pt2[0])/2, (pt1[1] + pt2[1])/2]


def get_line_length(line:list):
    """
    计算线的长度
    """
    return ((line[0] - line[2])**2 + (line[1] - line[3])**2)**0.5


def get_vector_length(vec:list):
    """
    计算向量长度
    """
    return (vec[0]**2 + vec[1]**2)**0.5


def unitize(vec:list):
    """
    单位化向量
    """
    length = get_vector_length(vec)
    return [vec[0]/length, vec[1]/length]


if __name__ == "__main__":
    test = wkt.loads("POLYGON ((600 650, 264 366, 860 230, 1300 290, 1304 300, 1380 640, 1090 680, 730 460, 555 483, 600 650))")
    wall = closed_polygon_to_2_walls(test, 5)
    for w in wall:
        print(w)

