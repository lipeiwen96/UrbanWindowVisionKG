# 读取模式0为直接自动转换成art的数据结构，1为把所有信息原汁原味写成包含若干个字典的list
READ_OPTION = 0
# 用于匹配高度值
HEIGHT_PATTERN = r'[H][=][0-9]*.[0-9]*(?=[m])'
# 用于匹配楼层数
FLOOR_PATTERN = r'[0-9]*[F]'
# 暂时可以转换的3dm数据
READABLE_OBJ = ['ObjectType.Curve', 'ObjectType.Point']


# 寫入的rhino版本
WRITE_VERSION = 6
# 可以转换的shapely geometry数据
WRITABLE_OBJ = ['Point', 'LineString', 'Polygon']

# Rhino图层颜色匹配表
RHINO_LAYER_COLOR_DICT = {
    "Base": (0, 138, 251, 255),  # 蓝色基座
    "Land_Boundary": (180, 180, 180, 255),  # 深灰
    "ROAD": (80, 80, 80, 255),  # 深灰
    "LOT": (220, 220, 220, 255),  # 深灰
    "GLA": (220, 220, 220, 255),  # 深灰
    "Landscape": (0, 184, 90, 255),  # 深灰
    "Building": (255, 255, 255, 255),
}