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
    "Xkool_ProjectBoundary": (255, 255, 255, 255),  # 深灰
    "Xkool_PlotBoundary": (105, 105, 105, 255),  # 深灰
    "Xkool_SurroundBuilding": (105, 105, 105, 255),  # 深灰
    "Xkool_TargetSiteBoundary": (255, 0, 255, 255),  # 洋红
    "Xkool_SiteUnbuildableRegion": (0, 127, 0, 255),  # 深灰
    "Xkool_BuildingMinSeperation": (255, 0, 0, 255),  # 红线
    "Xkool_HabitatRHP_Internal": (0, 255, 255, 255),  # 建筑2：青色
    "Xkool_HabitatRHP_4.5m+Street": (0, 255, 255, 255),  # 建筑2：青色
    "Xkool_HabitatRHP_ForReference": (0, 255, 255, 50),  # 建筑2：青色
    "Xkool_OtherRHP_Internal": (255, 127, 0, 255),
    "Xkool_OtherRHP_4.5m+Street": (255, 127, 0, 255),
    "Xkool_OtherRHP_ForReference": (255, 127, 0, 50),
    "Xkool_BuildingDOutline": (255, 255, 255, 255),
    "Xkool_BuildingNDOutline": (255, 255, 255, 255),
    "Xkool_BuildingFloorLine": (0, 0, 0, 255),
}