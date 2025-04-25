# 读取模式0为直接自动转换成art的数据结构，1为把所有信息原汁原味写成包含若干个字典的list
READ_OPTION = 0
# 用于匹配高度值
HEIGHT_PATTERN = r'[H][=][0-9]*.[0-9]*(?=[m])'
# 用于匹配楼层数
FLOOR_PATTERN = r'[0-9]*[F]'
# 暂时可以转换的3dm数据
READABLE_OBJ = ["LWPOLYLINE", "POLYLINE", "LINE", "CIRCLE", "ARC", "SPLINE", "TEXT", "MTEXT"]

# CAD图层颜色匹配表
DXF_LAYER_COLOR_DICT = {
    "Xkool_ProjectBoundary": 252,  # 深灰
    "Xkool_PlotBoundary": 252,  # 深灰
    "Xkool_SurroundBuilding": 252,  # 深灰
    "Xkool_TargetSiteBoundary": 6,  # 洋红
    "Xkool_SiteUnbuildableRegion": 82,  # 绿
    "Xkool_BuildingMinSeperation": 14,  # 红线
    "Xkool_HabitatRHP_Internal": 130,
    "Xkool_HabitatRHP_4.5m+Street": 130,
    "Xkool_HabitatRHP_ForReference": 136,
    "Xkool_OtherRHP_Internal": 30,
    "Xkool_OtherRHP_4.5m+Street": 30,
    "Xkool_OtherRHP_ForReference": 36,
}

# CAD文字样式
NAME_TEXT_HEIGHT = 2
TITLE_TEXT_HEIGHT = 14