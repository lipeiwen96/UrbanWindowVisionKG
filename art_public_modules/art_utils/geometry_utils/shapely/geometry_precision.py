"""
几何物件的精度处理
"""
from typing import List
from shapely.geometry import Polygon, LineString, Point


class GeometryCorrection:
    @staticmethod
    def correct_precision(geometry_list: list, tolerance=2) -> List[Polygon or LineString or Point]:
        """
        【预处理】把列表中所有物件的精度调整到小数点后的某个位数
        """
        return [GeometryCorrection.correct_one_precision(geometry, tolerance) for geometry in geometry_list]

    @staticmethod
    def correct_one_precision(geometry: Polygon or LineString or Point, tolerance) -> Polygon or LineString or Point:
        """
        【预处理】把列表中所有物件的精度调整到小数点后的某个位数
        """
        tolerance = str('.{}f'.format(tolerance))
        if isinstance(geometry, Polygon):
            geometry = Polygon([(float(format(coord[0], tolerance)), float(format(coord[1], tolerance))) for coord in list(geometry.exterior.coords)])
        if isinstance(geometry, LineString):
            geometry = LineString([(float(format(coord[0], tolerance)), float(format(coord[1], tolerance))) for coord in list(geometry.coords)])
        if isinstance(geometry, Point):
            geometry = Point((float(format(geometry.x, tolerance)), float(format(geometry.y, tolerance))))
        return geometry
