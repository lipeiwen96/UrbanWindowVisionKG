import copy

from shapely.geometry import Polygon, Point, LineString, GeometryCollection, box, MultiPolygon, MultiLineString
from shapely.geometry.base import BaseGeometry
from typing import List
from map_system.tile_genenrator.map_structure import MapBaseGeometry


class MapCropper:
    @staticmethod
    def crop(map_geometry_list: List[MapBaseGeometry], site_boundary: Polygon, tile_box: Polygon) -> List[MapBaseGeometry]:
        map_geo_after_difference = []
        for geo in map_geometry_list:
            outcome = MapCropper.single_geometry_difference(geo, site_boundary)
            if outcome is not None:
                for out_geo in outcome:
                    map_geo_after_difference.append(out_geo)

        map_geo_after_intersection = []
        for geo in map_geo_after_difference:
            outcome = MapCropper.single_geometry_intersection(geo, tile_box)
            if outcome is not None:
                for out_geo in outcome:
                    map_geo_after_intersection.append(out_geo)
        return map_geo_after_intersection

    @staticmethod
    def filter(map_geometry_list: List[MapBaseGeometry], site_boundary: Polygon, tile_box: Polygon) -> List[MapBaseGeometry]:
        map_geo_after_filter = []
        for geo in map_geometry_list:
            if geo.geometry.intersects(site_boundary):
                pass
            else:
                map_geo_after_filter.append(geo)

        map_geo_after_intersection = []
        for geo in map_geo_after_filter:
            outcome = MapCropper.single_geometry_intersection(geo, tile_box)
            if outcome is not None:
                for out_geo in outcome:
                    map_geo_after_intersection.append(out_geo)
        return map_geo_after_intersection

    @staticmethod
    def single_geometry_difference(single_map_geometry: MapBaseGeometry, difference_polygon: Polygon):
        outcome = single_map_geometry.geometry.difference(difference_polygon)
        processed_outcome = MapCropper.outcome_process(outcome)
        if processed_outcome is not None:
            outcome_map_geo_list = []
            for single_geometry in processed_outcome:
                map_geo = copy.deepcopy(single_map_geometry)
                map_geo.geometry = single_geometry
                outcome_map_geo_list.append(map_geo)
            return outcome_map_geo_list
        else:
            return None

    @staticmethod
    def single_geometry_intersection(single_map_geometry: MapBaseGeometry, intersection_polygon: Polygon):
        if single_map_geometry.geometry.intersects(intersection_polygon):
            outcome = single_map_geometry.geometry.intersection(intersection_polygon)
            processed_outcome = MapCropper.outcome_process(outcome)
            if processed_outcome is not None:
                outcome_map_geo_list = []
                for single_geometry in processed_outcome:
                    map_geo = copy.deepcopy(single_map_geometry)
                    map_geo.geometry = single_geometry
                    outcome_map_geo_list.append(map_geo)
                return outcome_map_geo_list
            else:
                return None
        else:
            return None

    @staticmethod
    def outcome_process(outcome):
        if outcome.is_empty:
            return None
        elif isinstance(outcome, Polygon) or isinstance(outcome, LineString):
            # 单图形结果，不做处理，直接输出
            return [outcome]
        elif isinstance(outcome, MultiLineString) or isinstance(outcome, MultiPolygon) or isinstance(outcome, GeometryCollection):
            geo_list = []
            for single_geometry in outcome:
                if not single_geometry.is_empty:
                    geo_list.append(single_geometry)
            return geo_list
