"""
各种坐标的转化工具
[国际标准]谷歌地球/GPS坐标系使用-WGS84
[中国标准]GoogleMap/高德/腾讯使用-GCJ02
百度地图坐标系-BD09: 其在GCJ-02上多增加了一次变换，用来保护用户隐私
----------------------------------------------------
开发者-李沛文
2021-11-04
参考链接：https://my.oschina.net/u/3771868/blog/1820943

[修改]-----------------------------------------------
导入pyproj库
增加香港1980坐标系，支持其与WGS84的相互转化
2023-05-30
"""
import numpy as np
import math
import pyproj


class CoordConvertor:
    @ staticmethod
    def gcj_to_wgs(gcj_lat, gcj_lon):
        """
        将输入的GCJ-02火星坐标系的坐标点转化为WGS-84通用坐标系
        :param gcj_lat: GCJ-02火星坐标系-纬度
        :param gcj_lon: GCJ-02火星坐标系-经度
        :return: WGS-84通用坐标系的经纬度
        """
        a = 6378245.0
        ee = 0.00669342162296594323
        pi = 3.14159265358979324
        x = np.float64(gcj_lon) - 105.0
        y = np.float64(gcj_lat) - 35.0

        # 转化经度
        d_lon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        d_lon += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        d_lon += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
        d_lon += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0

        # 转化纬度
        d_lat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        d_lat += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        d_lat += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
        d_lat += (160.0 * math.sin(y / 12.0 * pi) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
        rad_lat = np.float64(gcj_lat) / 180.0 * pi
        magic = math.sin(rad_lat)
        magic = 1 - ee * magic * magic
        sqrt_magic = math.sqrt(magic)
        d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * pi)
        d_lon = (d_lon * 180.0) / (a / sqrt_magic * math.cos(rad_lat) * pi)

        wgs_lon = np.float64(gcj_lon) - d_lon
        wgs_lat = np.float64(gcj_lat) - d_lat
        return [wgs_lat, wgs_lon]

    @staticmethod
    def wgs_to_gcj(wgs_lat, wgs_lon):
        """
        将输入的WGS-84通用坐标系的坐标点转化为GCJ-02火星坐标系
        :param wgs_lat: WGS-84通用坐标系-纬度
        :param wgs_lon: WGS-84通用坐标系-经度
        :return: GCJ-02通用坐标系的经纬度
        """
        a = 6378245.0
        ee = 0.00669342162296594323
        pi = 3.14159265358979324
        x = np.float64(wgs_lon) - 105.0
        y = np.float64(wgs_lat) - 35.0

        # 转化经度
        d_lon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        d_lon += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        d_lon += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
        d_lon += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0

        # 转化纬度
        d_lat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        d_lat += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        d_lat += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
        d_lat += (160.0 * math.sin(y / 12.0 * pi) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0

        rad_lat = wgs_lat / 180.0 * pi
        magic = math.sin(rad_lat)
        magic = 1 - ee * magic * magic
        sqrt_magic = math.sqrt(magic)
        d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * pi)
        d_lon = (d_lon * 180.0) / (a / sqrt_magic * math.cos(rad_lat) * pi)

        gcj_lat = np.float64(wgs_lat) + d_lat
        gcj_lon = np.float64(wgs_lon) + d_lon
        return [gcj_lat, gcj_lon]

    @staticmethod
    def gcj_to_bd(gcj_lat, gcj_lon):
        """
        将输入的GCJ-02火星坐标系的坐标点转化为BD-09百度坐标系
        :param gcj_lat: GCJ-02火星坐标系-纬度
        :param gcj_lon: GCJ-02火星坐标系-经度
        :return: BD-09百度坐标系
        """
        x_pi = 3.14159265358979324 * 3000.0 / 180.0

        x = np.float64(gcj_lon)
        y = np.float64(gcj_lat)
        z = math.sqrt(x * x + y * y) + 0.00002 * math.sin(y * x_pi)

        theta = math.atan2(y, x) + 0.000003 * math.cos(x * x_pi)
        bd_lon = z * math.cos(theta) + 0.0065
        bd_lat = z * math.sin(theta) + 0.006

        return [bd_lon, bd_lat]

    @staticmethod
    def bd_to_gcj(bd_lon, bd_lat):
        """
        将输入的BD-09百度坐标系的坐标点转化为GCJ-02火星坐标系
        :param bd_lon: BD-09百度坐标系-经度
        :param bd_lat: BD-09百度坐标系-纬度
        :return: GCJ-02火星坐标系
        """
        x_pi = 3.14159265358979324 * 3000.0 / 180.0

        x = bd_lon - 0.0065
        y = bd_lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)

        gcj_lon = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)
        return [gcj_lat, gcj_lon]

    @staticmethod
    def hk_to_wgs(hk_nor, hk_eas):
        """
        将输入的香港1980坐标系的坐标点转化为WGS-84通用坐标系
        :param hk_nor: 香港1980坐标系-Northing(m), 如826033
        :param hk_eas: GCJ-02火星坐标系-Easting(m), 如830151
        :return: WGS-84通用坐标系的经纬度
        """
        transformer_hk80_to_wgs84 = pyproj.Transformer.from_crs(pyproj.CRS.from_string('EPSG:2326'), pyproj.CRS.from_string('EPSG:4326'),
                                                                always_xy=True)
        wgs_lon, wgs_lat = transformer_hk80_to_wgs84.transform(hk_eas, hk_nor)
        return [wgs_lat, wgs_lon]

    @staticmethod
    def wgs_to_hk(wgs_lat, wgs_lon):
        """
        将输入的WGS-84通用坐标系的坐标点转化为香港1980坐标系
        :param wgs_lat: WGS-84通用坐标系-纬度
        :param wgs_lon: WGS-84通用坐标系-经度
        :return: 香港1980坐标系的Northing(m)和Easting(m)
        """
        transformer_wgs84_to_hk80 = pyproj.Transformer.from_crs(pyproj.CRS.from_string('EPSG:4326'), pyproj.CRS.from_string('EPSG:2326'),
                                                                always_xy=True)
        hk_eas, hk_nor = transformer_wgs84_to_hk80.transform(wgs_lon, wgs_lat)
        return [hk_nor, hk_eas]

    @staticmethod
    def wgs_to_hk_reverse(wgs_lon, wgs_lat, wgs_type: str = "4326"):
        transformer_wgs84_to_hk80 = pyproj.Transformer.from_crs(pyproj.CRS.from_string(f'EPSG:{wgs_type}'),
                                                                pyproj.CRS.from_string('EPSG:2326'),
                                                                always_xy=True)
        hk_eas, hk_nor = transformer_wgs84_to_hk80.transform(wgs_lon, wgs_lat)
        return [hk_eas, hk_nor]


if __name__ == '__main__':
    # 例如谷歌地图中 哈工深荔园餐厅的柱状体块中心点为：22.586654665264128, 113.96895058422614
    gcj_coord = [22.586654665264128, 113.96895058422614]
    wgs_coord = CoordConvertor.gcj_to_wgs(gcj_coord[0], gcj_coord[1])
    print(wgs_coord)

    gcj_coord_by_wgs = CoordConvertor.wgs_to_gcj(wgs_coord[0], wgs_coord[1])
    print(gcj_coord_by_wgs)

    bd_coord = CoordConvertor.gcj_to_bd(gcj_coord[0], gcj_coord[1])
    print(bd_coord)

    gcj_coord_by_bd = CoordConvertor.bd_to_gcj(bd_coord[0], bd_coord[1])
    print(gcj_coord_by_bd)

    # 测试，以香港国际天粮事工有限公司为例
    wgs_lat, wgs_lon = [22.460663017907912, 114.00404138929345]
    hk_nos, hk_eas = CoordConvertor.wgs_to_hk(wgs_lat, wgs_lon)
    print("香港1980坐标：", hk_nos, hk_eas)

    hk_nos, hk_eas = [835547, 818731]
    wgs_lat, wgs_lon = CoordConvertor.hk_to_wgs(hk_nos, hk_eas)
    print("WGS84坐标：", wgs_lat, wgs_lon)

