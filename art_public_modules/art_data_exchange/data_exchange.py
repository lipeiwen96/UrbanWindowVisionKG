"""
用户输入与输出数据
"""
import uuid
from datetime import datetime
from typing import Dict

import wget as wget

from art_public_modules.art_data_exchange.dxf_module.dxf_file_reader import DxfFileReader
from art_public_modules.art_data_exchange.dxf_module.dxf_file_writer import DxfFileWriter
from art_public_modules.art_data_exchange.frontend_module.front_end_data_exchange import FrontEndDataExchanger
from art_public_modules.art_data_exchange.frontend_module.front_end_data_exchange_hk import HKFrontEndDataExchanger
from art_public_modules.art_data_exchange.json_module.hk_json_model_reader import HKJsonFileReader
from art_public_modules.art_data_exchange.json_module.hk_json_model_writer import HKJsonFileWriter
from art_public_modules.art_data_exchange.json_module.json_model_reader import JsonFileReader
from art_public_modules.art_data_exchange.json_module.json_model_writer import JsonFileWriter
from art_public_modules.art_data_exchange.rhino_module.rhino_file_reader import RhinoFileReader
from art_public_modules.art_data_exchange.rhino_module.rhino_file_writer import RhinoFileWriter
from art_public_modules.art_data_structure.shapely.core_structure import DataModel
import os


class ARTShapelyDataExchanger:
    """
    用于ART-shapely数据结构与CAD、Rhino、Json、前端之间的格式转换
    """

    @staticmethod
    def read_rhino_file(file_path: str) -> DataModel:
        return RhinoFileReader(file_path=file_path).read_3dm_file()

    @staticmethod
    def write_rhino_file(file_path: str, data_model: DataModel):
        RhinoFileWriter(file_path=file_path).write_3dm_file(data_model=data_model)

    @staticmethod
    def read_json_file(file_path: str) -> DataModel:
        return JsonFileReader().read_json_file(file_path=file_path)

    @staticmethod
    def write_json_file(file_path: str, data_model: DataModel):
        JsonFileWriter(file_path=file_path).write_json_file(data_model=data_model, file_path=file_path)

    @staticmethod
    def read_hk_json_file(file_path: str) -> DataModel:
        return HKJsonFileReader().read_json_file(file_path=file_path)

    @staticmethod
    def write_hk_json_file(file_path: str, data_model: DataModel):
        HKJsonFileWriter(file_path=file_path).write_json_file(data_model=data_model, file_path=file_path)

    @staticmethod
    def to_front_end(data_model: DataModel, move_model_to_origin: bool = False) -> Dict:
        return FrontEndDataExchanger().transform_to_front_end(data_model=data_model, move_model_to_origin=move_model_to_origin)

    @staticmethod
    def to_hk_front_end(data_model: DataModel, move_model_to_origin: bool = False) -> Dict:
        return HKFrontEndDataExchanger().transform_to_front_end(data_model=data_model, move_model_to_origin=move_model_to_origin)

    @staticmethod
    def hk_frontend_to_data_model(data_model_dict: Dict) -> DataModel:
        return HKFrontEndDataExchanger().reassemble_data_model(data_model_dict=data_model_dict)

    @staticmethod
    def frontend_to_data_model(data_model_dict: Dict) -> DataModel:
        return FrontEndDataExchanger().reassemble_data_model(data_model_dict=data_model_dict)

    @staticmethod
    def read_dxf_file(file_path: str) -> DataModel:
        if file_path.startswith('http'):
            print("使用DXF解析器讀取雲端DXF數據中...")
            base_path = os.path.abspath(os.path.join("files", "users_upload_files", "upload_model"))
            download_path = os.path.join(base_path, datetime.now().strftime("%Y_%m%d_%H-%M-%S") + str(uuid.uuid4()) + ".dxf")
            temp_url = wget.download(file_path, download_path)
            output = DxfFileReader(file_path=temp_url).read_dxf_file()
            print("雲端DXF數據讀取成功")
            # 銷毀文件
            if os.path.exists(download_path):
                # 删除文件
                os.remove(download_path)
                print(f"臨時存儲文件{download_path}已删除，避免占用空間")
            return output
        else:
            temp_url = file_path
            return DxfFileReader(file_path=temp_url).read_dxf_file()

    @staticmethod
    def write_dxf_file(file_path: str, data_model: DataModel):
        DxfFileWriter(file_path=file_path, file_name=data_model.name).write_dxf_file(data_model=data_model)
