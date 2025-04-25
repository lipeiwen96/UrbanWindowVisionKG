from dataclasses import dataclass, field
from typing import AnyStr, List, Dict
from modules.input_module.BPR_constant import BPR_Schedule


class BPRCompute:
    @staticmethod
    def compute_BPR_params(site_classification: str, min_bulding_height: float, max_bulding_height: float):
        for each_schedule in BPR_Schedule:
            if min_bulding_height >= each_schedule["height"]["min_height"] and max_bulding_height <= each_schedule["height"]["max_height"]:
                DSC = each_schedule["DSC"][site_classification]
                DPR = each_schedule["DPR"][site_classification]
                NDSC = each_schedule["NDSC"][site_classification]
                NDPR = each_schedule["NDPR"][site_classification]
                return DSC, DPR, NDSC, NDPR

        # 存在最小高度最大高度跨越多个分类级别的情况
        # 这时以最高的级别来计算
        for each_schedule in BPR_Schedule:
            if each_schedule["height"]["min_height"] <= max_bulding_height <= each_schedule["height"]["max_height"]:
                DSC = each_schedule["DSC"][site_classification]
                DPR = each_schedule["DPR"][site_classification]
                NDSC = each_schedule["NDSC"][site_classification]
                NDPR = each_schedule["NDPR"][site_classification]
                return DSC, DPR, NDSC, NDPR