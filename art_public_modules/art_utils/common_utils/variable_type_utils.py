import os
import string
from typing import List, Tuple


class ListUtils:
    @staticmethod
    def flatten_list(input_list: List) -> List:
        """
        递归拍平list
        """
        output_list = []
        for _ in input_list:
            if isinstance(_, list):
                output_list += ListUtils.flatten_list(input_list=_)
            else:
                output_list.append(_)
        return output_list

    @staticmethod
    def make_list_right(input_list: List) -> List:
        """
        检查list，过滤掉其中的None或空值
        """
        input_list = list(filter(lambda each: each is not None, input_list))
        return input_list


# 【DXF文字特征】
# DXF文件中,字符串内各种类型的字符对应的长度（以字符大小1为单位）
# DXF英文字符的单位长度
DXF_EN_LENGTH = 0.73
# DXF数字字符的单位长度
DXF_DG_LENGTH = 0.73
# DXF空格字符的单位长度
DXF_SP_LENGTH = 0.73
# DXF中文字符的单位长度
DXF_ZH_LENGTH = 1.44
# DXF特殊字符的单位长度
DXF_PU_LENGTH = 0.73

# 【Matplotlib导出文字特征】
# Matplotlib导出文字中,字符串内各种类型的字符对应的长度（以字符大小1为单位）
# TEXT英文字符的单位长度
TEXT_EN_LENGTH = 0.19
# TEXT数字字符的单位长度
TEXT_DG_LENGTH = 0.19
# TEXT空格字符的单位长度
TEXT_SP_LENGTH = 0.19
# TEXT中文字符的单位长度
TEXT_ZH_LENGTH = 0.26
# TEXT特殊字符的单位长度
TEXT_PU_LENGTH = 0.19


class StringUtils:
    """
    【字符串操作函数】
    * string_length: 用于计算任意一串字符串的各种类型字符的数量
    * dxf_string_length: 用于计算任意一串字符串在各种文件中的实际长度
    @建筑研究 李沛文
    """

    @staticmethod
    def dxf_string_length(input_string: str, string_size: float):
        """
        计算DXF文件中任意一串字符串对应的估算长度
        """
        count_en, count_dg, count_sp, count_zh, count_pu = StringUtils.string_length(input_string)
        total_length = count_en * DXF_EN_LENGTH + count_dg * DXF_DG_LENGTH + count_sp * DXF_SP_LENGTH + \
                       count_zh * DXF_ZH_LENGTH + count_pu * DXF_PU_LENGTH
        return round(total_length * string_size, 1)

    @staticmethod
    def matplotlib_string_length(input_string: str, string_size: float):
        """
        计算matplotlib导出图片中任意一串字符串对应的估算长度
        """
        count_en, count_dg, count_sp, count_zh, count_pu = StringUtils.string_length(input_string)
        total_length = count_en * TEXT_EN_LENGTH + count_dg * TEXT_DG_LENGTH + count_sp * TEXT_SP_LENGTH + \
                       count_zh * TEXT_ZH_LENGTH + count_pu * TEXT_PU_LENGTH
        return round(total_length * string_size, 1)

    @staticmethod
    def string_length(input_string: str):
        """
        计算任意的输入的字符串中，英文字符/数字字符/中文字符/空格/特殊字符的个数,并返回；
        """
        count_en = 0  # 英文字符数量
        count_dg = 0  # 数字字符数量
        count_sp = 0  # 空格字符数量
        count_zh = 0  # 中文字符数量
        count_pu = 0  # 特殊字符数量
        for s in input_string:
            # 英文
            if s in string.ascii_letters:
                count_en += 1
            # 数字
            elif s.isdigit():
                count_dg += 1
            # 空格
            elif s.isspace():
                count_sp += 1
            # 中文
            elif s.isalpha():
                count_zh += 1
            # 特殊字符
            else:
                count_pu += 1
        # print('英文字符：', count_en)
        # print('数字：', count_dg)
        # print('空格：', count_sp)
        # print('中文：', count_zh)
        # print('特殊字符：', count_pu)
        return count_en, count_dg, count_sp, count_zh, count_pu


if __name__ == "__main__":
    a = [[1, 2, 3, [12, 56], [5, 6, 8]], [123], [234, [234, [56]]]]  # --> [1, 2, 3, 12, 56, 5, 6, 8, 123, 234, 234, 56]
    a = ListUtils.flatten_list(input_list=a)
    print(a)

    b = [None, 1]
    print(ListUtils.make_list_right(b))

    print(StringUtils.dxf_string_length("[拼合信息: B2-轻钢1500 * 2 + B1-轻钢1000]", 3))
    print(StringUtils.dxf_string_length("建筑面积:4000平米  计容面积:8000平米", 3))
