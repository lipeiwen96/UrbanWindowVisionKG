import os
import shutil
from dataclasses import dataclass, field
from typing import AnyStr, List, Dict


@dataclass(order=True)
class PathUtils:
    @staticmethod
    def check_and_create_dir(file_dir: str, whether_clean: bool = False):
        """
        检查指定路径是否存在，若不存在则创建；若存在，支持直接清空
        """
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        else:
            if whether_clean:
                PathUtils.clean_all_content_in_dir(file_dir)
            else:
                pass

    @staticmethod
    def clean_all_files_in_dir(file_dir: str):
        """
        清空指定路径下所有的文件(不包含文件夹)
        """
        if not os.path.exists(file_dir):
            return
        for i in os.listdir(file_dir):
            path_file = os.path.join(file_dir, i)
            if os.path.isfile(path_file):  # 如果该内容为文件
                os.remove(path_file)
            elif os.path.isdir(path_file):
                PathUtils.clean_all_files_in_dir(path_file)

    @staticmethod
    def clean_all_content_in_dir(file_dir: str):
        """
        清空指定路径下所有的文件及文件夹
        """
        for i in os.listdir(file_dir):
            path_file = os.path.join(file_dir, i)
            if os.path.isfile(path_file):  # 如果该内容为文件
                os.remove(path_file)
            elif os.path.isdir(path_file):
                shutil.rmtree(path_file, True)
            else:
                pass

    @staticmethod
    def delete_file_if_exists(file_path):
        if os.path.isfile(file_path):
            # 如果是文件，则删除文件
            os.remove(file_path)
            print(f"文件:{file_path}删除成功！")
        elif os.path.isdir(file_path):
            # 如果是文件夹，则删除文件夹及其内容
            shutil.rmtree(file_path, True)
            print(f"文件夹:{file_path}删除成功！")
        else:
            print(f"File {file_path} does not exist.")
