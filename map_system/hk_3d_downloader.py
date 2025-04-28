# -*- coding: utf-8 -*-
# hk_3d_downloader.py
"""
=======================
v1.9 – 中文化输出与注释，改进地形加载信息，尝试修复 .3ds 加载
----------------------------------------------------------------
🔑 主要改动
1. **修复 .3ds 加载尝试:** 在 `trimesh.load` 中明确添加 `file_type='3ds'` 参数。
2. **中文化:** 所有用户可见的 print 输出和代码注释已更新为中文。
3. **地形加载信息增强:**
   - 加载前打印正在处理的 Tile 名称。
   - 显示正在加载的文件名（简化版或原始版）。
   - 显示文件大小 (KB)。
   - 显示加载后的 Mesh 基本信息（顶点数、面数）。
   - 显示 Mesh 的边界框 (Bounding Box)。
   - 对面数超过阈值的 Mesh 进行中文提示。
4. **移除示例打印:** 示例用法中不再打印最终的文件夹列表。
5. **依赖:** 保持不变 (`requests`, `trimesh`, `pyassimp`)。

依赖: `requests`, `trimesh`, `pyassimp` (通常通过 `pip install trimesh[easy]` 安装)。
"""
from __future__ import annotations

import concurrent.futures as _cf
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Tuple, Optional

import requests
import trimesh
import numpy as np
TRIMESH_AVAILABLE = True


class SpatialTileDownloader:
    """
    用于香港 CSDI B1000 3D Tiles 的并行下载、解压、验证和安全清理工具。
    强制使用无连字符命名，处理单/双位数名称，并包含可选的地形加载功能。
    """

    # --- 正则表达式定义 ---
    # 标准格式: 7SW15C, 11NW6C, 11NW20B (无连字符) - 允许数字部分为 1 或 2 位
    STANDARD_NAME_PAT = re.compile(r"^\d{1,2}[A-Z]{2}\d{1,2}[A-Z]$")
    # 旧格式: 7-SW-15C, 11-NW-6C, 11-NW-20B (带连字符)
    OLD_NAME_PAT = re.compile(r"^\d{1,2}-[A-Z]{2}-\d{1,2}[A-Z]$")

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    @staticmethod
    def download_b1000_tiles(
        *,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        fmt: str,
        geojson_path: os.PathLike | str,
        out_dir: os.PathLike | str,
        overwrite: bool = False,
        workers: int = 6,
        load_terrain: bool = True, # 控制是否加载地形的新标志
        terrain_face_threshold: int = 2000 # 面数警告阈值
    ) -> List[str]:
        """
        批量下载和解压 B1000 Tiles，包含验证、清理、并行处理和可选的地形网格加载。

        参数:
            min_x, min_y, max_x, max_y: 研究区域边界框坐标 (HK80 坐标系)。
            fmt: Tile 格式 ("3DS", "FBX", "MAX", "VRML")。
            geojson_path: 指向 B1000 Tile 索引 GeoJSON 文件的路径。
            out_dir: 保存下载和解压后的 Tiles 的目录。
            overwrite: 若为 True，则重新下载并覆盖已存在的完整 Tiles。
            workers: 并行下载/处理的线程数。
            load_terrain: 若为 True 且 'trimesh' 已安装，则加载地形网格。
            terrain_face_threshold: 面数超过此值时打印警告信息。

        返回:
            一个包含成功下载/验证的 Tile 文件夹路径的列表。
        """
        fmt = fmt.upper()
        if fmt not in {"3DS", "FBX", "MAX", "VRML"}:
            raise ValueError("参数 'fmt' 必须是 '3DS', 'FBX', 'MAX', 'VRML' 中的一个")

        geojson_path = Path(geojson_path).expanduser().resolve()
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        trash_dir = out_dir / ".trash" # 回收站目录
        trash_dir.mkdir(exist_ok=True)

        print("--- 正在加载 GeoJSON 数据 ---")
        try:
            with geojson_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            print(f"GeoJSON 加载成功，共包含 {len(data.get('features', []))} 个要素。")
        except Exception as e:
            print(f"错误：无法加载或解析 GeoJSON 文件 {geojson_path}: {e}")
            return []

        bbox = (min_x, min_y, max_x, max_y)
        features_all = data.get("features", [])
        print(f"\n--- 正在筛选与 BBox {bbox} 相交的 Tiles ---")
        selected_features = []
        for f in features_all:
            props = f.get("properties", {})
            coords = (
                props.get("Min_x"),
                props.get("Min_y"),
                props.get("Max_x"),
                props.get("Max_y"),
            )
            # 验证坐标是否有效
            if not all(isinstance(c, (int, float)) for c in coords):
                sheet_no_display = props.get("Sheet_No", "未知")
                # print(f"警告：跳过要素 {sheet_no_display}，因为其边界坐标无效或缺失。") # 减少冗余信息
                continue

            if SpatialTileDownloader._bbox_intersects(bbox, coords):
                # 验证 Sheet_No 是否有效
                sheet_no_original = props.get("Sheet_No")
                if not isinstance(sheet_no_original, str) or not sheet_no_original:
                    # print(f"警告：跳过一个相交的要素，因为其 Sheet_No 无效或缺失。") # 减少冗余信息
                    continue
                selected_features.append(f)

        SpatialTileDownloader._print_selection_summary(bbox, selected_features)
        total_selected = len(selected_features)
        if not total_selected:
            print("在指定的 BBox 内未找到有效的相交 Tiles。")
            return []

        # --- 修改：仅打印选中 Tiles 的几何信息 ---
        SpatialTileDownloader._print_dataset_geometry(selected_features)
        # --- 结束修改 ---

        print("\n--- 正在检查并规范化现有目录 ---")
        # 启动时检查现有目录（使用更新后的正则表达式）
        SpatialTileDownloader._normalize_and_clean(out_dir, fmt, trash_dir)
        print("--- 目录检查完成 ---")

        lock = Lock() # 用于线程安全的打印和列表追加
        successful_tile_paths: list[str] = [] # 存储成功处理的 Tile 路径
        processed_count = 0 # 跟踪已尝试处理的 Tile 数量

        print("\n--- 开始并行下载和处理 ---")
        # 使用 enumerate(selected_features, 1) 提供从 1 开始的索引
        def _process(idx_feat):
            nonlocal processed_count
            idx, feat = idx_feat
            props = feat["properties"]
            sheet_no_original: str = props["Sheet_No"] # 来自 GeoJSON，可能带连字符
            # 统一使用无连字符的标准名称
            sheet_no_std = sheet_no_original.replace("-", "")

            # **核心改动：使用更新后的正则验证标准名称格式**
            if not SpatialTileDownloader.STANDARD_NAME_PAT.match(sheet_no_std):
                with lock:
                    # 打印更详细的跳过原因
                    print(f"[{idx}/{total_selected}] {sheet_no_original} -> ✖ 跳过 (名称 '{sheet_no_std}' 不符合格式 ^\\d{{1,2}}[A-Z]{{2}}\\d{{1,2}}[A-Z]$)")
                return "" # 返回空字符串表示失败

            url = props.get(f"Format_{fmt}")
            if not url:
                 with lock:
                    print(f"[{idx}/{total_selected}] {sheet_no_std} -> ✖ 跳过 (缺少 Format_{fmt} 的 URL)")
                 return ""

            # 目标文件夹使用标准名称
            dest_folder = out_dir / sheet_no_std
            folder_result = "" # 存储成功的路径，用于后续地形加载

            # 检查目标文件夹是否存在且完整
            if dest_folder.exists():
                if SpatialTileDownloader._tile_complete(dest_folder, fmt):
                    if not overwrite:
                        with lock:
                            print(f"[{idx}/{total_selected}] {sheet_no_std} -> ↺ 跳过 (已存在且完整)")
                        # 不再在此处添加路径，在 _process 返回后统一添加
                        processed_count += 1 # 计入已处理
                        return str(dest_folder) # 返回路径，以便主流程知道它存在
                    else:
                        with lock:
                            print(f"[{idx}/{total_selected}] {sheet_no_std} -> 准备覆盖...")
                        SpatialTileDownloader._move_to_trash(dest_folder, trash_dir, prefix="overwriting")
                else:
                    with lock:
                         print(f"[{idx}/{total_selected}] {sheet_no_std} -> 存在但不完整，正在清理...")
                    SpatialTileDownloader._move_to_trash(dest_folder, trash_dir, prefix="incomplete_existing")

            # --- 开始下载和处理 ---
            status = ""
            t0 = time.time()
            tmp_path = None # 确保变量存在
            try:
                # 使用临时文件进行下载
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=out_dir) as tmp:
                    tmp_path = Path(tmp.name)
                    with lock:
                         print(f"[{idx}/{total_selected}] {sheet_no_std} -> 下载中...")
                    SpatialTileDownloader._download_file(url, tmp)

                with lock:
                    print(f"[{idx}/{total_selected}] {sheet_no_std} -> 解压中...")
                # 解压并确保使用标准命名
                SpatialTileDownloader._unzip_and_standardize(tmp_path, dest_folder, sheet_no_std, trash_dir)

                # 解压后再次检查完整性
                if not SpatialTileDownloader._tile_complete(dest_folder, fmt):
                    SpatialTileDownloader._move_to_trash(dest_folder, trash_dir, prefix="incomplete_download")
                    raise RuntimeError("下载并解压后 Tile 仍不完整 (可能缺少地形或模型文件)")

                elapsed = time.time() - t0
                status = f"✔ 保存成功 ({elapsed:.1f}秒)"
                # 不在此处添加路径，在 _process 返回后统一添加
                folder_result = str(dest_folder) # 标记成功并记录路径
            except Exception as e:
                status = f"✖ 出错: {e}"
                # 如果出错，清理可能产生的残缺目标文件夹
                if dest_folder.exists():
                    SpatialTileDownloader._move_to_trash(dest_folder, trash_dir, prefix="error_during_process")
            finally:
                # 清理临时下载的 zip 文件
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

            with lock:
                print(f"[{idx}/{total_selected}] {sheet_no_std} -> {status}")
                processed_count += 1 # 计入已处理

            return folder_result # 返回路径（如果成功）或空字符串（如果失败）

        # 执行并行处理
        with _cf.ThreadPoolExecutor(max_workers=workers) as exe:
            # 收集所有线程的返回结果（路径或空字符串）
            results_paths = list(exe.map(_process, enumerate(selected_features, 1)))

        # 过滤掉处理失败的结果（空字符串），得到有效的 Tile 路径列表
        valid_tile_paths = [p for p in results_paths if p]

        print(f"\n--- 处理完成 ---")
        print(f"总共筛选出 {total_selected} 个 Tiles")
        print(f"已尝试处理 (下载/跳过/检查): {processed_count} 个 Tiles")
        print(f"成功下载或验证的 Tile 文件夹数量: {len(valid_tile_paths)}")

        # --- 新增：加载地形网格 ---
        if load_terrain and TRIMESH_AVAILABLE:
            print("\n--- 开始加载地形网格 ---")
            SpatialTileDownloader._load_terrain_meshes(
                tile_paths=valid_tile_paths, # 只加载成功处理的 Tiles
                fmt=fmt,
                face_threshold=terrain_face_threshold
            )
        elif load_terrain and not TRIMESH_AVAILABLE:
            print("\n--- 地形加载已跳过 ('trimesh' 或 'numpy' 库未安装) ---")
        else:
             print("\n--- 地形加载功能已禁用 ---")
        # --- 结束新增部分 ---

        return valid_tile_paths # 返回成功处理的 Tile 路径列表

    # ------------------------------------------------------------------
    # 内部辅助方法 (大部分保持不变, 除了 _normalize_and_clean 使用新正则)
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_intersects(b1: Tuple[float, float, float, float],
                         b2: Tuple[float, float, float, float]) -> bool:
        """检查两个边界框是否相交。"""
        # 检查无效输入 (例如 None)，尽管主要的筛选逻辑已处理此情况
        if not all(isinstance(c, (int, float)) for c in b1) or \
           not all(isinstance(c, (int, float)) for c in b2):
            return False
        # 标准相交逻辑
        return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])

    @staticmethod
    def _download_file(url: str, fp):
        """将文件下载到提供的文件对象 fp，支持重试。"""
        try:
            with requests.Session() as session:
                # 配置重试机制，针对常见的服务器错误
                retries = requests.adapters.Retry(
                    total=3, # 总重试次数
                    backoff_factor=0.5, # 重试间隔时间的增长因子
                    status_forcelist=[500, 502, 503, 504] # 需要重试的状态码
                )
                adapter = requests.adapters.HTTPAdapter(max_retries=retries)
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                # 使用超时设置 (连接超时, 读取超时)
                with session.get(url, stream=True, timeout=(10, 180)) as r:
                    r.raise_for_status() # 对错误的 HTTP 状态码 (4xx 或 5xx) 抛出异常
                    # 分块写入文件
                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        fp.write(chunk)
        except requests.exceptions.RequestException as e:
            # 处理请求相关的错误 (如网络问题、超时)
            raise RuntimeError(f"下载失败 {url}: {e}") from e
        except Exception as e:
            # 捕获其他可能的下载错误
            raise RuntimeError(f"从 {url} 下载过程中发生未知错误: {e}") from e

    @staticmethod
    def _tile_complete(folder: Path, fmt: str) -> bool:
        """
        检查 Tile 目录是否完整。
        要求包含一个以 'T' 开头的子目录，该子目录内含一个与子目录同名且格式后缀匹配的模型文件。
        例如: folder/T12345/T12345.3ds
        """
        fmt_ext = fmt.lower()
        if not folder.is_dir() or not any(folder.iterdir()):
            return False # 不是目录或目录为空

        # 查找以 'T' 开头的子目录
        for sub in folder.iterdir():
            if sub.is_dir() and sub.name.startswith("T"):
                # 检查 'T' 子目录内是否存在对应的模型文件
                target_model_file = sub / f"{sub.name}.{fmt_ext}"
                if target_model_file.is_file():
                    return True # 找到了所需的结构和文件
        return False # 未找到 T 子目录或其内部的模型文件

    @staticmethod
    def _move_to_trash(path: Path, trash_dir: Path, prefix: str):
        """安全地将文件或目录移动到回收站目录，并添加时间戳和前缀。"""
        if not path.exists():
            return
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 确保回收站中的名称唯一
            target_trash_path = trash_dir / f"{prefix}_{stamp}_{path.name}"
            counter = 0
            while target_trash_path.exists():
                counter += 1
                target_trash_path = trash_dir / f"{prefix}_{stamp}_{path.name}_{counter}"
            # 移动文件/目录
            shutil.move(str(path), target_trash_path)
            # print(f"已将 '{path.name}' 移动到回收站，名为 '{target_trash_path.name}'") # 可选：记录移动操作
        except Exception as e:
            print(f"错误：将 {path.name} 移动到回收站时失败: {e}")

    @staticmethod
    def _normalize_and_clean(out_dir: Path, fmt: str, trash_dir: Path):
        """
        启动时检查：检查所有子目录。
        将旧格式（带连字符）重命名为标准格式。
        将空目录、不完整目录或名称不规范的目录移动到回收站。
        """
        print(f"正在检查目录: {out_dir}")
        if not out_dir.is_dir():
            print("输出目录不存在，无需检查。")
            return

        # 获取所有子目录（排除 .trash 目录）
        subdirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name != ".trash"]
        print(f"发现 {len(subdirs)} 个现有子目录（不含 .trash）待检查...")
        checked_count, renamed_count, trashed_count = 0, 0, 0

        for p in sorted(subdirs): # 按字母顺序处理
            checked_count += 1
            current_name = p.name
            standard_name = current_name # 初始假定为标准名称
            original_path = p # 保留原始路径对象引用

            # 1. 检查旧格式 (带连字符) -> 重命名
            if SpatialTileDownloader.OLD_NAME_PAT.match(current_name):
                standard_name = current_name.replace("-", "")
                target_path = out_dir / standard_name
                print(f"  检查: {current_name} (旧格式) -> 目标: {standard_name}")
                if target_path.exists():
                    # 如果标准名称已存在，将旧格式的目录移入回收站
                    print(f"    状态: 目标 '{standard_name}' 已存在。正在将旧格式 '{current_name}' 移入回收站。")
                    SpatialTileDownloader._move_to_trash(original_path, trash_dir, "duplicate_old_format")
                    trashed_count += 1
                    continue # 处理下一个目录
                else:
                    # 尝试重命名
                    try:
                        original_path.rename(target_path)
                        print(f"    状态: 已成功重命名为 '{standard_name}'")
                        p = target_path # 更新路径引用，以便后续内容检查
                        renamed_count += 1
                    except Exception as e:
                        print(f"    错误: 重命名 '{current_name}' 为 '{standard_name}' 失败: {e}")
                        SpatialTileDownloader._move_to_trash(original_path, trash_dir, "rename_failed")
                        trashed_count += 1
                        continue # 处理下一个目录

            # 2. 检查标准格式 (使用更新后的正则表达式)
            elif SpatialTileDownloader.STANDARD_NAME_PAT.match(current_name):
                 print(f"  检查: {current_name} (标准格式)")
                 # 名称格式正确，继续进行内容检查
                 pass
            # 3. 非标准格式 -> 移入回收站
            else:
                 print(f"  检查: {current_name} (非标准格式)")
                 print(f"    状态: 名称格式不符合标准 (\\d{{1,2}}[A-Z]{{2}}\\d{{1,2}}[A-Z])。正在移入回收站。")
                 SpatialTileDownloader._move_to_trash(original_path, trash_dir, "non_standard_name")
                 trashed_count += 1
                 continue # 跳过内容检查

            # 4. 内容检查 (仅当名称为标准格式或已成功重命名后执行)
            if not any(p.iterdir()):
                print(f"    状态: 目录为空。正在移入回收站。")
                SpatialTileDownloader._move_to_trash(p, trash_dir, "empty")
                trashed_count += 1
            elif not SpatialTileDownloader._tile_complete(p, fmt):
                print(f"    状态: 目录不完整 (缺少地形/模型文件)。正在移入回收站。")
                SpatialTileDownloader._move_to_trash(p, trash_dir, "incomplete")
                trashed_count += 1
            else:
                print(f"    状态: 验证通过 (目录完整且命名标准)。")

        print(f"\n目录检查总结: 共检查 {checked_count} 个, 重命名 {renamed_count} 个, 移入回收站 {trashed_count} 个。")


    @staticmethod
    def _unzip_and_standardize(zip_path: Path, dest_folder: Path, sheet_no_std: str, trash_dir: Path):
        """
        解压 ZIP 压缩包，并确保最终解压出的文件夹使用标准的无连字符名称。
        处理压缩包内文件夹可能带或不带连字符的情况。
        """
        extract_base = dest_folder.parent # 解压到最终目标文件夹的父目录
        extracted_folder_path: Path | None = None # 存储实际解压出的文件夹路径

        try:
            with zipfile.ZipFile(zip_path) as zf:
                # --- 预测 ZIP 包内顶层文件夹的名称 ---
                # 获取压缩包内所有文件路径中的顶层目录名称（去重）
                top_level_dirs = {Path(info.filename).parts[0] for info in zf.infolist()
                                  if ('/' in info.filename or '\\' in info.filename) and info.filename.strip()}
                potential_extracted_name = ""

                if len(top_level_dirs) == 1:
                    # 如果只有一个顶层目录，假定就是它
                    potential_extracted_name = list(top_level_dirs)[0]
                elif sheet_no_std in top_level_dirs:
                    # 如果标准名称本身就是顶层目录之一
                    potential_extracted_name = sheet_no_std
                else:
                    # 尝试查找移除连字符后与标准名称匹配的目录名
                    for name in top_level_dirs:
                        if name.replace("-", "") == sheet_no_std:
                            potential_extracted_name = name
                            break
                    # 如果还没找到，可能是 ZIP 没有单一根目录，或者命名非常规

                # --- 解压所有内容 ---
                zf.extractall(extract_base)

            # --- 查找实际解压出的文件夹路径 ---
            if potential_extracted_name:
                 actual_extracted_path = extract_base / potential_extracted_name
                 if actual_extracted_path.is_dir():
                     extracted_folder_path = actual_extracted_path
                 else: # 预测错误，需要手动查找
                     potential_extracted_name = "" # 重置预测

            # 如果预测失败或错误，在解压目录下搜索
            if not extracted_folder_path:
                 for p in extract_base.iterdir():
                     # 检查目录名移除连字符后是否与标准名匹配
                     if p.is_dir() and p.name.replace("-", "") == sheet_no_std:
                         extracted_folder_path = p
                         break

            if extracted_folder_path is None:
                # 如果仍然找不到，说明 ZIP 结构有问题
                raise RuntimeError(f"从 {zip_path.name} 解压后未能找到预期的目录 '{sheet_no_std}' (或其变体)")

            # --- 确保最终文件夹名称是标准的无连字符格式 ---
            if extracted_folder_path.name != sheet_no_std:
                # 如果解压出的名称与标准名不同 (例如，带了连字符)，则进行重命名
                if dest_folder.exists():
                    # 如果目标标准名称的文件夹已存在 (例如，上次运行中断留下的)
                    print(f"    警告: 目标目录 '{dest_folder.name}' 已存在。尝试合并...")
                    try:
                        # 将解压出的文件夹 (非标准名) 的内容移动到已存在的标准名文件夹中
                        for item in extracted_folder_path.iterdir():
                            target_item_path = dest_folder / item.name
                            if target_item_path.exists():
                                print(f"      跳过已存在的文件/目录: {item.name}")
                            else:
                                shutil.move(str(item), target_item_path)
                        # 移除现已为空的、非标准名的解压文件夹
                        shutil.rmtree(extracted_folder_path, ignore_errors=True)
                        print(f"    成功合并到 '{dest_folder.name}'。")
                    except Exception as merge_err:
                        # 如果合并失败，将两个文件夹都移入回收站，避免状态不一致
                        print(f"    合并过程中出错: {merge_err}。正在将两个相关文件夹移入回收站。")
                        SpatialTileDownloader._move_to_trash(dest_folder, trash_dir, "merge_target_conflict")
                        SpatialTileDownloader._move_to_trash(extracted_folder_path, trash_dir, "merge_source_conflict")
                        raise RuntimeError(f"合并目录失败: {merge_err}") from merge_err
                else:
                    # 如果目标标准名称的文件夹不存在，直接重命名解压出的文件夹
                    try:
                        extracted_folder_path.rename(dest_folder)
                        print(f"    已将解压出的文件夹 '{extracted_folder_path.name}' 重命名为 '{dest_folder.name}'。")
                    except Exception as rename_err:
                        # 如果重命名失败，将被解压出的文件夹移入回收站
                        print(f"    重命名解压文件夹时出错: {rename_err}。正在将 '{extracted_folder_path.name}' 移入回收站。")
                        SpatialTileDownloader._move_to_trash(extracted_folder_path, trash_dir, "rename_extracted_failed")
                        raise RuntimeError(f"重命名解压目录失败: {rename_err}") from rename_err
            # else: # 如果解压出的文件夹名称已经是标准格式，则无需操作
            #     print(f"    解压后的文件夹名称 '{extracted_folder_path.name}' 符合标准。")


        except zipfile.BadZipFile:
            # 处理无效的 ZIP 文件错误
            raise RuntimeError(f"下载的文件 {zip_path.name} 不是一个有效的 ZIP 文件") from None
        except Exception as e:
            # 捕获其他解压或标准化过程中的错误
            # 如果目标文件夹存在于不确定状态，清理它
            if dest_folder.exists():
                 SpatialTileDownloader._move_to_trash(dest_folder, trash_dir, "unzip_error_cleanup")
            # 如果解压出的文件夹仍然存在且与目标不同，也清理它
            if extracted_folder_path and extracted_folder_path.exists() and extracted_folder_path != dest_folder:
                 SpatialTileDownloader._move_to_trash(extracted_folder_path, trash_dir, "unzip_source_cleanup")
            raise RuntimeError(f"为 {sheet_no_std} 解压或标准化过程中出错: {e}") from e


    @staticmethod
    def _print_selection_summary(bbox: Tuple[float, float, float, float], features: list[dict]):
        """打印筛选出的 Tiles 的摘要信息。"""
        print(f"\n· 研究范围 BBox (HK80): {bbox}")
        tile_count = len(features)
        print(f"· 相交且有效的 Tiles 数量: {tile_count}")
        if tile_count == 0:
            print("-" * 60)
            return
        max_display = 10 # 最多显示的名称数量
        # 以标准 (无连字符) 格式显示名称
        names_to_display = [f.get("properties", {}).get("Sheet_No", "未知").replace("-", "") for f in features]
        if tile_count <= max_display:
            print(f"· Tile 名称 (标准格式): {', '.join(names_to_display)}")
        else:
            # 如果列表太长，显示开头和结尾的部分名称
            print(f"· Tile 名称示例 (标准格式): {', '.join(names_to_display[:max_display//2])} ... {', '.join(names_to_display[-max_display//2:])}")
        print("-" * 60)

    @staticmethod
    def _print_dataset_geometry(selected_features: list[dict]):
        """
        修改：仅打印 *选中的* Tiles 的 GEOMETRYCOLLECTION WKT。
        """
        polygons = []
        print("\n===== 选中 Tiles 的 GeometryCollection WKT =====")
        if not selected_features:
            print("没有选中的 Tiles。")
            print("===========================================\n")
            return

        for f in selected_features:
            try:
                # 从要素的 geometry 中提取坐标
                # 假设是 Polygon 几何类型，且只有一个外部环
                coords = f["geometry"]["coordinates"][0]
                # 将坐标格式化为 WKT 字符串的一部分
                wkt_coords = ",".join(f"{x} {y}" for x, y in coords)
                # 创建 WKT Polygon 字符串
                wkt = f"POLYGON(({wkt_coords}))"
                polygons.append(wkt)
            except (KeyError, IndexError, TypeError) as e:
                # 处理无效或缺失的几何信息
                sheet_no = f.get("properties", {}).get("Sheet_No", "未知")
                print(f"警告：无法提取 Tile {sheet_no} 的几何信息: {e}")
                continue # 跳过此要素

        if polygons:
            # 将所有 Polygon WKT 字符串合并成一个 GEOMETRYCOLLECTION
            all_gc = "GEOMETRYCOLLECTION(" + ",".join(polygons) + ")"
            print(all_gc)
        else:
            print("在选中的 Tiles 中未找到有效的几何信息。")
        print("===========================================\n")

    # --- 新增：地形加载方法 ---
    @staticmethod
    def _load_terrain_meshes(tile_paths: List[str], fmt: str, face_threshold: int):
        """
        使用 trimesh 从下载的 Tile 文件夹中加载地形网格 (.3ds)。
        优先加载 'simplify_*.3ds' 文件。
        """
        if not TRIMESH_AVAILABLE:
            print("Trimesh 或 numpy 库不可用，无法加载地形。")
            return

        fmt_ext = fmt.lower()
        if fmt_ext != '3ds':
            print(f"地形加载目前仅支持 '.3ds' 格式，而非 '.{fmt_ext}'。正在跳过。")
            return

        loaded_count = 0
        warning_count = 0
        error_count = 0

        print("-" * 60) # 分隔符

        for tile_path_str in tile_paths:
            tile_path = Path(tile_path_str)
            tile_name = tile_path.name
            print(f"正在处理 Tile: {tile_name}") # 打印正在处理的 Tile 名称

            terrain_folder: Optional[Path] = None

            # 查找地形子文件夹 (以 'T' 开头)
            try:
                terrain_folders = list(tile_path.glob("T*"))
                if not terrain_folders:
                    # print(f"  信息: 在 {tile_name} 中未找到地形子文件夹 (T*)")
                    continue # 跳过没有 T 文件夹的 Tile
                # 假定找到的第一个是正确的地形文件夹
                terrain_folder = terrain_folders[0]
                if not terrain_folder.is_dir():
                     # print(f"  信息: 在 {tile_name} 中找到以 'T' 开头的项，但它不是目录")
                     continue # 跳过非目录项
            except Exception as e:
                print(f"  错误: 访问 {tile_name} 的子文件夹时出错: {e}")
                error_count += 1
                continue

            if not terrain_folder: # 理论上不会执行到这里，但为了安全起见检查
                continue

            terrain_folder_name = terrain_folder.name
            simplified_file = terrain_folder / f"simplify_{terrain_folder_name}.{fmt_ext}"
            original_file = terrain_folder / f"{terrain_folder_name}.{fmt_ext}"
            file_to_load: Optional[Path] = None
            load_type = "" # 记录加载的是简化版还是原始版

            # 优先加载简化文件
            if simplified_file.is_file():
                file_to_load = simplified_file
                load_type = "简化版"
            elif original_file.is_file():
                file_to_load = original_file
                load_type = "原始版"
            else:
                print(f"  警告: 在 {terrain_folder} 中未找到 '.{fmt_ext}' 或 'simplify_*.{fmt_ext}' 地形文件")
                error_count += 1
                continue

            # 获取文件大小
            try:
                file_size_bytes = os.path.getsize(file_to_load)
                file_size_kb = file_size_bytes / 1024
                size_str = f"{file_size_kb:.2f} KB"
            except OSError as e:
                print(f"  警告: 无法获取文件大小 {file_to_load.name}: {e}")
                size_str = "未知大小"
                error_count += 1 # 算作一个错误

            print(f"  尝试加载: {file_to_load.name} ({load_type}), 大小: {size_str}")

            # 使用 trimesh 加载网格
            try:
                # **关键改动：明确指定 file_type='3ds'**
                mesh = trimesh.load(str(file_to_load), file_type='3ds', force="scene", process=False)

                # 检查加载结果是 Scene 还是 Mesh 对象
                if isinstance(mesh, trimesh.Scene):
                     # 如果是场景 (Scene)，尝试将其中的几何体合并为一个网格
                     # .3ds 文件可能包含多个对象
                     if not mesh.geometry:
                         print(f"  警告: 加载的 {file_to_load.name} 是一个空场景 (Scene)。")
                         error_count += 1
                         continue
                     # 合并场景中的所有网格
                     mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
                     print(f"  信息: 以场景形式加载了 {file_to_load.name} ({load_type})，已合并为单个网格。")

                if not isinstance(mesh, trimesh.Trimesh):
                    # 如果加载结果不是有效的 Trimesh 对象
                    print(f"  警告: 未能将 {file_to_load.name} 加载为有效的网格对象 (类型: {type(mesh)})。")
                    error_count += 1
                    continue

                # 获取网格信息
                num_faces = len(mesh.faces)
                num_vertices = len(mesh.vertices)
                bounds = mesh.bounds # 获取边界框 [[min_x, min_y, min_z], [max_x, max_y, max_z]]
                bbox_min = bounds[0]
                bbox_max = bounds[1]
                loaded_count += 1

                # 打印网格信息
                print(f"    -> 加载成功: 顶点数={num_vertices}, 面数={num_faces}")
                print(f"    -> 边界框 (Min): [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}]")
                print(f"    -> 边界框 (Max): [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")


                # 检查面数阈值
                if num_faces > face_threshold:
                    print(f"    --> 警告: 网格面数 {num_faces} (> {face_threshold})，建议进行手动简化。")
                    warning_count += 1

            except ValueError as ve:
                 # 处理 trimesh/pyassimp 可能抛出的值错误 (例如文件损坏或格式无法识别)
                 print(f"  错误加载 {file_to_load.name}: {ve}。文件可能已损坏或格式不受支持。")
                 error_count += 1
            except ImportError as ie:
                 # 特别捕捉可能的导入错误，如果 pyassimp 依赖缺失
                 print(f"  错误加载 {file_to_load.name}: 导入错误 {ie}。请确保 pyassimp 及其依赖已正确安装。")
                 error_count += 1
            except Exception as e:
                # 捕获其他意外的加载错误
                # 检查是否是 'file_type not supported' 错误
                if "file_type" in str(e) and "not supported" in str(e):
                     print(f"  错误加载 {file_to_load.name}: 明确指定 file_type='3ds' 后仍然失败。错误: {e}")
                     print("    -> 可能原因: 底层 Assimp 库不支持 .3ds 或 pyassimp 安装不完整。")
                else:
                     print(f"  错误加载 {file_to_load.name}: 发生未知错误: {e}")
                error_count += 1
            print("-" * 30) # 每个 Tile 处理后的分隔符

        print(f"\n--- 地形加载总结 ---")
        print(f"成功加载的网格数量: {loaded_count}")
        print(f"面数超过阈值 ({face_threshold}) 的网格数量: {warning_count}")
        print(f"加载过程中出错的数量: {error_count}")
        print("-" * 60)


# --- 示例用法 (保留用于测试, 请确保路径正确) ---
if __name__ == "__main__":
    # --- 配置 ---
    # 研究区域的边界框 (示例坐标)
    MIN_X, MIN_Y = 830000, 815000
    MAX_X, MAX_Y = 843500, 828800 # 示例范围
    # MAX_X, MAX_Y = 831000, 816000 # 用于快速测试的小范围

    FORMAT = "3DS" # 要下载的格式

    # !!! 重要：请将此路径更新为您实际的 GeoJSON 索引文件路径 !!!
    # 例如: GEOJSON_FILE = Path("C:/data/maps/HK_index.geojson")
    GEOJSON_FILE = Path("library/HK_map/row_map/3d_spatial_data/b1000_Tile_Feb25.gdb_converted.json")

    # !!! 重要：请将此路径更新为您希望保存 Tiles 的目录 !!!
    # 例如: OUTPUT_DIR = Path("./downloaded_hk_tiles")
    OUTPUT_DIR = Path("library/HK_map/processed/3d_spatial_tiles_test")

    # --- 运行前检查 ---
    if not GEOJSON_FILE.is_file():
        print("="*30)
        print(f"错误：GeoJSON 索引文件未找到:")
        print(GEOJSON_FILE.resolve())
        print("\n请修改脚本中的 'GEOJSON_FILE' 变量。")
        print("="*30)
    else:
        print(f"使用 GeoJSON 索引: {GEOJSON_FILE.resolve()}")
        print(f"输出目录:           {OUTPUT_DIR.resolve()}")
        print(f"研究区域 BBox (HK80): ({MIN_X}, {MIN_Y}) 到 ({MAX_X}, {MAX_Y})")
        print(f"下载格式:           {FORMAT}")
        print("-" * 30)

        # --- 执行下载和加载 ---
        try:
            # 注意：download_b1000_tiles 现在返回成功处理的文件夹列表
            # 但根据要求，我们不再打印这个列表
            SpatialTileDownloader.download_b1000_tiles(
                min_x=MIN_X, min_y=MIN_Y, max_x=MAX_X, max_y=MAX_Y,
                fmt=FORMAT,
                geojson_path=GEOJSON_FILE,
                out_dir=OUTPUT_DIR,
                overwrite=False, # 设置为 True 强制重新下载
                workers=8,       # 根据您的系统/网络调整线程数
                load_terrain=True # 启用地形加载
            )
            print("\n下载和地形加载流程已完成。")
            # --- 移除打印列表 ---
            # print("\n成功处理的文件夹:")
            # for folder in downloaded_folders:
            #     print(f" - {folder}")

        except Exception as e:
            print(f"\n处理过程中发生严重错误: {e}")
            import traceback
            traceback.print_exc() # 打印详细的错误回溯信息，方便调试
