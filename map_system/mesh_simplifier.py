import trimesh
import os
from pathlib import Path
import numpy as np # trimesh 的依赖

# --- 用户配置 ---
INPUT_DIR = Path(r"E:\Code\HITSZ\UrbanWindowVisionKG\library\HK_map\processed\test\11SW8B\T33750162000106E1Z")  # <--- 修改为包含 .3ds 文件的根目录
OUTPUT_DIR = Path(r"E:\Code\HITSZ\UrbanWindowVisionKG\library\HK_map\processed\test\11SW8B\T33750162000106E1Z") # <--- 修改为保存 .obj 文件的目标目录
TARGET_FACE_COUNT = 10000                  # <--- 设置简化后的目标面数 (根据需要调整)
RECURSIVE = True                           # 是否递归查找子目录中的 .3ds 文件 (True/False)
PROCESS_EXISTING = False                   # 如果输出 .obj 文件已存在，是否重新处理 (True/False)
# --- 结束配置 ---

# 检查输入目录是否存在
if not INPUT_DIR.is_dir():
    print(f"错误：输入目录 '{INPUT_DIR}' 不存在或不是一个目录。")
    exit()

# 创建输出目录 (如果不存在)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"开始批量转换和简化...")
print(f"源目录: {INPUT_DIR.resolve()}")
print(f"目标目录: {OUTPUT_DIR.resolve()}")
print(f"目标面数: {TARGET_FACE_COUNT}")
print(f"递归查找: {RECURSIVE}")
print("-" * 40)

processed_count = 0
success_count = 0
skipped_count = 0
error_count = 0

# 根据 RECURSIVE 设置文件查找模式
if RECURSIVE:
    file_pattern = "**/*.3ds"
else:
    file_pattern = "*.3ds"

# 遍历查找 .3ds 文件
for input_path in INPUT_DIR.glob(file_pattern):
    processed_count += 1
    print(f"\n[{processed_count}] 处理: {input_path.relative_to(INPUT_DIR)}")

    # 构建输出路径，保持子目录结构
    relative_path = input_path.relative_to(INPUT_DIR)
    output_path = (OUTPUT_DIR / relative_path).with_suffix(".obj")

    # 检查输出文件是否已存在，并根据 PROCESS_EXISTING 决定是否跳过
    if output_path.exists() and not PROCESS_EXISTING:
        print(f"  -> 跳过: 输出文件 '{output_path.name}' 已存在。")
        skipped_count += 1
        continue

    # 确保输出文件的父目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 加载 .3ds 文件
        # 明确指定 file_type='3ds' 仍然是好习惯，以防万一
        # process=False 避免 trimesh 自动处理，我们后面手动合并
        print(f"  加载中...")
        loaded_data = trimesh.load(str(input_path), file_type='3ds', process=False)

        # 2. 处理 Scene 或 Mesh
        mesh: trimesh.Trimesh | None = None # 类型提示
        if isinstance(loaded_data, trimesh.Scene):
            # 如果是场景，合并所有几何体
            geometries = list(loaded_data.geometry.values())
            if not geometries:
                print("  警告: 加载的场景为空，无法处理。")
                error_count += 1
                continue
            print(f"  场景包含 {len(geometries)} 个几何体，正在合并...")
            # 使用 trimesh.util.concatenate 来合并列表中的网格
            if len(geometries) > 1:
                 mesh = trimesh.util.concatenate(geometries)
            else:
                 mesh = geometries[0] # 如果只有一个几何体，直接使用它

        elif isinstance(loaded_data, trimesh.Trimesh):
            # 如果直接加载为网格
            mesh = loaded_data
        else:
            # 加载了未知类型
            print(f"  错误: 加载返回了未知类型 {type(loaded_data)}。")
            error_count += 1
            continue

        # 检查合并或加载后的网格是否有效
        if not mesh or not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            print("  错误: 加载或合并后得到无效或空的网格。")
            error_count += 1
            continue

        print(f"  原始面数: {len(mesh.faces)}")

        # 3. 网格简化
        mesh_to_save = mesh # 默认保存原始（或合并后）的网格
        if len(mesh.faces) > TARGET_FACE_COUNT:
            print(f"  简化中 (目标: {TARGET_FACE_COUNT} 面)...")
            try:
                # 使用二次边收缩抽取算法进行简化
                simplified_mesh = mesh.simplify_quadric_decimation(TARGET_FACE_COUNT)
                print(f"  简化后面数: {len(simplified_mesh.faces)}")
                mesh_to_save = simplified_mesh
            except ImportError:
                 print("  错误: 简化失败，可能缺少 'pyfqmr' 库。请尝试 'pip install pyfqmr'。")
                 print("        将保存未简化的网格。")
                 # mesh_to_save 已经是原始 mesh，无需改变
            except Exception as simp_error:
                 print(f"  错误: 简化过程中发生错误: {simp_error}")
                 print("        将保存未简化的网格。")
                 # mesh_to_save 已经是原始 mesh，无需改变
        else:
            print(f"  面数 ({len(mesh.faces)}) 已低于或等于目标值，无需简化。")

        # 4. 导出为 .obj
        print(f"  导出到: {output_path}")
        # encoding='utf-8' 可能有助于避免某些环境下的字符问题
        mesh_to_save.export(str(output_path), file_type='obj', encoding='utf-8')
        success_count += 1

    except ValueError as ve:
        # 特别捕捉与格式支持相关的 ValueError
        if "file type" in str(ve).lower() and "not supported" in str(ve).lower():
            print(f"  错误: 加载失败 - {ve}")
            print("        => 请确认你的 trimesh/pyassimp 环境已正确安装并支持 .3ds 格式。")
        else:
            print(f"  错误: 处理文件时发生值错误: {ve}")
        error_count += 1
    except ImportError as ie:
        print(f"  错误: 缺少导入 - {ie}。请确保所有依赖已安装 (trimesh, numpy, pyfqmr?)。")
        error_count += 1
        # 如果是关键导入错误，可能需要停止脚本
        # raise ie # 取消注释以在导入错误时停止
    except Exception as e:
        print(f"  未知错误: 处理文件时发生异常: {e}")
        import traceback
        # traceback.print_exc() # 取消注释以查看详细的错误堆栈
        error_count += 1

print("\n" + "=" * 40)
print("批量处理完成！")
print(f"总共检查文件数: {processed_count}")
print(f"成功转换并保存: {success_count}")
print(f"跳过 (已存在):   {skipped_count}")
print(f"失败/错误:       {error_count}")
print("=" * 40)