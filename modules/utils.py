import time
import os
import json


# --- Utility Functions (Keep as before) ---
def log_time(message: str, start_time: float):
    print(f"[{time.strftime('%H:%M:%S')}] {message} (耗时: {time.perf_counter() - start_time:.2f}s)")


def load_json_data(json_path: str):
    start_time = time.perf_counter()
    print(f"开始加载 JSON: {os.path.basename(json_path)}")
    if not os.path.exists(json_path):
        print(f"错误: 文件未找到 {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log_time(f"完成加载 JSON: {os.path.basename(json_path)}", start_time)
        return data
    except json.JSONDecodeError as e:
        raise Exception(f"错误: 解析 JSON 文件失败 {json_path}: {e}")
    except IOError as e:
        raise Exception(f"错误: 读取 JSON 文件失败 {json_path}: {e}")

