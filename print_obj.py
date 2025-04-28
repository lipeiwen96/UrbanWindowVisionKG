# obj_to_triangles.py
from pathlib import Path
import numpy as np
from typing import List, Tuple
import sys, pprint, textwrap


# ──────────────────── 1) 基础类型 ────────────────────
Vec3 = Tuple[float, float, float]            # 单个顶点 (x,y,z)
Tri  = Tuple[Vec3, Vec3, Vec3]               # 单个三角面 (v1,v2,v3)


# ──────────────────── 2) 主函数 ────────────────────
def obj_to_triangles(obj_path: str | Path) -> List[Tri]:
    """读取 .obj 文件并返回三角面列表。"""
    obj_path = Path(obj_path)
    if not obj_path.is_file():
        raise FileNotFoundError(obj_path)

    verts: List[Vec3] = []      # 索引 0 ─► v1，但 .obj 索引从 1 计数
    tris:  List[Tri]  = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue                        # 注释 / 空行
            head, *rest = line.strip().split()
            if head == "v":                    # ── 顶点
                verts.append(tuple(map(float, rest)))
            elif head == "f":                  # ── 面
                # 支持诸如 "f v/vt/vn"、"f v//vn"，先只提取顶点索引部分
                raw_idx = [chunk.split("/")[0] for chunk in rest]
                idx = [int(i) - 1 for i in raw_idx]          # 调整到 0-based
                # 扇形三角化：v0-vi-vi+1
                for i in range(1, len(idx) - 1):
                    tris.append((
                        verts[idx[0]],
                        verts[idx[i]],
                        verts[idx[i + 1]],
                    ))
            # 其余指令 (vn/vt/usemtl/…) 可按需扩展

    return tris


# ──────────────────── 3) CLI & 简单测试 ────────────────────
if __name__ == "__main__":

    tri_list = obj_to_triangles("library/HK_map/processed/project/北侧地形.obj")
    print(f"✓ 解析完成，共 {len(tri_list):,} 个三角面")
    pprint.pprint(tri_list[:3])
