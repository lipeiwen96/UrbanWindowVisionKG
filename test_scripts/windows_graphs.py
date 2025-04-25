# -*- coding: utf-8 -*-
import networkx as nx
from pyvis.network import Network
import random
import webbrowser
import os
import math # 用于模拟向量距离计算

# --- 1. 定义图谱和节点/边类型 ---
# 创建一个有向图，因为关系通常是有方向的（例如 "可视"）
G = nx.DiGraph()

# 定义节点类型和颜色，方便可视化
NODE_TYPES = {
    "Window": {"color": "#42a5f5", "size": 25},  # 蓝色
    "Building": {"color": "#bdbdbd", "size": 20},  # 灰色
    "Landmark": {"color": "#ef5350", "size": 30},  # 红色
    "SemanticFeature": {"color": "#66bb6a", "size": 15}, # 绿色
    "CLIPVector": {"color": "#ffee58", "size": 10}, # 黄色
}

# 定义关系类型
RELATIONSHIP_TYPES = [
    "visibility",        # 可视
    "co_visibility",     # 共视
    "semantic_similarity", # 语义相似
    "adjacent",          # 邻接
    "has_feature",       # 具有特征
    "has_vector",        # 具有向量
    "part_of",           # 部分属于 (例如窗户属于建筑)
]

# --- 2. 生成假数据并构建图谱 ---

# --- 创建节点 ---
# 窗户节点 (Windows)
windows = []
for i in range(1, 6):
    window_id = f"Window_{i}"
    # 随机分配一些属性
    green_view_ratio = random.choice(["High", "Medium", "Low"])
    feature_id = f"Feature_GVR_{green_view_ratio}_{i}"
    vector_id = f"Vector_Window_{i}"
    building_id = f"Building_{random.randint(1, 2)}" # 假设窗户属于某栋建筑

    G.add_node(window_id,
               label=f"窗户 {i}",
               title=f"窗户 {i}\n绿视率: {green_view_ratio}\n所属建筑: {building_id}",
               node_type="Window",
               green_view_ratio=green_view_ratio,
               building=building_id)
    windows.append(window_id)

    # 添加对应的语义特征节点 (如果需要单独表示)
    G.add_node(feature_id,
               label=f"绿视率: {green_view_ratio}",
               title=f"语义特征: 绿视率 {green_view_ratio}",
               node_type="SemanticFeature",
               feature_type="GreenViewRatio",
               value=green_view_ratio)
    G.add_edge(window_id, feature_id, label="has_feature", title="具有特征") # 添加关系

    # 添加对应的CLIP向量节点 (简化表示 - 存储一个假的浮点数用于模拟距离)
    fake_vector_val = random.uniform(0, 1) # 生成一个0-1之间的随机浮点数作为模拟向量的关键特征
    G.add_node(vector_id,
               label=f"向量 {i}",
               title=f"CLIP 图像向量 (Window {i})\n模拟值: {fake_vector_val:.4f}",
               node_type="CLIPVector",
               vector_value=f"[{fake_vector_val:.4f} ...]", # 假的向量值
               simulated_value=fake_vector_val) # 存储模拟值用于计算距离
    G.add_edge(window_id, vector_id, label="has_vector", title="具有向量") # 添加关系

# 建筑节点 (Buildings)
buildings = []
for i in range(1, 3):
    building_id = f"Building_{i}"
    building_type = random.choice(["住宅", "办公楼"])
    G.add_node(building_id,
               label=f"建筑 {i}",
               title=f"建筑 {i}\n类型: {building_type}",
               node_type="Building",
               type=building_type,
               location=f"区域 {chr(64+i)}") # 假位置
    buildings.append(building_id)

# 地标节点 (Landmarks)
landmarks = ["公园A", "塔楼B", "河流C"]
landmark_nodes = []
for name in landmarks:
    landmark_id = f"Landmark_{name}"
    vector_id = f"Vector_Landmark_{name}"
    G.add_node(landmark_id,
               label=name,
               title=f"地标: {name}",
               node_type="Landmark")
    landmark_nodes.append(landmark_id)

    # 添加地标的CLIP文本向量节点 (简化表示)
    fake_vector_val = random.uniform(0, 1)
    G.add_node(vector_id,
               label=f"向量 ({name})",
               title=f"CLIP 文本向量 ({name})\n模拟值: {fake_vector_val:.4f}",
               node_type="CLIPVector",
               vector_value=f"[{fake_vector_val:.4f} ...]", # 假的向量值
               simulated_value=fake_vector_val)
    G.add_edge(landmark_id, vector_id, label="has_vector", title="具有向量") # 添加关系

# --- 创建关系 (边) ---
# 窗户 -> 地标 (可视性) - 随机分配
for window in windows:
    visible_landmarks = random.sample(landmark_nodes, k=random.randint(0, len(landmarks)))
    for landmark in visible_landmarks:
        G.add_edge(window, landmark, label="visibility", title="可视")

# 窗户 -> 建筑 (所属关系)
for node, data in G.nodes(data=True):
    if data.get("node_type") == "Window" and "building" in data:
        # Ensure the building node exists before adding the edge
        if G.has_node(data["building"]):
             G.add_edge(node, data["building"], label="part_of", title="属于")

# 建筑 -> 建筑 (邻接) - 假设建筑1和建筑2邻接
if len(buildings) >= 2:
    G.add_edge(buildings[0], buildings[1], label="adjacent", title="邻接")
    G.add_edge(buildings[1], buildings[0], label="adjacent", title="邻接") # 双向

# 窗户 -> 窗户 (共视) - 如果两个窗户能看到同一个地标
visible_map = {} # 地标 -> [能看到它的窗户]
for u, v, data in G.edges(data=True):
    # Make sure v is a Landmark node before adding to visible_map
    if G.has_node(v) and data.get("label") == "visibility" and G.nodes[v].get("node_type") == "Landmark":
        if v not in visible_map:
            visible_map[v] = []
        visible_map[v].append(u) # u is the window

for landmark, viewers in visible_map.items():
    if len(viewers) > 1:
        for i in range(len(viewers)):
            for j in range(i + 1, len(viewers)):
                # FIX: Removed key argument from has_edge for DiGraph
                # Check if an edge already exists between these two windows in this direction
                if not G.has_edge(viewers[i], viewers[j]) and viewers[i] != viewers[j]:
                     G.add_edge(viewers[i], viewers[j], label="co_visibility", title=f"共视({G.nodes[landmark]['label']})")
                     # Consider if the reverse edge should also be added for co-visibility
                     # if not G.has_edge(viewers[j], viewers[i]):
                     #    G.add_edge(viewers[j], viewers[i], label="co_visibility", title=f"共视({G.nodes[landmark]['label']})")


# 窗户 -> 窗户 (语义相似) - 假设绿视率相同的窗户语义相似
gvr_map = {} # 绿视率 -> [具有该绿视率的窗户]
for node, data in G.nodes(data=True):
    if data.get("node_type") == "Window" and "green_view_ratio" in data:
        gvr = data["green_view_ratio"]
        if gvr not in gvr_map:
            gvr_map[gvr] = []
        gvr_map[gvr].append(node)

for gvr, similar_windows in gvr_map.items():
     if len(similar_windows) > 1:
        for i in range(len(similar_windows)):
            for j in range(i + 1, len(similar_windows)):
                 # FIX: Removed key argument from has_edge for DiGraph
                 # Check if an edge already exists between these two windows in this direction
                 if not G.has_edge(similar_windows[i], similar_windows[j]) and similar_windows[i] != similar_windows[j]:
                    G.add_edge(similar_windows[i], similar_windows[j], label="semantic_similarity", title=f"语义相似(绿视率:{gvr})")
                    # Consider if the reverse edge should also be added for semantic similarity
                    # if not G.has_edge(similar_windows[j], similar_windows[i]):
                    #    G.add_edge(similar_windows[j], similar_windows[i], label="semantic_similarity", title=f"语义相似(绿视率:{gvr})")


# --- 3. 可视化图谱 ---
net = Network(notebook=False, height="800px", width="100%", directed=True, bgcolor="#222222", font_color="white")

# 添加节点和样式
for node, data in G.nodes(data=True):
    node_type = data.get("node_type", "Unknown")
    style = NODE_TYPES.get(node_type, {"color": "grey", "size": 10})
    net.add_node(node,
                 label=data.get("label", node),
                 title=data.get("title", node),
                 color=style["color"],
                 size=style["size"],
                 physics=True) # 启用物理引擎让布局更好看

# 添加边和样式
# Use a set to keep track of added edges to avoid duplicates in visualization if reverse edges were added
added_edges = set()
for u, v, data in G.edges(data=True):
    # Ensure we don't add the reverse edge if it represents the same relationship type visually
    # (This depends on whether you added symmetric edges earlier)
    # edge_tuple = tuple(sorted((u, v))) + (data.get('label'),) # Consider label for uniqueness
    # if edge_tuple in added_edges:
    #    continue
    # added_edges.add(edge_tuple)

    edge_label = data.get("label", "")
    edge_title = data.get("title", edge_label) # 鼠标悬停时显示的文字
    net.add_edge(u, v, label=edge_label, title=edge_title, arrows="to")


# 配置物理引擎和交互选项
net.set_options("""
var options = {
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.01,
      "springLength": 100,
      "springConstant": 0.08,
      "damping": 0.4,
      "avoidOverlap": 0.5
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based",
    "stabilization": {
      "enabled": true,
      "iterations": 1000,
      "updateInterval": 50
    }
  },
  "interaction": {
    "tooltipDelay": 200,
    "hideEdgesOnDrag": true,
    "navigationButtons": true,
    "keyboard": true
  },
  "edges": {
    "smooth": {
       "enabled": true,
       "type": "dynamic",
       "roundness": 0.5
     }
  }
}
""")


# 保存并尝试打开HTML文件
output_filename = "urban_knowledge_graph.html"
try:
    net.save_graph(output_filename)
    print(f"图谱已保存为 {output_filename}")
    # 尝试在默认浏览器中打开
    webbrowser.open('file://' + os.path.realpath(output_filename))
except Exception as e:
    print(f"无法自动打开HTML文件: {e}")
    print(f"请手动在浏览器中打开文件: {os.path.realpath(output_filename)}")


# --- 4. 查询示例 ---
print("\n--- 图谱查询示例 ---")

# 查询1: 找到所有类型为 "Landmark" 的节点
landmark_nodes_found = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Landmark"]
print(f"\n查询1: 所有地标节点:")
for node_id in landmark_nodes_found:
    print(f"- {node_id} (名称: {G.nodes[node_id].get('label', 'N/A')})")

# 查询2: 找到所有可以看到 "Landmark_塔楼B" 的窗户
target_landmark = "Landmark_塔楼B"
windows_seeing_target = []
if G.has_node(target_landmark):
    # G.predecessors(node) 返回所有指向 node 的节点
    # 在有向图中，如果边是 Window -> Landmark (visibility)，则前驱是 Window
    # Use G.predecessors to find nodes pointing TO the landmark
    possible_viewers = list(G.predecessors(target_landmark))
    # Filter to ensure they are Windows and the edge label is 'visibility'
    windows_seeing_target = [
        u for u in possible_viewers
        if G.has_node(u) and G.nodes[u].get("node_type") == "Window" and G.has_edge(u, target_landmark) and G[u][target_landmark].get("label") == "visibility"
    ]

print(f"\n查询2: 可以看到 '{G.nodes[target_landmark].get('label', target_landmark)}' 的窗户:")
if windows_seeing_target:
    for window_id in windows_seeing_target:
        print(f"- {window_id} ({G.nodes[window_id].get('label', '')})")
else:
    print(f"- 没有窗户可以看到该地标。")

# 查询3: 找到绿视率为 "High" 的窗户
high_gvr_windows = [
    n for n, d in G.nodes(data=True)
    if d.get("node_type") == "Window" and d.get("green_view_ratio") == "High"
]
print(f"\n查询3: 绿视率为 'High' 的窗户:")
if high_gvr_windows:
    for window_id in high_gvr_windows:
        print(f"- {window_id} ({G.nodes[window_id].get('label', '')})")
else:
    print("- 没有找到高绿视率的窗户。")


# 查询4: 找到 "Window_1" 的邻居节点及其关系
target_window = "Window_1"
print(f"\n查询4: '{target_window}' 的直接关联节点:")
if G.has_node(target_window):
    # G.adj[node] or G.succ[node] gives successors (outgoing edges)
    print(f"  出向关系 (从 {target_window} 指向 ->):")
    # Check if target_window has successors
    if target_window in G.adj:
        for neighbor, edge_data_dict in G.adj[target_window].items():
             # For DiGraph, edge_data_dict is just the attribute dict for the single edge
             edge_attributes = edge_data_dict
             # Ensure neighbor node exists before accessing its attributes
             if G.has_node(neighbor):
                 print(f"  - 指向 -> {neighbor} ({G.nodes[neighbor].get('label', '')}), 关系: {edge_attributes.get('label', 'N/A')}")
             else:
                 print(f"  - 指向 -> {neighbor} (节点数据缺失), 关系: {edge_attributes.get('label', 'N/A')}")

    else:
        print("  - 无出向关系。")


    # G.pred[node] gives predecessors (incoming edges)
    print(f"\n  入向关系 (指向 -> {target_window}):")
    # Check if target_window has predecessors
    if target_window in G.pred:
        for predecessor, edge_data_dict in G.pred[target_window].items():
            # For DiGraph, edge_data_dict is just the attribute dict for the single edge
            edge_attributes = edge_data_dict
            # Ensure predecessor node exists before accessing its attributes
            if G.has_node(predecessor):
                 print(f"  - 来自 <- {predecessor} ({G.nodes[predecessor].get('label', '')}), 关系: {edge_attributes.get('label', 'N/A')}")
            else:
                 print(f"  - 来自 <- {predecessor} (节点数据缺失), 关系: {edge_attributes.get('label', 'N/A')}")
    else:
        print("  - 无入向关系。")


else:
    print(f"- 节点 '{target_window}' 不存在。")

# 查询5: 查找两个节点之间的路径 (例如，Window_1 到 Landmark_公园A)
start_node = "Window_1"
end_node = "Landmark_公园A"
print(f"\n查询5: 查找从 '{start_node}' 到 '{end_node}' 的路径:")
if G.has_node(start_node) and G.has_node(end_node):
    try:
        # 查找所有简单路径 (无环)
        paths = list(nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=5)) # cutoff 限制路径长度
        if paths:
            print(f"- 找到 {len(paths)} 条路径:")
            for i, path in enumerate(paths):
                path_labels = [G.nodes[n].get('label', n) for n in path] # 获取路径上节点的标签
                print(f"  路径 {i+1}: {' -> '.join(path_labels)}")
        else:
            print(f"- 未找到从 '{G.nodes[start_node].get('label', start_node)}' 到 '{G.nodes[end_node].get('label', end_node)}' 的路径。")
    except nx.NetworkXNoPath:
        print(f"- 未找到从 '{G.nodes[start_node].get('label', start_node)}' 到 '{G.nodes[end_node].get('label', end_node)}' 的路径。")
    except Exception as e:
        print(f"- 查找路径时出错: {e}")
else:
    print(f"- 起始节点 '{start_node}' 或结束节点 '{end_node}' 不存在。")


# --- 5. 模拟基于图像的查询 ---
print("\n--- 模拟基于图像的查询 ---")

# 5.1 模拟: 基于图像内容/视角匹配查找相似窗户
print("\n查询 5.1: 模拟通过图像向量查找最相似的窗户视角")

# 模拟上传的图像并提取其向量特征 (这里用一个随机值代替)
simulated_image_vector_value = random.uniform(0, 1)
print(f"模拟上传图像的特征值: {simulated_image_vector_value:.4f}")

min_distance = float('inf')
most_similar_windows = []
window_distances = []

# 遍历所有窗户节点
for window_node, data in G.nodes(data=True):
    if data.get("node_type") == "Window":
        # 找到该窗户关联的向量节点
        vector_node = None
        for successor in G.successors(window_node):
             if G.has_node(successor) and G.nodes[successor].get("node_type") == "CLIPVector" and G[window_node][successor].get("label") == "has_vector":
                vector_node = successor
                break

        if vector_node and G.has_node(vector_node) and "simulated_value" in G.nodes[vector_node]:
            window_vector_value = G.nodes[vector_node]["simulated_value"]
            # 计算模拟距离 (这里仅使用差值的绝对值作为示例)
            # !! 注意：实际应用中应使用余弦相似度或欧氏距离等
            distance = abs(simulated_image_vector_value - window_vector_value)
            window_distances.append((window_node, distance))

            # 更新最小距离和最相似窗户列表
            # if distance < min_distance:
            #     min_distance = distance
            #     most_similar_windows = [window_node]
            # elif distance == min_distance: # 处理距离相同的情况
            #     most_similar_windows.append(window_node)

# 基于距离排序，找到最接近的几个
window_distances.sort(key=lambda item: item[1])

# 输出最相似的窗户 (例如前 3 个)
print("\n根据模拟向量距离找到的最相似窗户 (距离越小越相似):")
if window_distances:
    for i in range(min(3, len(window_distances))): # 最多显示3个
        win_id, dist = window_distances[i]
        print(f"- {win_id} ({G.nodes[win_id].get('label', '')}), 模拟距离: {dist:.4f}")
else:
    print("- 未找到带有向量信息的窗户节点。")


# 5.2 模拟: 识别图像中的地标，并查找能看到该地标的窗户
print("\n查询 5.2: 模拟识别图像中的地标，并查找能看到该地标的窗户")

# 模拟从图像中识别出的地标 (随机选择一个已存在地标)
if landmark_nodes: #确保有地标节点
    identified_landmark_id = random.choice(landmark_nodes)
    identified_landmark_label = G.nodes[identified_landmark_id].get('label', identified_landmark_id)
    print(f"模拟从图像中识别出的地标: {identified_landmark_label} ({identified_landmark_id})")

    # 执行查询: 查找可以看到此地标的窗户 (复用查询2的逻辑)
    windows_seeing_identified_landmark = []
    if G.has_node(identified_landmark_id):
        possible_viewers = list(G.predecessors(identified_landmark_id))
        windows_seeing_identified_landmark = [
            u for u in possible_viewers
            if G.has_node(u) and G.nodes[u].get("node_type") == "Window" and G.has_edge(u, identified_landmark_id) and G[u][identified_landmark_id].get("label") == "visibility"
        ]

    print(f"\n找到可以看到 '{identified_landmark_label}' 的窗户:")
    if windows_seeing_identified_landmark:
        for window_id in windows_seeing_identified_landmark:
            print(f"- {window_id} ({G.nodes[window_id].get('label', '')})")
    else:
        print(f"- 没有窗户记录可以看到地标 '{identified_landmark_label}'。")
else:
    print("- 图谱中没有地标节点可供模拟识别。")

