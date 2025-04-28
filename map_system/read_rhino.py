from art_public_modules.art_data_structure.shapely.core_structure import DataModel, DataElement # 自定义数据模型和元素结构 (可能用于可视化)
from art_public_modules.art_data_structure.shapely.data_element_material import DataElementMaterial # 自定义数据元素的材质信息
from art_public_modules.art_data_exchange.data_exchange import ARTShapelyDataExchanger

file_path = r"E:\Code\HITSZ\UrbanWindowVisionKG\library\HK_map\processed\project\project_simplified_only_crv.3dm"
dm = ARTShapelyDataExchanger.read_rhino_file(file_path)
print(len(dm.elements))
# for ele in dm.elements:
#     print(f"图层：{ele.layer}, 面积：{round(ele.geometry.area, 2)}, 高度: {round(ele.height, 2)}")

json_path = r"E:\Code\HITSZ\UrbanWindowVisionKG\library\HK_map\processed\project\project_simplified_only_crv.json"
dm = ARTShapelyDataExchanger.write_json_file(json_path, dm)
