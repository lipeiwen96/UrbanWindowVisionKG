"""
SolutionGenerator
在调用设计云算法的输入及输出端进行封装
"""
import multiprocessing
import os
import traceback
from dataclasses import dataclass
from dataclasses import field
from typing import List

from fastapi.logger import logger
from shapely.affinity import affine_transform
from shapely.geometry import Point
from shapely.geometry import Polygon

from algorithm_module.estimator.building_count_calculator import \
    BuildingCountCalculator
from algorithm_module.generator.multiprocessing_generator import single_process_generate_plan_series
from algorithm_module.generator.plan_greedy_generator import GreedyPlanGenerator
from algorithm_module.model.plan import Plan
from algorithm_module.util.multiprocessing_pool import my_pool_size
from algorithm_module.util.time import time_util
from art_public_modules.art_data_exchange.data_exchange import \
    ARTShapelyDataExchanger
from modules.generate_module.config_schema import BPRGenerationConfig
from modules.generate_module.config_schema import BuildingConfig
from modules.generate_module.config_schema import OZPGenerationConfig
from modules.regulation_module.building_structure import BuildingStructure
from modules.regulation_module.building_structure import BuildingTemplate
from modules.regulation_module.building_structure import SiteStructure


@dataclass(order=True)
class SolutionGenerator:
    project_name: str = field(default=str)
    ozp_config: OZPGenerationConfig = field(default_factory=OZPGenerationConfig)
    bpr_config: BPRGenerationConfig = field(default_factory=BPRGenerationConfig)
    use_ozp: bool = field(default=True)
    building_config: BuildingConfig = field(default_factory=BuildingConfig)
    # 地块
    site: SiteStructure = field(default_factory=SiteStructure)
    # 建筑
    building: BuildingStructure = field(default_factory=BuildingStructure)
    building_template: BuildingTemplate = None

    def setup(self, project_name: str, use_ozp: bool,
              ozp_config: OZPGenerationConfig, bpr_config: BPRGenerationConfig,
              building_config: BuildingConfig) -> 'SolutionGenerator':
        self.project_name = project_name
        self.use_ozp = use_ozp
        self.building_config = building_config
        if use_ozp:
            self.ozp_config = ozp_config
            building_config.min_building_height = ozp_config.min_bulding_height
            building_config.max_building_height = ozp_config.max_bulding_height
        else:
            self.bpr_config = bpr_config
            building_config.min_building_height = bpr_config.min_bulding_height
            building_config.max_building_height = bpr_config.max_bulding_height

        self.__load_site()
        self.__load_building_plan()
        return self

    # 创建场地信息
    # mode2如下
    # TODO： Mode0/1 重新构造
    def __load_site(self):
        # 读入场地地块
        site_model = ARTShapelyDataExchanger.read_dxf_file(self.project_name)
        # 构建site数据结构
        self.site = SiteStructure()
        # 输入结构
        # 【简化版的结构，为了计算，只保留了场地轮廓、内部障碍区域（list）】
        for ele in site_model.elements:
            if ele.layer == "Xkool_SiteBoundary":
                # 地块轮廓
                if isinstance(ele.geometry, Polygon):
                    self.site.site_geometry = ele.geometry
            elif ele.layer == "Xkool_FacingStreet_4.5m" or ele.layer == "Xkool_FacingStreet_Smaller4.5m":
                # 非4.5m道路的检测线
                self.site.normal_road_line_list.append(ele.geometry)
            elif ele.layer == "Xkool_SiteUnbuildableRegion":  # 算法已兼容， TODO: Mode0/1未接入
                # 不可排布的建筑区域
                if isinstance(ele.geometry, Polygon):
                    self.site.inside_unplaced_area_list.append(ele.geometry)
        # 更新site指标
        if self.use_ozp:
            self.site.site_area = self.ozp_config.site_area
            self.site.site_plot_ratio = self.ozp_config.permitted_total_plot_ratio
            self.site.site_site_coverage = self.ozp_config.permitted_d_site_coverage / 100
            self.site.input_dpr = self.ozp_config.permitted_d_plot_ratio
            self.site.input_ndpr = self.ozp_config.permitted_nd_plot_ratio
            self.site.input_dsc = self.ozp_config.permitted_d_site_coverage
            self.site.input_ndsc = self.ozp_config.permitted_nd_site_coverage
        else:
            self.site.site_area = self.bpr_config.site_area
            self.site.input_dpr = self.bpr_config.actual_d_plot_ratio
            self.site.input_ndpr = self.bpr_config.actual_nd_plot_ratio
            self.site.site_plot_ratio = self.bpr_config.actual_total_plot_ratio
            self.site.site_site_coverage = self.bpr_config.permitted_d_site_coverage / 100
            self.site.input_dsc = self.bpr_config.permitted_d_site_coverage
            self.site.input_ndsc = self.bpr_config.permitted_nd_site_coverage

    def __load_building_plan(self):
        # 楼型Library
        # 读入建筑楼型
        file_target = os.path.abspath(
            os.path.join("library", "building_plan_dxf",
                         self.building_config.choose_plan_name + ".dxf"))
        data_model = ARTShapelyDataExchanger.read_dxf_file(file_target)
        # 预处理, 模型移动至原点
        centroid = Point()
        for ele in data_model.elements:
            if ele.layer == "Xkool_BuildingOutline":
                centroid = ele.geometry.centroid
                break
        for ele in data_model.elements:
            ele.geometry = affine_transform(ele.geometry,
                                            [1, 0, 0, 1, -centroid.x,
                                             -centroid.y])
        # 生成建筑数据结构
        outline = None
        habitat_window_list = []
        others_window_list = []
        for ele in data_model.elements:
            if ele.layer == "Xkool_BuildingOutline":
                outline = ele.geometry
            elif ele.layer == "Xkool_WindowHabitat":
                habitat_window_list.append(ele.geometry)
            elif ele.layer == "Xkool_WindowOthers":
                others_window_list.append(ele.geometry)

        # 初始化BuildingStructure！！！！
        self.building = BuildingStructure(building_name=self.building_config.choose_plan_name)
        # 更新building_template指标
        if self.use_ozp:
            self.building.min_building_height = self.ozp_config.min_bulding_height
            self.building.max_building_height = self.ozp_config.max_bulding_height
        else:
            self.building.min_building_height = self.bpr_config.min_bulding_height
            self.building.max_building_height = self.bpr_config.max_bulding_height
        self.building.first_floor_area = self.building_config.first_floor_area
        self.building.min_tower_distance = self.building_config.min_tower_seperation
        self.building.non_domestic_floor_height = self.building_config.floor_nd_height
        self.building.non_domestic_storey = self.building_config.nd_storey
        self.building.domestic_floor_height = self.building_config.floor_d_height
        # 初始化
        self.building.inititialize_building(building_geometry=outline,
                                            habitat_window_list=habitat_window_list,
                                            others_window_list=others_window_list,
                                            site=self.site)
        # 这里有个template
        # TODO: 先不管
        self.building_template = BuildingTemplate(building_name=self.building_config.choose_plan_name,
                                                  building_geometry=outline,
                                                  first_floor_area=self.building_config.first_floor_area,
                                                  habitat_window_list=habitat_window_list,
                                                  others_window_list=others_window_list,
                                                  site=self.site,
                                                  building_config=self.building_config)

    # 生成算法主入口
    @staticmethod
    def generate_single_plan(shared_finished_worker_count: multiprocessing.Value,
                             global_best_plan_json: multiprocessing.Value,
                             total_main_plan_count: int,
                             site, building_template, building_structure, derived_count: int = 0, max_iter: int = 3000) -> List['Plan']:
        try:
            calculator = BuildingCountCalculator(site, building_template, building_structure)
            building_generation_settings = calculator.calculate_building_count_and_floor_num()  # 包含了所有的参数
            # 最高楼层数
            max_domestic_floor_num = max([setting.domestic_floor_num for setting in building_generation_settings])
            print(f"开始排布方案，根据输入参数测算，本次排布至多能排【{len(building_generation_settings)}栋楼】,"
                  f"每栋楼底商【{building_generation_settings[0].non_domestic_floor_num}层】+住宅层最高【{max_domestic_floor_num}层】,"
                  f"预期指标: TotalPR-【{round(building_structure.site.site_plot_ratio, 2)}: {round(building_structure.site.input_dpr, 2)}(D)+{round(building_structure.site.input_ndpr, 2)}(ND)】 SC-【{round(building_structure.site.site_site_coverage, 2)}】"
                  f"场地面积-【{round(building_structure.site.site_area)}】 GFA-【{round(building_structure.site.site_area * building_structure.site.site_plot_ratio)}】")
            generator = GreedyPlanGenerator(site=site,
                                            building_template=building_template,
                                            building_structure=building_structure,
                                            building_generation_settings=building_generation_settings,
                                            max_iter=max_iter)
            main_plan = generator.generate_plan()
            logger.info(f"主方案生成完毕，总共有{len(main_plan.buildings)}个建筑")
            logger.info(f"global best plan name is {global_best_plan_json['name']}")
            global_best_plan = Plan.from_json(global_best_plan_json)
            if len(main_plan.buildings) > len(global_best_plan.buildings):
                data = main_plan.to_json()
                global_best_plan_json["name"] = data["name"]
                global_best_plan_json["buildings"] = data["buildings"]
                global_best_plan = main_plan
                logger.info(f"全局最优方案现在是{len(global_best_plan.buildings)}栋")
            all_plans = [main_plan]
            # 一开始用原始方案作为基础
            global_best_plan = Plan.from_json(global_best_plan_json)
            base_plan = global_best_plan
            # 如果有派生数量，继续派生
            for i in range(derived_count):
                # 一旦有50%的主任务已经结束，其他的主任务也允许提早结束
                continue_generate = shared_finished_worker_count.value * 1.0 / total_main_plan_count <= 0.7
                logger.info(
                    f"开始派生方案，当前顺序为{i}, 当前全局flag是{continue_generate}，已完成线程数为{shared_finished_worker_count.value}/{total_main_plan_count}")
                if continue_generate:
                    # 尚不足30%的worker休息，还可以继续来一个方案
                    new_generator = generator.get_another_generator_by_plan(global_best_plan)
                    new_plan = new_generator.generate_plan()
                    new_plan.parent_id = global_best_plan.id
                    logger.info(
                        f"衍生方案生成完成, "
                        f"新方案是{len(new_plan.buildings)}栋{new_plan.total_domestic_floor_num}层，"
                        f"原方案是{len(base_plan.buildings)}栋{base_plan.total_domestic_floor_num}层")
                    # 这时候做一个判断，如果新生成的方案，比原始方案更好，则新方案作为基础方案
                    if new_plan.total_domestic_floor_num > base_plan.total_domestic_floor_num:
                        logger.info(f"新方案比原方案更好，新方案是{new_plan.total_domestic_floor_num}，原方案是{base_plan.total_domestic_floor_num}，做一个替换")
                        base_plan = new_plan
                    global_best_plan = Plan.from_json(global_best_plan_json)
                    if len(new_plan.buildings) > len(global_best_plan.buildings):
                        data = new_plan.to_json()
                        global_best_plan_json["name"] = data["name"]
                        global_best_plan_json["buildings"] = data["buildings"]
                        logger.info("正在更新全局最优方案栋数")
                    logger.info(f"已生成方案是{len(new_plan.buildings)}栋, 全局最优方案现在是{len(global_best_plan.buildings)}栋")
                    all_plans.append(new_plan)
                else:
                    # 否则停止
                    logger.info("全局flag已关闭，无需继续生成")
            for i, plan in enumerate(all_plans):
                logger.info(f"plan {i} 排布完成, 建筑数量{len(plan.buildings)}, "
                            f"建筑住宅总层数{sum(b.domestic_storey for b in plan.buildings)}, "
                            f"目标值是{sum([s.domestic_floor_num for s in building_generation_settings])}")
            logger.info(f"线程已结束，修改已完成线程数，当前已结束数量为{shared_finished_worker_count.value}")
            shared_finished_worker_count.value = shared_finished_worker_count.value + 1
            return all_plans
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())
            return []

    @time_util.log_time
    def generate_plans(self, process_pool: multiprocessing.Pool, main_plan_count: int = 8, derived_plan_count: int = 0, max_iter: int = 3000):
        """
        返回总方案数量 = main_plan_count * (derived_plan_count + 1)
        """
        # No need to partial here, as generate_single_plan is a static method
        with multiprocessing.Manager() as manager:
            finished_worker_count = manager.Value('i', 0)
            global_best_plan_json = manager.dict()
            global_best_plan_json["name"] = "default"
            global_best_plan_json["building_template"] = self.building_template.to_json()
            global_best_plan_json["buildings"] = []
            global_best_plan_json["site"] = self.site.to_json()
            list_of_plans = process_pool.starmap(self.generate_single_plan,
                                                 [(finished_worker_count,
                                                   global_best_plan_json,
                                                   # main plan数量是衡量是否结束的原因
                                                   main_plan_count,
                                                   self.site,
                                                   self.building_template,
                                                   self.building,
                                                   derived_plan_count,
                                                   max_iter) for _
                                                  in range(main_plan_count)])
            total_plans = []
            for plans in list_of_plans:
                total_plans.extend(plans)
            return total_plans

    @time_util.log_time
    def generate_plans_v2(self, task_id: str, main_plan_count: int = 8, derived_plan_count: int = 0, max_iter: int = 3000,
                          is_local_test: bool = False):
        """
        返回总方案数量 = main_plan_count * (derived_plan_count + 1)
        """
        # No need to partial here, as generate_single_plan is a static method
        if not is_local_test:
            # initialize the worker process
            with multiprocessing.Pool(my_pool_size) as process_pool:
                list_of_plans = process_pool.starmap(single_process_generate_plan_series,
                                                     [(task_id,
                                                       self.site,
                                                       self.building_template,
                                                       self.building,
                                                       derived_plan_count,
                                                       max_iter) for _
                                                      in range(main_plan_count)])
                total_plans = []
                for plans in list_of_plans:
                    total_plans.extend(plans)
                return total_plans
        else:
            total_plans = []
            for _ in range(main_plan_count):
                print(_, 'of', main_plan_count)
                list_of_plans = single_process_generate_plan_series(task_id, self.site, self.building_template, self.building,
                                                                    derived_plan_count, max_iter, is_local_test=True)
                total_plans.extend(list_of_plans)
            return total_plans
