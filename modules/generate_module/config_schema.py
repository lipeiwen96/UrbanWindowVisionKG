import math
from dataclasses import dataclass
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Params:
	site_classification: str = "A"
	min_bulding_height: float = 55.00
	max_bulding_height: float = 61.00


@dataclass_json
@dataclass
class OZPGenerationConfig:
	site_area: float = 30000
	site_classification: str = "A"
	min_bulding_height: float = 55.00
	max_bulding_height: float = 61.00
	# OZP Mode
	permitted_d_plot_ratio: float = 3.00
	permitted_nd_plot_ratio: float = 4.00
	permitted_total_plot_ratio: float = 7.00
	permitted_d_site_coverage: float = 45.00
	permitted_nd_site_coverage: float = 20.00


@dataclass_json
@dataclass
class BPRGenerationConfig:
	site_area: float = 30000
	site_classification: str = "A"
	min_bulding_height: float = 55.00
	max_bulding_height: float = 55.00
	# OZP Mode
	permitted_d_plot_ratio: float = 0.00
	permitted_nd_plot_ratio: float = 0.00
	actual_d_plot_ratio: float = 0.00
	actual_nd_plot_ratio: float = 0.00
	actual_total_plot_ratio: float = 0.00
	permitted_d_site_coverage: float = 45.00
	permitted_nd_site_coverage: float = 20.00


@dataclass_json
@dataclass
class BuildingConfig:
	choose_plan_name: str = '4FLAT-AEROPLANE'
	first_floor_area: float = 230
	floor_d_height: float = 3.15
	floor_nd_height: float = 5
	nd_storey: int = 2
	min_tower_seperation: float = 5.00
	# min_building_height: float = 55.00
	# max_building_height: float = 55.00
	
	# @property
	# def min_building_floor_number(self):
	# 	return math.ceil((self.min_building_height - (self.floor_nd_height *
	# 	                                              self.nd_storey)) /
	# 	                 self.floor_d_height)
	#
	# @property
	# def max_building_floor_number(self):
	# 	return math.floor((self.max_building_height - (self.floor_nd_height *
	# 	                                               self.nd_storey)) /
	# 	                  self.floor_d_height)