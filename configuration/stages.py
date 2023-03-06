from enum import Enum


class UnsupportedStage(Exception):
	"""
	This class is used when the requested stage is not supported
	"""
	pass


class RunStages(Enum):
	resistive_mesh = ("resistive_mesh", 0)
	current_sequence = ("current_sequence", 1)
	circuit = ("circuit", 2)
	simulation = ("simulation", 3)
	post_process = ("post_process", 4)
	plot_results = ("plot_results", 5)

	def __new__(cls, value, number):
		run_stage = object.__new__(cls)
		run_stage._value_ = value
		run_stage.number = number
		return run_stage


class StageManager:
	@staticmethod
	def get_all_available_run_stages():
		"""
		This function returns a list of all available run stages
		:return:
		"""
		return [stage.name for stage in RunStages]

	@staticmethod
	def get_stage_number_by_name(stage):
		"""
		This function returns the number of the stage in the run stages sequence based on its name
		:param stage: the name of the stage
		:return: the number of this stage in the sequence
		"""
		match = [run_stage.number for run_stage in RunStages if run_stage.name == stage]
		if match and len(match) == 1:
			return match[0]
		raise UnsupportedStage("The requested stage {} is not supported".format(stage))

	@staticmethod
	def get_stage_name_by_number(index):
		"""
		This function returns the name of the stage at the given number
		:param index: the number in the list of run stages to retrieve
		:return: corresponding stage name
		"""
		return [run_stage.name for run_stage in RunStages if run_stage.number == index][0]


