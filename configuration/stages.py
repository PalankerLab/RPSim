from enum import Enum


class UnsupportedStage(Exception):
	"""
	This class is used when the requested stage is not supported
	"""
	pass


class RunStages(Enum):
	pattern_generation = ("pattern_generation", 0, False)
	multiplexing = ("multiplexing", 1, False)
	resistive_mesh = ("resistive_mesh", 2)
	current_sequence = ("current_sequence", 3)
	circuit = ("circuit", 4)
	simulation = ("simulation", 5)
	post_process = ("post_process", 6, False)
	plot_results = ("plot_results", 7, False)

	def __new__(cls, value, number, mandatory=True):
		run_stage = object.__new__(cls)
		run_stage._value_ = value
		run_stage.number = number
		run_stage.mandatory = mandatory
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

	@staticmethod
	def get_mandatory_stages():
		return [run_stage.name for run_stage in RunStages if run_stage.mandatory == True]



