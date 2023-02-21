"""
This module holds all the logic for running the full execution flow as separate run stages, which enables using the
results of the stages that were already executed before, without rerunning them, or executing the flow only up to a
certain point
"""

import os
from collections import OrderedDict
from configuration.configuration_manager import Configuration

from run_stages.resistive_mesh_stage import ResistiveMeshStage
from run_stages.current_sequence_stage import CurrentSequenceStage
from run_stages.circuit_stage import CircuitStage
from run_stages.simulation_stage import SimulationStage
from run_stages.plot_results_stage import PlotResultsStage
from utilities.common_utilities import CommonUtils


class NeededOutputNotFound(Exception):
	"""
	This class is used when the needed output is not available
	"""
	pass


class RunStageManager:
	"""
	This class handles all the logic concerning the separation of the full execution flow into individual run stages
	"""

	def __init__(self, output_directory, run_stages):
		"""
		This function initializes all the parameters for the handling of requested run stages
		:param output_directory: the output directory for this run
		:param run_stages: the run stages requested by the user
		"""
		self.output_directory = output_directory
		self._all_stages = OrderedDict({
			"resistive_mesh": {"constructor": ResistiveMeshStage, "output": list(), "output_file_name": [Configuration().params["r_matrix_output_file"]]},
			"current_sequence": {"constructor": CurrentSequenceStage, "output": list(), "output_file_name": [os.path.join(Configuration().params["video_sequence_name"],"video_sequence.pkl")]},
			"circuit": {"constructor": CircuitStage, "output": list(), "output_file_name": [Configuration().params["netlist_output_file"]]},
			"simulation": {"constructor": SimulationStage, "output": list(), "output_file_name": ["simulation_results.pkl"]},
			"plot_results": {"constructor": PlotResultsStage, "output": list(), "output_file_name": ["diode_voltage_vs_time.png", "current_vs_time.png"]}
		})

		# if none are provided, run all stages
		self.run_stages = list()
		run_stages = run_stages if run_stages else self.get_all_stage_names()
		if isinstance(run_stages, str):
			self.run_stages.append(run_stages)
		else:
			self.run_stages.extend(run_stages)

	def get_all_stage_names(self):
		"""
		This function returns a list of all available run stages
		:return:
		"""
		return self._all_stages.keys()

	def get_run_stages(self):
		"""
		This function returns a list of all requested run stages
		:return: list of run stages
		"""
		return self.run_stages

	def initialize_stage(self, stage):
		"""
		This function initializes the requested stage
		:param stage: the name of the stage to be initialized
		:return:
		"""
		return self._all_stages[stage]["constructor"](self.output_directory, self.get_stages_output)

	def get_stages_number(self, stage):
		"""
		This function returns the number of the stage in the run stages sequence based on its name
		:param stage: the name of the stage
		:return: the number of this stage in the sequence
		"""
		return list(self._all_stages.keys()).index(stage)

	def get_stage_by_number(self, index):
		"""
		This function returns the name of the stage at the given number
		:param index: the number in the list of run stages to retrieve
		:return: corresponding stage name
		"""
		return list(self._all_stages.keys())[index]

	def update_stages_output(self, stage, output):
		"""
		This function sets the given run stage with the given output
		:param stage: the stage to update
		:param output: the output to update
		:return:
		"""
		self._all_stages[stage]["output"].extend(output)

	def get_stages_output(self, stage, index=None):
		"""
		This function returns all or one of the inputs of the requested stage
		:param stage: the name of the stage to retrieve
		:param index: the index of the output to retrieve (for stages with more than one output)
		:return: the corresponding output
		"""
		return self._all_stages[stage]["output"] if index is None else self._all_stages[stage]["output"][index]

	def get_stages_output_file_name(self, stage, index=None):
		"""
		This function returns the output file name for the requested run stage
		:param stage: the requested stage
		:param index: the index of the output to retrieve (for stages with more than one output)
		:return:
		"""
		return self._all_stages[stage]["output_file_name"] if index is None else self._all_stages[stage]["output_file_name"][index]

	def initialize_missing_outputs(self, identical_configurations):
		"""
		This function tries to find the needed outputs for all the run stages that the user requested to skip in this run
		using previous run folders, if available
		:param identical_configurations: a list of paths with previous identical runs
		:return:
		"""
		# if the current execution doesn't include all possible run stages
		if self.run_stages != self.get_all_stage_names():

			# check if we have previous runs to rely on, if not we have a problem
			if not identical_configurations:
				raise NeededOutputNotFound("Current run relies on previous executions, but no such executions were "
										"found. Please rerun full flow.")

			# if we do, let's check if these previous runs contain the needed outputs
			first_stage_number = self.get_stages_number(self.run_stages[0])
			if first_stage_number > 0:
				skipped_stage = self.get_stage_by_number(first_stage_number - 1)
				missing_output = self.find_missing_outputs(skipped_stage, identical_configurations)
				self._all_stages[skipped_stage]["output"].append(CommonUtils.load_output(missing_output))

	def find_missing_outputs(self, skipped_stage, identical_configurations):
		"""
		This function tries to find the output of the skipped run stage in one of the previous run folders
		:param skipped_stage: the stage the output of which to search for
		:param identical_configurations: a list of paths with identical runs
		:return: a path with an existing output for the requested stage, or an exception if none was found
		"""
		# if the output is not available in the current run, search for it in previous identical runs
		if not self.get_stages_output(skipped_stage):
			for directory in identical_configurations:
				for root, _, filenames in os.walk(directory):
					for filename in filenames:
						if filename == self.get_stages_output_file_name(skipped_stage, 0):
							return os.path.join(root, filename)

			raise NeededOutputNotFound("Couldn't find required files in any of the previous runs, please run needed stages")

