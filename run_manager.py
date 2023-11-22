"""
This module holds all the logic for running the full execution flow as separate run stages, which enables using the
results of the stages that were already executed before, without rerunning them, or executing the flow only up to a
certain point
"""

import os
import shutil
from collections import defaultdict, Counter
from pathlib import Path

from configuration.configuration_manager import Configuration
from configuration.stages import RunStages, StageManager
from utilities.common_utilities import CommonUtils
from utilities.exceptions import NeededOutputNotFound

from run_stages.circuit_stage import CircuitStage
from run_stages.pattern_generation_stage import PatternGenerationStage
from run_stages.current_sequence_stage import CurrentSequenceStage
from run_stages.plot_results_stage import PlotResultsStage
from run_stages.post_process_stage import PostProcessStage
from run_stages.resistive_mesh_stage import ResistiveMeshStage
from run_stages.simulation_stage import SimulationStage



class RunManager:
	"""
	This class handles all the logic concerning the separation of the full execution flow into individual run stages
	"""

	def __init__(self, output_directory, run_stages, skip_stages):
		"""
		This function initializes all the parameters for the handling of requested run stages
		:param output_directory: the output directory for this run
		:param run_stages: the run stages requested by the user
		"""
		self.output_directory = output_directory
		self.outputs_container = defaultdict(list)

		# if none are provided, run all stages
		self.run_stages = list()
		skip_stages = skip_stages if skip_stages else list()
		run_stages = run_stages if run_stages else [stage for stage in StageManager.get_all_available_run_stages() if
													stage not in skip_stages]
		if isinstance(run_stages, str):
			self.run_stages.append(run_stages)
		else:
			self.run_stages.extend(run_stages)

		# make sure we do not have any excess spaces
		self.run_stages = list(map(str.strip, self.run_stages))

		# define non-mandatory stages
		self.mandatory_stages = StageManager.get_mandatory_stages()

		# check whether to look for previous runs
		self.find_previous_runs = self._should_look_for_previous_runs(self.run_stages)

	@staticmethod
	def _stage_data_factory(stage):
		if stage == RunStages.pattern_generation.name: 
			stage_data = PatternGenerationStage, ["list_ndarray_images.pkl", "script.csv", "dict_PIL_images.bmp", "proj_seq_script.py"], Configuration().params["video_sequence_name"]		
		elif stage == RunStages.resistive_mesh.name:
			stage_data = ResistiveMeshStage, [Configuration().params["r_matrix_output_file"]], ""
		elif stage == RunStages.current_sequence.name:
			stage_data = CurrentSequenceStage, ["video_sequence.pkl", "video_sequence.gif"], Configuration().params["video_sequence_name"]
		elif stage == RunStages.circuit.name:
			stage_data = CircuitStage, ["netlist.sp"], ""
		elif stage == RunStages.simulation.name:
			stage_data = SimulationStage, ["simulation_results.pkl"], ""
		elif stage == RunStages.post_process.name:
			stage_data = PostProcessStage, ["4D_potential_matrix_with_time_and_coordinates.pkl"], RunStages.post_process.name
		elif stage == RunStages.plot_results.name:
			stage_data =  PlotResultsStage, ["diode_voltage_vs_time.png", "current_vs_time.png", "on_diode_pulse.png"], ""
		else:
			raise KeyError("The requested run stage is not supported")

		return dict(zip(["constructor", "output_file_names", "output_directory_name"], stage_data))

	def _should_look_for_previous_runs(self, requested_run_stages):
		# check if number of requested stages is smaller than the total available number of stages
		if requested_run_stages and Counter(requested_run_stages) != Counter(StageManager.get_all_available_run_stages()):
			# if so, check if we are missing necessary stages
			if not all(stage in requested_run_stages for stage in self.mandatory_stages):
				return True

		return False

	def initialize_stage(self, stage):
		stage_data = self._stage_data_factory(stage)
		return stage_data["constructor"](self.output_directory, self.outputs_container,
										 stage_data["output_file_names"], stage_data["output_directory_name"])

	def get_output_for_all_stages(self):
		return self.outputs_container

	def get_stage_output(self, stage, index=None):
		"""
		This function returns all or one of the inputs of the requested stage
		:param stage: the name of the stage to retrieve
		:param index: the index of the output to retrieve (for stages with more than one output)
		:return: the corresponding output
		"""
		return self.outputs_container[stage] if index is None else self.outputs_container[stage][index]

	def get_stage_output_file_names(self, stage, index=None):
		output_file_names =  self._stage_data_factory(stage)["output_file_names"]
		return output_file_names if not index else output_file_names[index]

	def get_stage_output_directory_name(self, stage):
		return self._stage_data_factory(stage)["output_directory_name"]

	def get_stage_output_directory(self, stage):
		return os.path.join(self.output_directory, self.get_stage_output_directory_name(stage))

	def get_requested_run_stages(self):
		"""
		This function returns a list of all requested run stages
		:return: list of run stages
		"""
		return self.run_stages

	def initialize_missing_outputs(self, identical_configurations):
		"""
		This function tries to find the needed outputs for all the run stages that the user requested to skip in this run
		using previous run folders, if available
		:param identical_configurations: a list of paths with previous identical runs
		:return:
		"""
		# check if provided previous runs contain the needed outputs
		first_stage_number = StageManager.get_stage_number_by_name(self.run_stages[0])
		if first_stage_number > 0:
			for stage_number in range(first_stage_number):
				skipped_stage = StageManager.get_stage_name_by_number(stage_number)
				missing_outputs = self.find_missing_outputs(skipped_stage, identical_configurations)
				self.add_missing_outputs_to_current_run(skipped_stage, missing_outputs)

	def find_missing_outputs(self, skipped_stage, identical_runs):
		"""
		This function tries to find the output of the skipped run stage in one of the previous run folders
		:param skipped_stage: the stage the output of which to search for
		:param identical_runs: a list of paths with identical runs
		:return: a path with an existing output for the requested stage, or an exception if none was found
		"""
		# if output exists in current run, our work here is done
		if self.get_stage_output(skipped_stage):
			return

		# else, get needed path names
		missing_outputs = list()
		stage_output_filenames = self.get_stage_output_file_names(skipped_stage)
		stage_directory_name = self.get_stage_output_directory_name(skipped_stage)

		# search for each one of the needed output files
		for output_file in stage_output_filenames:
			file_found = False
			for run_directory in identical_runs:
				if file_found: break
				for root, _, filenames in os.walk(run_directory):
					if file_found: break
					for filename in filenames:
						if filename == output_file and (not stage_directory_name or os.path.basename(root) == stage_directory_name):
							missing_outputs.append(os.path.join(root, filename))
							file_found = True
							break

			if not file_found:
				if skipped_stage in self.mandatory_stages:
					raise NeededOutputNotFound("Couldn't find output file {} for stage {} in any of the previous runs, "
										   "please run needed stages".format(output_file, skipped_stage))

		return missing_outputs

	def add_missing_outputs_to_current_run(self, skipped_stage, missing_outputs):
		# create destination directory, if doesn't exist
		destination_directory = self.get_stage_output_directory(skipped_stage)
		Path(destination_directory).mkdir(parents=True, exist_ok=True)

		# restore missing output files
		for output in missing_outputs:
			# add to simulation flow
			self.outputs_container[skipped_stage].append(CommonUtils.load_output(output))
			# add to output_directory
			shutil.copy(output, destination_directory)


