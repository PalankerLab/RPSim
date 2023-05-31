"""
This module defines the common behaviour and structure for all the run stages
"""
import os
from abc import ABC, abstractmethod

from configuration.stages import RunStages
from utilities.common_utilities import CommonUtils


class StageExecutionError(Exception):
	"""
	This class is used for all errors related to the execution of a run stage
	"""
	pass


class CommonRunStage(ABC):
	"""
	This class defined the needed structure for all run stages and implements the logic that is shared among them all
	"""

	def __init__(self, output_directory, outputs_container, output_file_names, output_directory_name):
		"""
		This function initializes a run stage
		:param output_directory: the output directory for this run
		:param outputs_container:
		"""
		self.executed = False
		self.succeeded = None
		self._stage_output = None
		self.outputs_container = outputs_container
		self.output_file_names = output_file_names
		self.output_directory_name = output_directory_name
		self.output_directory = os.path.join(output_directory, output_directory_name)

	@property
	@abstractmethod
	def stage_name(self):
		pass

	@abstractmethod
	def run_stage(self):
		pass

	def __str__(self):
		return "{} Stage".format(self.stage_name.title())

	@property
	def stage_output(self):
		return self._stage_output

	def run(self):
		"""
		This function defines the overall execution flow for any run stage
		:return:
		"""
		try:
			# execute stage and get output
			self._stage_output = self.run_stage()

			# make sure output structure is provided as a list
			self._stage_output = self._stage_output if isinstance(self._stage_output, list) else [self._stage_output]

			# save stage outputs
			for number, output in enumerate(self._stage_output):
				# save to file
				file_name = self.output_file_names[number] if number < len(self.output_file_names) else None
				#file_name = None if isinstance(output,str) and os.path.isdir(output) else self.output_file_names[
				# number]
				CommonUtils.store_output(output_directory=self.output_directory, output=output, file_name=file_name)
				# save to runtime structure
				self.outputs_container[self.stage_name].append(output)

			# document success
			self.succeeded = True
			return self._stage_output

		# catch and report any problems executing the stage
		except Exception as error:
			print("{} execution failed with the following error: {}".format(self.__str__(), error))
			self.succeeded = False
			raise StageExecutionError(error)

	def get_stage_output(self):
		"""
		this function returns the output of the stage
		:return:
		"""
		# check first if was executed and if not, get previous output
		return self.output
