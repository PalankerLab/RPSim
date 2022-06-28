"""
This module defines the common behaviour and structure for all the run stages
"""

from abc import ABC, abstractmethod

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

	def __init__(self, output_directory, get_stage_output_func):
		"""
		This function initializes a run stage
		:param output_directory: the output directory for this run
		:param get_stage_output_func:
		"""
		self.executed = False
		self.succeeded = None
		self._stage_output = None
		self.get_stage_output_func = get_stage_output_func
		self.output_directory = output_directory

	@abstractmethod
	def __str__(self):
		pass

	@property
	@abstractmethod
	def stage_name(self):
		pass

	@property
	@abstractmethod
	def output_file_name(self):
		pass

	@abstractmethod
	def run_stage(self):
		pass

	@property
	def stage_output(self):
		return self._stage_output

	@property
	def output_as_pickle(self):
		return True

	@property
	def output_as_figure(self):
		return False

	def run(self):
		"""
		This function defines the overall execution flow for any run stage
		:return:
		"""
		try:
			# execute stage
			self._stage_output = self.run_stage()
			# convert to list, if necessary
			self._stage_output = self._stage_output if isinstance(self._stage_output, list) else [self._stage_output]
			# save stage outputs to file
			for number, output in enumerate(self._stage_output):
				CommonUtils.store_output(output_directory=self.output_directory, output=output,
										 file_name=self.output_file_name[number],
										 as_pickle=self.output_as_pickle,
										 as_figure=self.output_as_figure)
			# document success
			self.succeeded = True
			return self._stage_output

		# catch and report any problems executing the stage
		except Exception as error:
			print("{} execution failed with the following error {}:".format(self.__str__(), error))
			self.succeeded = False
			raise StageExecutionError(error)

	def get_stage_output(self):
		"""
		this function returns the output of the stage
		:return:
		"""
		# check first if was executed and if not, get previous output
		return self.output
