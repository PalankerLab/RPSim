"""
This is the main module of the RPSim Software Tool
"""

import os
import time
import logging
import traceback

from configuration.configuration_manager import Configuration
from run_manager import RunManager
from utilities.common_utilities import CommonUtils
from utilities.exceptions import NeededOutputNotFound

global RPSIM_LOGGER


def run_rpsim(configuration=None, run_stages=None, skip_stages=None):
	"""
	This is the main function of the tool which handles its overall initialization and flow
	:param configuration: a dictionary with the user's configuration parameters
	:param run_stages: the run stages that the user would like to execute
	:return:
	"""
	try:

		# start runtime clock
		start_time = time.time()

		# setup logging
		RPSIM_LOGGER = CommonUtils.setup_logging(__file__)

		# parse configuration, the configuration manager and parsing is done once at the beginning of the run
		configuration_manager = Configuration(configuration)

		while next(configuration_manager):

			# remove any previous logger file handles
			RPSIM_LOGGER.handlers = [h for h in RPSIM_LOGGER.handlers if not isinstance(h, logging.FileHandler)]

			# create a new output folder for this run with the requested user prefix
			output_directory = CommonUtils.generate_output_directory(parent_directory=os.path.join(os.getcwd(),'user_files', 'user_output'))

			# redirect logging to a file inside the newly created output directory
			CommonUtils.add_logger_file_handle(RPSIM_LOGGER, file_name=os.path.join(output_directory, 'execution.log'))

			# report the start of a new run
			RPSIM_LOGGER.info('Staring a new run')
			RPSIM_LOGGER.info("Output directory: {}".format(output_directory))

			# start a new run manager
			run_manager = RunManager(output_directory, run_stages, skip_stages)

			# update the run stages that will be executed, and report
			run_stages = run_manager.get_requested_run_stages()
			RPSIM_LOGGER.info("Requested run stages: {}".format(list(run_stages)))

			# print current configuration to file
			RPSIM_LOGGER.info("Running the following configuration\n{}\n".format(configuration_manager.get_configuration_as_table()))

			# save configuration to file as dictionary for bookkeeping purposes
			configuration_manager.store_configuration(output_directory=output_directory)

			# Check if the same configuration was already executed in this location;
			if run_manager.find_previous_runs:

				RPSIM_LOGGER.info("Not all stages selected for execution, searching for previous runs with the same configuration...")
				identical_configurations = configuration_manager.find_identical_configurations(os.path.dirname(output_directory))

				# check if we have previous runs to rely on, if not we have a problem
				if not identical_configurations:
					raise NeededOutputNotFound("Current run relies on previous executions, but no such executions were found. Please run full flow.")

				run_manager.initialize_missing_outputs(identical_configurations)
				RPSIM_LOGGER.info("Missing outputs were initialized successfully")

			# execute all requested run stages
			for stage in run_stages:
				# initialize stage
				run_stage = run_manager.initialize_stage(stage)
				# print initialization message
				RPSIM_LOGGER.info("Running {}".format(run_stage.__str__()))
				# run stage
				run_stage.run()

		RPSIM_LOGGER.info("Finished running all provided configurations")

	except Exception as run_error:
		# report execution errors
		RPSIM_LOGGER.error(run_error)
		RPSIM_LOGGER.error("Whole error traceback:\n {}".format(traceback.format_exc()))

	finally:
		# stop runtime clock
		RPSIM_LOGGER.info("Execution time is {:.2f} minutes".format((time.time()-start_time)/60))

		# terminate execution
		logging.shutdown()


if __name__ == '__main__':
	run_rpsim()
