"""
This is the main module of the RPSim Software Tool
"""

import os
import sys
import time
import logging

from configuration.configuration_manager import Configuration
from run_stages.run_stage_manager import RunStageManager
from utilities.common_utilities import CommonUtils

global RPSIM_LOGGER


def run_rpsim(configuration=None, run_stages=None, find_similar_runs=True):
	"""
	This is the main function of the tool which handles its overall initialization and flow
	:param configuration: a dictionary with the user's configuration parameters
	:param run_stages: the run stages that the user would like to execute
	:param find_similar_runs: whether the tool should check for similar runs (currently based on configuration params)
	:return:
	"""
	try:

		# start runtime clock
		start_time = time.time()

		# setup logging
		RPSIM_LOGGER = CommonUtils.setup_logging(__file__)

		# parse configuration, the configuration manager and parsing is done once at the beginning of the run
		configuration_manager = Configuration(configuration)

		# iterate all provided configurations and execute
		while next(configuration_manager):

			# remove any previous logger file handles
			RPSIM_LOGGER.handlers = [h for h in RPSIM_LOGGER.handlers if not isinstance(h, logging.FileHandler)]

			# create a new output folder for this run with the requested user prefix
			output_directory = CommonUtils.generate_output_directory(parent_directory=os.path.join(os.getcwd(),'user_files', 'user_output'),
																	 directory_prefix=Configuration().params["output_prefix"])

			# redirect logging to a file inside the newly created output directory
			CommonUtils.add_logger_file_handle(RPSIM_LOGGER, file_name=os.path.join(output_directory, 'execution.log'))

			# report the start of a new run
			RPSIM_LOGGER.info('Staring a new run')

			# start a new run manager
			run_manager = RunStageManager(output_directory, run_stages)

			# print current configuration to file
			RPSIM_LOGGER.info("Running the following configuration:{}".format(configuration_manager.get_configuration_table()))

			# save configuration to file as dictionary for bookkeeping purposes
			configuration_manager.store_configuration(output_directory=output_directory)

			# if requested by the user, check if this configuration was already executed in this location
			if find_similar_runs:
				RPSIM_LOGGER.info("Checking for similar runs...")
				identical_configurations = configuration_manager.find_identical_configurations(
					os.path.dirname(output_directory))
				if identical_configurations:
					RPSIM_LOGGER.warning("Please note that the same configuration was executed in the following "
										 "runs:\n" + "\n".join(identical_configurations))
					RPSIM_LOGGER.warning("The latest execution is: {}".format(identical_configurations[0]))
				else:
					RPSIM_LOGGER.info("No similar runs were detected")

			# initialize skipped run stages, if necessary
			run_manager.initialize_missing_outputs(identical_configurations)

			# execute all requested run stages
			for stage in run_manager.get_run_stages():
				# initialize stage
				run_stage = run_manager.initialize_stage(stage)
				# print initialization message
				RPSIM_LOGGER.info("Running {}".format(run_stage.__str__()))
				# run stage
				stage_output = run_stage.run()
				# save stage output to run manager
				run_manager.update_stages_output(stage, stage_output)

		RPSIM_LOGGER.info("Finished running all provided configurations")

	except Exception as run_error:
		# report execution errors
		RPSIM_LOGGER.error(run_error)
		sys.exit(1)

	finally:
		# stop runtime clock
		RPSIM_LOGGER.info("Execution time is {:.2f} minutes".format((time.time()-start_time)/60))

		# terminate execution
		RPSIM_LOGGER.info("Done")
		logging.shutdown()


if __name__ == '__main__':
	run_rpsim()
