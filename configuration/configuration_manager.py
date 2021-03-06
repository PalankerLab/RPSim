"""
This module handles everything related to the initialization, parsing, and analysis of the tool's configuration
"""

import functools
import os
import random
import datetime
import numpy as np
from pathlib import Path
from deepdiff import DeepDiff
from prettytable import PrettyTable
from collections import defaultdict

from configuration.models import Models
from utilities.singleton import Singleton


class ConfigurationError(Exception):
	"""
	This class is used for configuration-related exceptions
	"""
	pass


class Configuration(metaclass=Singleton):
	"""
	This class is created once, and it handles all the configuration parameters for this tool
	"""

	# define all path-related variables
	input_string = "input"
	output_string = "output"
	default_path_prefix = os.path.join(Path(__file__).parents[1], 'user_files')

	def __init__(self, configuration):
		"""
		This function handles all the initialization and parsing of the user-provided configuration
		"""
		# initialization
		self.initial_params = defaultdict(None)
		self.params = defaultdict(None)
		self.configurations = list()
		self.identical_configurations = list()

		# parse configuration
		self._parse_configuration_file(configuration)

		# filter by geometry
		self._filter_by_model()

		# if some parameters have several values, expand to all possible configuration
		self._expand_permutations()

	def __iter__(self):
		"""
		This function is essential to implement an iterator
		:return:
		"""
		return self

	def __next__(self):
		"""
		This function returns the next configuration from the set of configurations requested by the user
		:return:
		"""
		# if there are additional configurations left to run, pop the next configuration and update needed values
		if len(self.configurations) > 0:
			# initialize path-related variables
			self.input_path_variables = list()
			self.output_path_variables = list()
			self.path_variables = list()

			# update params by popping the next configuration item
			self.params = self.configurations.pop()

			# set input and output paths for the user files
			if not self.params["user_files_path"]:
				self.params["user_files_path"] = Configuration.default_path_prefix
			self.params["user_input_path"] = os.path.join(self.params["user_files_path"], 'user_input')
			self.params["user_output_path"] = os.path.join(self.params["user_files_path"], 'user_output')

			# make sure all provided paths are valid
			self._extract_path_related_variables()
			self._validate_paths()

			# add new variables based on calculations
			self._add_common_calculated_values()

			# set configuration file name
			self.configuration_file_name = 'configuration_{}.pickle'.format(str(random.randint(0, 10000000)))

			return True
		return False

	def _filter_by_model(self):
		"""
		This function enables to filter the configuration parameters
		Currently the filtering only includes the model
		:return:
		"""
		for key in list(self.initial_params.keys()):
			if key in [Models.BIPOLAR.value, Models.MONOPOLAR.value]:
				if key == self.initial_params["model"]:
					for name, value in self.initial_params[key].items():
						self.initial_params[name] = value
				del self.initial_params[key]

	def _parse_configuration_file(self, configuration=None):
		"""
		This function performs the initial parsing of the input user configuration
		:param configuration: A dictionary with the user configuration parameters or a configuration python file
		:return:
		"""
		# if user provided a configuration file, just use it
		if configuration:
			if isinstance(configuration, dict):
				self.initial_params = configuration
			else:
				# else, check the path and if path is not valid, report
				if not os.path.exists(configuration):
					raise FileNotFoundError

				# if valid, check prefix and use this path for parsing
				if not configuration.endswith('.py'):
					raise ConfigurationError("Configuration file format is not supported, please provide a "
											"Python dictionary or a Python script")
				output = {}
				with open(configuration, 'r') as file:
					exec(file.read(), output, output)
					self.initial_params = output['configuration']

	def _expand_permutations(self):
		"""
		This function expands the list variables in the given configuration set to all applicable sets
		:return:
		"""
		# initialize expansion lists
		keys_to_expand = list()
		values_to_expand = list()

		# retrieve variables that have several values
		for key, value in self.initial_params.items():
			if isinstance(value, list):
				keys_to_expand.append(key)
				values_to_expand.append(value)

		# if no list values, create a single configuration instance
		if not values_to_expand:
			self.configurations.append(self.initial_params)

		# otherwise, generate all applicable configuration instances
		else:
			# verify that lists have identical number of values
			if max(values_to_expand, key=len) != min(values_to_expand, key=len):
				raise ConfigurationError(
					"The number of values in the list arguments of the provided configuration is not identical, please fix and rerun")

			# expand all permutations to different configuration schemes
			for permutation in zip(*values_to_expand):
				params_with_permutation = self.initial_params.copy()
				for index, key in enumerate(keys_to_expand):
					params_with_permutation[key] = permutation[index]
				# append current scheme to the previous schemes
				self.configurations.append(params_with_permutation)

	def _extract_path_related_variables(self):
		"""
		This function extracts configuration variables that represent input and output paths
		:return:
		"""
		# extract variables that hold input and output paths
		for name in self.params.keys():
			if Configuration.input_string in name:
				self.input_path_variables.append(name)
			# elif Configuration.output_string in name:
			# 	self.output_path_variables.append(name)
		self.path_variables = [*self.input_path_variables, *self.output_path_variables]

	def _validate_paths(self):
		"""
		This function verifies that all input path variables are valid paths.
		If not, it tries adding the common prefix, and checking again.
		:return:
		"""
		for variable in self.path_variables:
			# if input variable provided in config and path doesn't exist, add prefix
			if self.params.get(variable, None) and not os.path.exists(os.path.realpath(self.params[variable])):
				full_path_variable = os.path.join(self.params["user_input_path"], self.params[variable])

				# for input paths, validate again
				if variable in self.input_path_variables and not os.path.exists(full_path_variable):
					raise ConfigurationError("Provided input path {} doesn't exist!".format(full_path_variable))

				# if all is well, update path variable
				self.params[variable] = full_path_variable

	def _add_common_calculated_values(self):
		"""
		This function handles all configuration parameters generated by composition/calculation of several
		other parameters
		:return:
		"""
		self.params["sirof_active_capacitance_nF"] = self.params["sirof_capacitance"] * np.pi * self.params[
			"active_electrode_radius"] ** 2 * 1E-2
		self.params["return_width"] = (self.params["pixel_size"] - self.params["photosensitive_area_edge_to_edge"]) / 2
		self.params["return_width"] = (self.params["pixel_size"] - self.params["photosensitive_area_edge_to_edge"]) / 2
		if self.params["model"] == Models.BIPOLAR.value:
			self.params["return_to_active_area_ratio"] = np.sqrt(3) / 2 * (
						self.params["pixel_size"] ** 2 - self.params["photosensitive_area_edge_to_edge"] ** 2) / \
														(np.pi * self.params["active_electrode_radius"] ** 2)
		if self.params["photosensitive_area"] is None:
			self.params["photosensitive_area"] = np.sqrt(3) / 2 * self.params["photosensitive_area_edge_to_edge"] ** 2 \
												- np.pi * self.params["active_electrode_radius"] ** 2

		self.params["sequence_script_input_file"] = os.path.join(self.params["user_input_path"], "image_sequence",
																 self.params["video_sequence_name"], "seq_time.csv")
		self.params["gif_output_file"] = os.path.join(self.params["user_output_path"], "image_sequence",
													  self.params["video_sequence_name"], "demo.gif")
		self.params["video_sequence_output_file"] = os.path.join(self.params["user_output_path"], "image_sequence",
																 self.params[
																	 "video_sequence_name"] + "video_sequence.pkl")

	def print_configuration(self, logger=None):
		"""
		This function outputs the configuration in a table form to a logger or a file
		:param logger: logger object to use for the output stream
		:return:
		"""
		# generate output table
		configuration_table = PrettyTable()
		configuration_table.field_names = ['Attribute', 'Value']
		for key, value in self.params.items():
			configuration_table.add_row([key, value])

		# print output table to logger or file
		configuration_output = "\n{}\n".format(configuration_table)
		if logger:
			logger.info(configuration_output)
		else:
			print(configuration_output)

	def store_configuration(self, output_directory):
		"""
		This function stores the current configuration to a file at the given output directory
		:param output_directory: the directory in which to store the file
		:return:
		"""
		import pickle
		with open(os.path.join(output_directory, self.configuration_file_name), 'wb') as handle:
			pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def find_identical_configurations(self, root_directory):
		"""
		This function finds all previous configurations in a given output directory, and extracts a sorted list of
		the configurations that are identical to the current run
		:param root_directory: the directory in which to search for
		:return: sorted list of paths with identical configurations
		"""
		# find all configuration pickles in given output folder
		configuration_pickles = list()
		for root, _, filenames in os.walk(root_directory):
			for filename in filenames:
				if "configuration" in filename and ".pickle" in filename and not filename == self.configuration_file_name:
					configuration_pickles.append(os.path.join(root, filename))

		# iterate and compare
		for configuration_pickle in configuration_pickles:
			# restore given configuration
			import pickle
			with open(configuration_pickle, 'rb') as handle:
				stored_configuration = pickle.load(handle)

			# compare restored configuration to current, and append if identical
			if not DeepDiff(stored_configuration, self.params):
				self.identical_configurations.append(os.path.dirname(configuration_pickle))

		# sort by date and time
		self.identical_configurations = sorted(self.identical_configurations, key=functools.cmp_to_key(
			self.datetime_compare), reverse=True)

		return self.identical_configurations

	@staticmethod
	def datetime_compare(directory_x, directory_y):
		"""
		This functon compares between two directories based on their date and time
		:param directory_x: name of first input directory
		:param directory_y: name of second input directory
		:return: 0 if they are identical, 1 if x is older than y, and -1 if vice versa
		"""
		# obtain datetime object for directory_x
		directory_x_date = directory_x.split('-')[1].split('_')
		directory_x_date.extend(directory_x.split('-')[2].split('_'))
		directory_x_datetime = datetime.datetime(*list(map(int, directory_x_date)))

		# obtain datetime object for directory_y
		directory_y_date = directory_y.split('-')[1].split('_')
		directory_y_date.extend(directory_y.split('-')[2].split('_'))
		directory_y_datetime = datetime.datetime(*list(map(int, directory_y_date)))

		if directory_x_datetime == directory_y_datetime:
			return 0
		if directory_x_datetime > directory_y_datetime:
			return 1
		return -1
