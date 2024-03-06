import glob
import multiprocessing
import os
import sys
import warnings
from collections import OrderedDict
from functools import partial
from multiprocessing.managers import SharedMemoryManager

import matplotlib
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from configuration.configuration_manager import Configuration
from configuration.models import Models
from configuration.stages import RunStages

from run_stages.common_run_stage import CommonRunStage
from utilities.common_utilities import load_csv

from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

class PostProcessStage(CommonRunStage):
	"""
	This class implements the logic for the simulation stage, which executes the previously generated circuit in Xyce,
	while abiding by the structure required by the common run stage
	"""

	def __init__(self, *args):
		super().__init__(*args)
		# load simulation results
		self.simulation_stage_output = self.outputs_container[RunStages.simulation.name][0]

	@property
	def stage_name(self):
		return RunStages.post_process.name

	def _initialize_analysis_parameters(self):
		"""
		Initialize the common and specific parameters for either the monopoar or bipolar configuration.

		The post-processing or synthesis goal is to compute how voltage spreads across the retina based
		on the currents generated by each pixel by the simulation stage. 
		The simulation stage generates currents across time for each pixel. The time outputed is handled
		by Xyce and varies depending on the steadiness of the circuit. When the circuit's response varies
		a lot, the outputed time points are closer in time (down to microsecond and below), while when the
		response is steady, the time points can be spaced by several milliseconds. 

		The modelling happens for simulation_duration_sec, which is usually 0.5 second. Generating 3D voltages
		for 0.5 s with microsecond resolution would take a considerable amount of time and computational ressources.
		Therefore, we are only focusing on a single pulse of interest. This pulse corresponds to a single 
		pulse from the projection sequence, once the circuit is steady. This is determined by pulse_start_time_ms
		and is usually the sixth pulse. 

		If the time dynamics is enabled (i.e. average_over_pulse_duration = False), we interpolate the currents based on a 
		generated time vectors, that is consistent across the different scenarios. The interpolated currents are then 
		averaged within time sections, or bins. The width of these bins depend on the time_averaging_resolution_ms, 
		usually 1 ms. Then for each z-layer, a 2D voltage map is computed based on the currents. 

		If the time dynamics is disabled (i.e. average_over_pulse_duration = True), we do not interpolate the currents
		and directly averaged them over the entire pulse and only get one voltage time point for the whole pulse. 

		The pulse of interst is determined starts at  pulse_start_time_ms - pulse_extra_ms to analyse what is
		happening before the pulse. For best result, have pulse_extra as a mulitple of averaging_resolution.
		It lasts until pulse_duration_ms + pulse_extra_ms. pulse_extra_ms can be set to 0
		to analyse for the pulse duration only. The whole period is determined by window_start_ms and window_end_ms. 

		"""
		# Common parameters to mono and bipolar configurations
		self.multiprocess = Configuration().params["multiprocessing"]
		self.number_of_pixels = Configuration().params["number_of_pixels"]
		self.pulse_start_time_ms = Configuration().params["pulse_start_time_in_ms"] # TODO automate 
		self.pulse_duration_ms = Configuration().params["pulse_duration_in_ms"]
		self.average_over_pulse_duration = Configuration().params["average_over_pulse_duration"]
		self.pixel_coordinates = np.loadtxt(Configuration().params["r_matrix_input_file_px_pos"], delimiter=',')
		self.interpolation_resolution_ms = Configuration().params["interpolation_resolution_ms"]
		self.pulse_extra_ms = Configuration().params["pulse_extra_ms"]

		self.comsol_depth_values = None
		active_results = load_csv(Configuration().params["r_matrix_input_file_active"], stage = self)
		self.active_x = active_results[0, :]
		self.active_voltage_mv = active_results[1:, :]

		# Time analysis - Averaging resolution
		if self.average_over_pulse_duration:
			# If we average over pulse duration, the time resolution is simply the pulse duration
			self.averaging_resolution_ms = self.pulse_duration_ms
		else:
			# If we do time analysis, we use the inputted time_averaging_resolution_ms
			self.averaging_resolution_ms = Configuration().params["time_averaging_resolution_ms"]
		
		self.window_start_ms = self.pulse_start_time_ms - self.pulse_extra_ms
		self.window_end_ms = self.pulse_start_time_ms + self.pulse_duration_ms + self.pulse_extra_ms
		self.time_points_to_analyze_ms = self._get_time_sections()
		##
		self.time_start_ms, self.time_end_ms = self._get_time_sections_new()
		##
    
		self._initialize_depth_values()

		# initialize for symmetry, will be overriden in the bipolar case
		self.return_x = None
		self.x_return_near = None
		self.y_return_near = None
		self.y_return_near = None
		self.return_voltage_mv = None
		self.return_near_voltage_mv = None
		self.V_dict_ret = None

		if Configuration().params["model"] == Models.MONOPOLAR.value:
			
			# Load voltages and correct dimensions of the implant
			V_dict = load_csv(Configuration().params["r_matrix_input_file_EP_return_2D"])  # mV  actually return_2D-whole
			self.x_frame = V_dict[0,:]
			self.y_frame = V_dict[1,:]
			self.V_dict_ret = V_dict[2:, :]

			# Load the currents for all the time points
			self.full_active_current_ua = np.array([self.simulation_stage_output[f'VCProbe{x + 1}'] for x in range(self.number_of_pixels)]) * 1E6 
			self.full_return_current_ua = np.array(self.simulation_stage_output[f'VCProbe{0}']) * 1E6 
			# Convert the monopolar currents to a 2D array for compatibility with bipolar configuration - shape (1, nb_time_points)
			self.full_return_current_ua = np.reshape(self.full_return_current_ua, (1,-1))

		if Configuration().params["model"] == Models.BIPOLAR.value:
			
			# populate arrays with simulation values
			self.full_active_current_ua = np.array([self.simulation_stage_output[f'VCProbe{x + 1}'] for x in range(self.number_of_pixels)]) * 1E6
			self.full_return_current_ua = np.array([self.simulation_stage_output[f'VrCProbe{x + 1}'] for x in range(self.number_of_pixels)]) * 1E6
	
			# load all needed COMSOL files
			return_results = load_csv(Configuration().params["r_matrix_input_file_return"])
			self.return_x = return_results[0, :]
			self.return_voltage_mv = return_results[1:, :]

			return_near_results = load_csv(Configuration().params["r_matrix_input_file_return_near"])
			self.x_return_near = return_near_results[0, :]
			self.y_return_near = return_near_results[1, :]
			self.return_near_voltage_mv = return_near_results[2:, :]

			# create a 2D and a 3D mesh for populating the potential matrices
			frame_width = Configuration().params["frame_width"]
			self.x_frame = np.arange(start=-frame_width, stop=frame_width + 1, step=4)
			self.y_frame = np.arange(start=-frame_width, stop=frame_width + 1, step=4)

		##### Parameters common to bipolar and monopolar configurations ####
			
		self.xx, self.yy = np.meshgrid(self.x_frame, self.y_frame)
		# The 3D mesh is not being used anymore, but I left the line of code just in case
		#self.xxx, self.yyy, self.zzz = np.meshgrid(self.x_frame, self.y_frame, self.z_values) 

		self.xx_element = np.zeros([self.x_frame.size * self.y_frame.size, self.number_of_pixels])
		self.yy_element = np.zeros([self.x_frame.size * self.y_frame.size, self.number_of_pixels])
		for kk in range(self.number_of_pixels):
			self.xx_element[:, kk] = np.abs(self.xx.flatten() - self.pixel_coordinates[kk, 0])
			self.yy_element[:, kk] = np.abs(self.yy.flatten() - self.pixel_coordinates[kk, 1])

		self.dist_elem = np.sqrt(self.xx_element ** 2 + self.yy_element ** 2)

		# get the interpolated currents for time dynamics analysis
		if not self.average_over_pulse_duration:
			self._interpolate_currents()

	def _initialize_depth_values(self):
		"""
		This fucntion initializes the depth values used in the post process stage. 
		If no depth values are provided, it defaults to a 1 µm resolution from 0 to 160 for human 
		and 0 to 126 for rats. 

		Then the function matches the requested depth values to the ones available in the COMSOL files.
		These values should also span from 0 to 126/160 with a 1 µm increment, but it may change in the future.
		
		self.depth_indices (list(int)): contains the indices of the requested depth values in COMSOL files
		"""
		# define depth resolution for human or rat, default is 1 µm resolution from 1 to 160 or 126
		default_depth_range = 160 if "human" in Configuration().params["geometry"].lower() else 126
		default_depth_params_um =  [x*1 for x in range(default_depth_range)]
		self.depth_values_in_um = Configuration().params["depth_values_in_um"] if Configuration().params.get("depth_values_in_um") else default_depth_params_um	
		
		if self.comsol_depth_values:
			new_values = [(self.comsol_depth_values.index(depth), depth) if depth in self.comsol_depth_values else self._find_nearest_depth(depth) for depth in self.depth_values_in_um]
			# Unpack indices and values
			self.depth_indices, updated_depth_values = zip(*new_values)
			# Convert back to list and remove duplicates (if any) 
			self.depth_indices, self.depth_values_in_um = sorted(list(set(self.depth_indices))), sorted(list(set(updated_depth_values)))

		else:
			self.depth_indices = self.depth_values_in_um
			
	def _find_nearest_depth(self, depth):
		"""
		If the requested depth value by the user does not exist, this function
		will find the nearested value.

		Params: depth (float): the requested depth 
		Returns: idx_nearest (int): the idx of the nearest depth value in COMSOL files
		"""
		dist = np.abs(np.array(self.comsol_depth_values) - depth)
		idx_nearest = np.argmin(dist)
		depth_nearest = self.comsol_depth_values[idx_nearest]
		warnings.warn(f"The requested depth value {depth} is not available. Using the nearest value instead: {depth_nearest}.")
		return idx_nearest, depth_nearest
	
	def _determine_start_pulse(sefl):
		"""
		TODO
		A function which determines when the steady state is reached for each cycle.
		The steady state is defined as a variation smaller than 1% between the max voltages reached for each cycle.

		Returns:
			pulse_start_time_ms (float): The first pulse considered as steady that we want to analyze
		"""

		# We need to get the label of the most illuminated pixels and extract its VCProbe current
		# Get the projection frequency from the current sequence stage or pattern generation
		# Get the time windows per pulse and extract the max current
		# Then find from which pulse the below 1% variation happens 
		pass 
	
	def _get_time_sections(self):
		"""
		Used to determine suiteable time sections for time averaging
		
		When we average over the pulse duration we only take one time point:
			- the start of the pulse
			- then we average the current through the whole pulse
		When we do the time dynamics analysis (i.e. average_over_pulse_duration = False)
			- We interpolate the currents to have currents available at each time point
				- This facilitates the comparison between different scenario that do not
				  necessarily have the same time points generated by Xyce
			- We generate time section the width of the averaging resolution

		Returns: 
			time_sections_to_inlcude list(float): The time sections that have to be used in the time averaging (whether we average over pulse duration or not)
		"""
		
		if self.average_over_pulse_duration:
			return [self.pulse_start_time_ms]
		else:
			# New, now we interpolate, so all time point exist!
			return np.arange(self.window_start_ms, self.window_end_ms, self.averaging_resolution_ms).tolist()

	def _get_time_sections_new(self):
		"""
		Used to determine suiteable time sections for time averaging. 
		Time averaging is happening wether we use average over pulse duration or not.
		The number of averaged windows differ. 
		
		When we average over the pulse duration, we only extract one averaged point over the window
			- pulse_start_time_ms to pulse_start_time_ms + pulse_duration_ms
			- Then we average the currents through the whole pulse to extract one current point
		
		When we do the time dynamics analysis (i.e. average_over_pulse_duration = False)
			- We interpolate the currents to have currents for each time window
				- This facilitates the comparison between different scenario that do not
				  necessarily have the same time points generated by Xyce
			- We generate time section the width of the averaging resolution

		Returns: 
			time_start_ms list(float): The start of each time sections that have to be used in the time averaging 
			time_end_ms list(float): The end of each time sections that have to be used in the time averaging
		"""
		
		if self.average_over_pulse_duration:
			return [self.pulse_start_time_ms], [self.pulse_start_time_ms + self.pulse_duration_ms]
		else:
			# First compute the extra time points required by the user
			extra_before = np.arange(self.window_start_ms, self.pulse_start_time_ms+1e-3, self.averaging_resolution_ms).tolist()
			extra_after = np.arange(self.pulse_start_time_ms + self.pulse_duration_ms, self.window_end_ms+1e-3, self.averaging_resolution_ms).tolist()
			# The time section for the actual pulse
			time_start = np.arange(self.pulse_start_time_ms, self.pulse_start_time_ms + self.pulse_duration_ms, self.averaging_resolution_ms).tolist()
			time_end = np.arange(self.pulse_start_time_ms + self.averaging_resolution_ms, self.pulse_start_time_ms + self.pulse_duration_ms + self.averaging_resolution_ms, self.averaging_resolution_ms).tolist()
			
			# Make sure we don't include more current points than required in the last window (e.g. if pulse is 9.8 ms, last point should be 209.8 ms not 210.0 ms)
			time_end[-1] = self.pulse_start_time_ms + self.pulse_duration_ms
			extra_before[-1] = self.pulse_start_time_ms
			
			# Add the extra time to the main vectors 
			time_start_ms = extra_before[:-1] + time_start + extra_after[:-1]
			time_end_ms = extra_before[1:] + time_end + extra_after[1:]
			
			return time_start_ms, time_end_ms
	
	def _interpolate_currents(self):
		"""
		This function interpolates the currents along the desired interpolated time points.
		These time points are spaced by self.interpolation_resolution_ms
		These time points are only centered around the pulse of interst (window_start and window_end) which is determined by 
		- self.pulse_start_time_ms
		- self.pulse_duration_ms
		-self.pulse_extra_ms
		
		The last parameter allows to analyze the result for the extra period of time before and after the pulse.

		nb_time_points = the number of "interpolated time points" around the pulse of interest
		interpolated_active_current_ua (Numpy.array (nb_pixels, nb_time_points)): The interpolated active currents
		interpolated_return_current_ua (Numpy.array (nb_pixels, nb_time_points)): The interpolated return currents
		"""
		
		# Extract the time points during the pulse of interest (in ms)
		time_vector = self.simulation_stage_output["time"] * 1e3 
		# Filter
		pulse_mask = (time_vector >= self.window_start_ms) & (time_vector < self.window_end_ms)
		pulse_time = time_vector[pulse_mask]
		self.interpolated_pulse_time_ms = np.arange(self.window_start_ms, self.window_end_ms, self.interpolation_resolution_ms) 
		
		# Get the currents during the pulse of interst
		active_current_ua = self.full_active_current_ua[:, pulse_mask]
		return_current_ua = self.full_return_current_ua[:, pulse_mask]
		
		# Prepare empty arrays for results		
		nb_pixels_active, nb_pixels_return, nb_time_points = active_current_ua.shape[0], return_current_ua.shape[0], self.interpolated_pulse_time_ms.shape[0]
		self.interpolated_active_current_ua = np.zeros((nb_pixels_active, nb_time_points))
		self.interpolated_return_current_ua = np.zeros((nb_pixels_return, nb_time_points))

		# TODO It is inefficient to use for loop on Numpy arrays, but this is the best I've found so far
		for pixel in range(nb_pixels_active):
			# Interpolate the currents of the given pixel
			self.interpolated_active_current_ua[pixel, :] = np.interp(self.interpolated_pulse_time_ms, pulse_time, active_current_ua[pixel, :])
		# In case we have a different number of pixels for active and return beside the monopolar case
		for pixel in range(nb_pixels_return):	
			self.interpolated_return_current_ua[pixel, :] = np.interp(self.interpolated_pulse_time_ms, pulse_time, return_current_ua[pixel, :])

						
	def _get_currents_for_time_averaging(self, start_time, end_time):
		"""
		This function returns the correct currents and time vector for the time window of interst
		This window should not be mixed with pusle of interest, the point here is to analyze what
		is happening in the pulse of interset. 
		The results depend on whether the pulse stimulation has to be averaged or not

		This function is called in generate_potential_matrix_per_time_section

		Params:
			start_time (float): In ms the start time of the pulse of interest within the time slice
			end_time (float): In ms the shifted end of the pulse of interest within the time slice (=self.averaging_resolution_ms)
		
		Returns:
			active_current_ua (Numpy.array (nb_pixels, nb_time_points))
			return_current_ua (Numpy.array (nb_pixels, nb_time_points))
			time_vector (Numpy.array (nb_time_points,))
		"""
		
		# All in ms
		actual_time_vector = self.simulation_stage_output['time'] * 1E3 if self.average_over_pulse_duration else self.interpolated_pulse_time_ms
		shifted_time_vector = actual_time_vector - start_time		
		# filter the time indices that correspond to the current time window
		time_indices_to_include = (shifted_time_vector > 1e-6) & (shifted_time_vector < end_time)
		time_vector = shifted_time_vector[time_indices_to_include]
		
		# Get the correct currents depending on the time analysis
		if self.average_over_pulse_duration:
			active_current_ua = self.full_active_current_ua[:, time_indices_to_include]
			return_current_ua = self.full_return_current_ua[:, time_indices_to_include]
		else:
			active_current_ua = self.interpolated_active_current_ua[:, time_indices_to_include]
			return_current_ua = self.interpolated_return_current_ua[:, time_indices_to_include]
		
		return active_current_ua, return_current_ua, time_vector

	def _compute_synthesis_2D_serial(self, active_current_ua, return_current_ua, z_value):
		"""
		This function computes the 2D (XY-plane) voltages at a cerain depth/z-height for a given time
		section, based on the time averaged currents for that given time section, outputed by Xyce per pixel. 
		These computations are specific to a either a bipolar or monopolar pixel configuration. 
	
		Params:
			active_currents_ua (Numpy.array (nb_pixels_active,)): The time averaged active currents for the given time section
			return_currents_ua (Numpy.array (nb_pixels_return,)): The time averaged return currents for the given time section
			z_value (int): The value and index of the depth at which we are working in the retina in the elementary field matrices
		
		Returns:
			voltage_xy_matrix (Numpy.array (?, ?)): The 2D voltage map for the given z and t slices
		"""
		
		# TODO change z_index variable name to z_value and ask Nathan whether his V_elem is 0 index or 1 indexed!
		if Configuration().params["model"] == Models.MONOPOLAR.value:
			# Actual computations for the XY potential at a a given z-height
			V_elem_act = np.interp(self.dist_elem, self.active_x, self.active_voltage_mv[z_value, :])
			V_ret = self.V_dict_ret[(z_value) * self.x_frame.size: (z_value + 1) * self.x_frame.size, :]

			voltage_xy_matrix = np.matmul(V_elem_act, active_current_ua)
			# The return currents are reshaped again into a 1D array TODO check if precision is lost through the reshapes
			voltage_xy_matrix = np.reshape(voltage_xy_matrix, self.xx.shape) + V_ret * np.reshape(return_current_ua, (-1,))
		
		if Configuration().params["model"] == Models.BIPOLAR.value:
			# Actual computations for the XY potential at a given z-height
			V_elem_act = np.interp(self.dist_elem, self.active_x, self.active_voltage_mv[z_value, :])
			V_elem_ret = np.interp(self.dist_elem, self.return_x, self.return_voltage_mv[z_value, :])

			V_near = self.return_near_voltage_mv[(z_value) * self.x_return_near.size: (z_value + 1) * self.x_return_near.size, :]
			myfun = interpolate.RectBivariateSpline(self.x_return_near, self.y_return_near, V_near.T)

			idx_near = (self.xx_element < np.max(self.x_return_near)) & (self.yy_element < np.max(self.y_return_near))
			V_elem_ret[idx_near] = myfun.ev(self.xx_element[idx_near], self.yy_element[idx_near])

			voltage_xy_matrix = np.matmul(V_elem_act, active_current_ua) + np.matmul(V_elem_ret, return_current_ua)
			voltage_xy_matrix = np.reshape(voltage_xy_matrix, self.xx.shape)
		
		return voltage_xy_matrix 

	@staticmethod
	def _compute_synthesis_2D_parallel(shared_params_keys, shared_params_values, model, shared_memory_name, z_value):
		"""
		This function computes the 2D (XY-plane) voltages at a certain depth/z-height for a given time
		section, based on the time averaged currents for that given time section, outputted by Xyce per pixel.
		These computations are specific to a either a bipolar or monopolar pixel configuration. 
	
		Params:
			active_currents_ua (Numpy.array (nb_pixels_active,)): The time averaged active currents for the given time section
			return_currents_ua (Numpy.array (nb_pixels_return,)): The time averaged return currents for the given time section
			z_value (int): The value in um and index of the depth at which we are working in the retina for the elementary field matrices
		
		Returns:
			voltage_xy_matrix (Numpy.array (?, ?)): The 2D voltage map for the given z and t slices
		"""
		
		# initialize output
		voltage_xy_matrix = None

		shared_memory_handle = multiprocessing.shared_memory.SharedMemory(name=shared_memory_name)
		try:
			# reconstruct params as dictionary
			shared_params_dictionary = dict(zip(shared_params_keys, shared_params_values))

			# extract given arguments
			active_current_ua = shared_params_dictionary.get('active_current_ua')
			return_current_ua = shared_params_dictionary.get('return_current_ua')
			dist_elem = shared_params_dictionary.get('dist_elem')
			active_x = shared_params_dictionary.get('active_x')
			active_voltage_mv = shared_params_dictionary.get('active_voltage_mv')
			V_dict_ret = shared_params_dictionary.get('V_dict_ret')
			x_frame = shared_params_dictionary.get('x_frame')
			xx = shared_params_dictionary.get('xx')
			xx_element = shared_params_dictionary.get('xx_element')
			yy_element = shared_params_dictionary.get('yy_element')
			return_x = shared_params_dictionary.get('return_x')
			x_return_near = shared_params_dictionary.get('x_return_near')
			y_return_near = shared_params_dictionary.get('y_return_near')
			return_voltage_mv = shared_params_dictionary.get('return_voltage_mv')
			return_near_voltage_mv = shared_params_dictionary.get('return_near_voltage_mv')
		
			if model == Models.MONOPOLAR.value:
				# Actual computations for the XY potential at a given z-height
				V_elem_act = np.interp(dist_elem, active_x, active_voltage_mv[z_value, :])
				V_ret = V_dict_ret[z_value * x_frame.size: (z_value + 1) * x_frame.size, :]

				voltage_xy_matrix = np.matmul(V_elem_act, active_current_ua)
				# The return currents are reshaped again into a 1D array TODO check if precision is lost through the reshapes
				voltage_xy_matrix = np.reshape(voltage_xy_matrix, xx.shape) + V_ret * np.reshape(return_current_ua, (-1,))

			elif model == Models.BIPOLAR.value:
				# Actual computations for the XY potential at a given z-height
				V_elem_act = np.interp(dist_elem, active_x, active_voltage_mv[z_value, :])
				V_elem_ret = np.interp(dist_elem, return_x, return_voltage_mv[z_value, :])

				V_near = return_near_voltage_mv[z_value * x_return_near.size: (z_value + 1) * x_return_near.size, :]
				myfun = interpolate.RectBivariateSpline(x_return_near, y_return_near, V_near.T)

				idx_near = (xx_element < np.max(x_return_near)) & (yy_element < np.max(y_return_near))
				V_elem_ret[idx_near] = myfun.ev(xx_element[idx_near], yy_element[idx_near])

				voltage_xy_matrix = np.matmul(V_elem_act, active_current_ua) + np.matmul(V_elem_ret, return_current_ua)
				voltage_xy_matrix = np.reshape(voltage_xy_matrix, xx.shape)

		finally:
			shared_memory_handle.close() # Ensure that the shared memory block is closed and unlinked

		return voltage_xy_matrix
	
	def _generate_potential_matrix_per_time_section(self, start_time_ms, end_time_ms, pb_idx_time, pb_total_time):
		"""
		Handles the time analysis for bipolar configuration.
		The time analysis for monopolar configuration is not available yet. 
			Parameters:
				start_time_ms (float): Begining of the time section/window to analyse
				end_time_ms (float): End of the time section/window to analyse
				pb_idx_time (int): PROGRESS BAR - the current time section used 
				pb_total_time (int): PROGRESS BAR - the total number of time section to analyze
		"""
	
		# Get the currents and time vector for the time section of interst  
		active_current_ua, return_current_ua, time_vector = self._get_currents_for_time_averaging(start_time_ms, end_time_ms)		
		
		# Time averaging in the frame of interest. The width of the window depends on the averaging_resolution_ms
		if time_vector.sum() > 1:
			active_current_ua = (active_current_ua[:, :-1] + active_current_ua[:, 1:]) / 2
			return_current_ua = (return_current_ua[:, :-1] + return_current_ua[:, 1:]) / 2
			Td = time_vector[1:] - time_vector[:-1]
			# Average over time
			active_current_ua = np.sum(active_current_ua * Td, axis=1) / np.sum(Td)
			return_current_ua = np.sum(return_current_ua * Td, axis=1) / np.sum(Td)
			# The current arrays start with shape (nb pixels, np time points) and end up (nb pixels,)
		else:
			# If there is only one time point, do not average but reshape 
			# Reshape into a (nb_pixels,) array from (nb_pixels, 1) array
			active_current_ua = active_current_ua.reshape(-1,)
			return_current_ua = return_current_ua.reshape(-1,)
		
		# initialize output structure for this time point
		voltage_3d_matrix = np.zeros(self.xx.shape + (len(self.depth_values_in_um),))
		
		## run all the different z-slices in parallel, as they are independent
		if self.multiprocess:

			# Use 2/3 of the CPUs available
			cpu_to_use = Configuration().params["cpu_to_use"]
			# extracting self values into a dictionary
			params = {
				'active_current_ua': active_current_ua,
				'return_current_ua': return_current_ua,
				'dist_elem': self.dist_elem,
				'active_x': self.active_x,
				'active_voltage_mv': self.active_voltage_mv,
				'V_dict_ret': self.V_dict_ret,
				'x_frame': self.x_frame,
				'xx': self.xx,
				'xx_element': self.xx_element,
				'yy_element': self.yy_element,
				'return_x': self.return_x,
				'x_return_near': self.x_return_near,
				'y_return_near': self.y_return_near,
				'return_voltage_mv': self.return_voltage_mv,
				'return_near_voltage_mv': self.return_near_voltage_mv
			}
		
			##### Anna's version ####
			
			# calculate the total size needed for the shared memory block
			total_size = sum(value.nbytes if value is not None else 0 for value in params.values())

			# initialize a shared memory block
			shm = multiprocessing.shared_memory.SharedMemory(create=True, size=total_size, name='shared_block')

			try:
				# copy shared arrays to shared memory block
				shared_values = []
				for value, offset in zip(params.values(), np.cumsum([0] + [value.nbytes if value is not None else 0
																		   for value in params.values()])[:-1]):
					if value is not None:
						copied_array = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf, offset=offset)
						#np.copyto(copied_array, value)
						copied_array[:] = value[:]
						shared_values.append(copied_array)
					else:
						shared_values.append(value)

				# Test
				#shared_values = multiprocessing.shared_memory.ShareableList(shared_values)
				shared_keys = multiprocessing.shared_memory.ShareableList(params.keys())
				shared_depth_values = multiprocessing.shared_memory.ShareableList(self.depth_indices)
				model = Configuration().params["model"]
				# process the needed z-slices in parallel
				with Pool(cpu_to_use) as pool:
					results = pool.map(partial(self._compute_synthesis_2D_parallel, shared_keys, shared_values, model,
											   shm.name), shared_depth_values, chunksize = cpu_to_use)

			finally:
				shm.unlink()

			# iterate over the results in series and load the values
			for z_index, voltage_2d_matrix in enumerate(results):
				voltage_3d_matrix[:, :, z_index] = voltage_2d_matrix

		else:
			with tqdm(total = len(self.depth_values_in_um), file = sys.stdout) as pbar_z: # used for PROGRESS BAR
				for z_index, z_value in enumerate(self.depth_indices):
					# Progress bar 
					pbar_z.set_description(f'Processing Z-slice: {1 + z_index} of time point {pb_idx_time + 1}/{pb_total_time}')
					pbar_z.update(1)
				
					# Compute the voltages and update the output structures
					voltage_3d_matrix[:, :, z_index] = self._compute_synthesis_2D_serial(active_current_ua, 
																				return_current_ua, 
																				z_value)
		
		return voltage_3d_matrix

	def _create_potential_plot_per_depth(self, array_to_plot, frame_width, z_index, z_value):
		vmin, vmax = -20, 180
		cmap = plt.cm.hot
		norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

		fig = plt.figure(dpi=300)
		fig.suptitle(f'Vxy_at_z_{z_value}um_averaged_over_pulse_duration.png')
		plt.imshow(array_to_plot[:, :, z_index, 0],
							   origin='lower',
							   extent=(-frame_width, frame_width, -frame_width,frame_width),
							   aspect=1,
							   cmap=cmap,
							   norm=norm)

		colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
		colorbar.set_ticks(np.linspace(vmin, vmax, 11))

		plt.xlabel("x ($\mu$m)", fontsize=14)
		plt.ylabel("y ($\mu$m)", fontsize=14)
		colorbar.ax.set_ylabel("Potential (mV)", fontsize=14)
		plt.tight_layout()
		plt.close()
		return fig
	
	@staticmethod
	def _extract_diode_number(diode_name):
		parts = diode_name.split("Pt")
		number_part = int(parts[1])
		return number_part

	def _extract_on_diode_pulses(self):
		# initialize output
		on_diode_data_during_stable_pulse = OrderedDict()

		# filter and sort the diodes
		active_voltage_diodes = [key for key in self.simulation_stage_output.keys() if "Pt" in key and "rPt" not in key]
		active_voltage_diodes = sorted(active_voltage_diodes, key=self._extract_diode_number)
		active_voltage_diodes_numbers = [str(self._extract_diode_number(item)) for item in active_voltage_diodes]

		# convert time array to ms
		time_ms = self.simulation_stage_output['time'] * 1E3

		# define a sttable pulse time window
		# TODO update with new parameters self.window_start and end
		#time_window_start = Configuration().params["pulse_start_time_in_ms"] if Configuration().params["pulse_start_time_in_ms"] else 200
		#time_window_end = time_window_start + (Configuration().params["pulse_duration_in_ms"] if Configuration().params["pulse_duration_in_ms"] else 9.8)

		# find the indices that correspond to the time window
		# TODO: make sure that the window is not empty 
		indices_in_window = np.where((time_ms >= self.window_start_ms) & (time_ms <= self.window_end_ms))

		# filter time vector to pulse window
		time_ms = time_ms[indices_in_window]

		# add time entry to output dictionary
		on_diode_data_during_stable_pulse["time_ms"] = time_ms

		# set the threshold for diode activation
		threshold = Configuration().params["on_diode_threshold_mV"] if Configuration().params["on_diode_threshold_mV"] else 50

		# search for diodes that are active within the needed pulse window
		for diode_number in active_voltage_diodes_numbers:

			# convert this diodes voltage to mV
			voltage_mv = self.simulation_stage_output['Pt' + diode_number] * 1E3
			current_ua = self.simulation_stage_output['VCProbe' + diode_number] * 1E6

			# extract a section that corresponds to a stable pulse
			voltage_in_time_window = voltage_mv[indices_in_window]
			current_in_time_window = current_ua[indices_in_window]

			# check if diode is on, and if so, add data to output
			if max(voltage_in_time_window) > threshold:
				on_diode_data_during_stable_pulse[diode_number] = OrderedDict()
				on_diode_data_during_stable_pulse[diode_number]["current"] = current_in_time_window
				on_diode_data_during_stable_pulse[diode_number]["voltage"] = voltage_in_time_window

		return on_diode_data_during_stable_pulse

	# def _generate_gif_data(self):
	# 	{"gif_data": self.gif_image, "gif_time": self.gif_time}, self.image_sequence_input_folder]
	# 	frames = [Image.open(image) for image in sorted(glob.glob(f"{self.output_directory}/*.png"),key=os.path.getmtime)]
	# 	frame_one = frames[0]
	# 	frame_one.save(os.path.join(self.output_directory, "3D_potential_averaged_over_pulse_duration.gif"), format="GIF", \
	# 				   append_images=frames, save_all=True, duration=1000, loop=0)

	def run_stage(self, *args, **kwargs):
		"""
		This function holds the execution logic for the simulation stage
		:param args:
		:param kwargs:
		:return:
		"""
		self.depth_values_in_um = None
		self._initialize_analysis_parameters()
		# find the diodes that are on
		on_diode_data_during_stable_pulse = self._extract_on_diode_pulses()
		# Initialize output structures
		voltage_4d_matrix = np.zeros(self.xx.shape + (len(self.depth_values_in_um),) + (len(self.time_points_to_analyze_ms),))
		output_dictionary = {"v(x,y,z,t)_mv": None,
								"2d_mesh_um": (self.xx, self.yy),
								"z_um": self.depth_values_in_um,
								"t_start_ms": self.time_start_ms,
								"t_end_ms": self.time_end_ms,
								"pixel_coordinates_um": self.pixel_coordinates,
								"on_diode_data": on_diode_data_during_stable_pulse}


		# Different progress bar whether we do multiprocessing or not
		if self.multiprocess:
			with tqdm(total = len(self.time_points_to_analyze_ms), file = sys.stdout) as pbar_t: # used for PROGRESS
				# BAR
				# iterate iver the time points and calculate the volumetric potential matrix for each one
				for time_point_index, (start_time, end_time) in enumerate(zip(self.time_start_ms, self.time_end_ms)):
					# update progress bar
					pbar_t.set_description(f'Processing the {len(self.depth_values_in_um)} z-slices of time-point: {1 + time_point_index}/{len(self.time_start_ms)}...')
					pbar_t.update(1)
					
					# calculate matrix for this time point
					voltage_3d_matrix = self._generate_potential_matrix_per_time_section(start_time_ms=start_time, end_time_ms=end_time, pb_idx_time=time_point_index, pb_total_time=len(self.time_start_ms))
					# add result to output structure
					voltage_4d_matrix[:, :, :, time_point_index] = voltage_3d_matrix

		else:
			for time_point_index, (start_time, end_time) in enumerate(zip(self.time_start_ms, self.time_end_ms)):
				# calculate matrix for this time point
				voltage_3d_matrix = self._generate_potential_matrix_per_time_section(start_time_ms=start_time, \
																		end_time_ms=end_time, pb_idx_time=time_point_index, pb_total_time=len(self.time_start_ms))
				# add result to output structure
				voltage_4d_matrix[:, :, :, time_point_index] = voltage_3d_matrix
		
		# Save results
		output_dictionary["v(x,y,z,t)_mv"] = voltage_4d_matrix	


		# TODO some code to use for parallelization	
	    # run post-processing concurrently for all the time points
		# with Pool(multiprocessing.cpu_count()//3) as pool:
		# 	results = pool.map(self._generate_potential_matrix_per_time_section, self.time_points_to_analyze_ms)
		
		# plot
		figures = list()
		# for z_index, z_value in enumerate(self.z_values):
		# 	figure = self._create_potential_plot_per_depth(voltage_4d_matrix, frame_width, z_index, z_value)
		# 	figures.append(figure)

		# visualize electric field in 3D
		# fig = plt.figure()
		# plot_3d_array(clean_v_4d[:,:,:,0], "3D Potential at t={}ms".format(time),"x [um]","y [um]","z [um]","Potential [mV]", mesh=three_d_mesh)

		# save as gif
		# self._generate_gif_data()

		return [output_dictionary, *figures]

