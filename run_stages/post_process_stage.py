import glob
import multiprocessing
import os
import sys
from collections import OrderedDict

import matplotlib
from PIL import Image
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from configuration.configuration_manager import Configuration
from configuration.models import Models
from configuration.stages import RunStages

from run_stages.common_run_stage import CommonRunStage

from tqdm import tqdm
from time import sleep


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
		"""
		# Common parameters to mono and bipolar configurations
		self.number_of_pixels = Configuration().params["number_of_pixels"]
		self.start_time_ms = Configuration().params["pulse_start_time_in_ms"] # TODO automate 
		self.pulse_duration = Configuration().params["pulse_duration_in_ms"]
		self.average_over_pulse_duration = Configuration().params["average_over_pulse_duration"]
		self.pixel_coordinates = np.loadtxt(Configuration().params["r_matrix_input_file_px_pos"], delimiter=',')

		active_results = np.loadtxt(Configuration().params["r_matrix_input_file_active"], delimiter=',')
		self.active_x = active_results[0, :]
		self.active_voltage_mv = active_results[1:, :]

		# Time analysis - Averaging resolution
		if self.average_over_pulse_duration:
			# If we average over pulse duration, the time resolution is simply the pulse duration
			self.averaging_resolution_ms = self.pulse_duration
		else:
			# If we do time analysis, we use the inputted time_averaging_resolution_ms
			self.averaging_resolution_ms = Configuration().params["time_averaging_resolution_ms"]
		self.time_points_to_analyze_ms = self._get_time_sections()
    
		if Configuration().params["model"] == Models.MONOPOLAR.value:
			
			# TODO rename into more meaningful variable names
			V_dict = np.loadtxt(Configuration().params["r_matrix_input_file_EP_return_2D"], delimiter=',')  # mV  actually return_2D-whole
			self.x_frame = V_dict[0,:]
			self.y_frame = V_dict[1,:]
			self.V_dict_ret = V_dict[2:, :]

		if Configuration().params["model"] == Models.BIPOLAR.value:
			
			# define depth resolution
			default_depth_params =  [x*1 for x in range(160)] # in µm, from the implant up to 159 µm into the retina
			self.z_values = Configuration().params["depth_values_in_um"] if Configuration().params.get("depth_values_in_um") else default_depth_params

			# populate arrays with simulation values
			self.full_active_current_ua = np.array([self.simulation_stage_output[f'VCProbe{x + 1}'] for x in range(self.number_of_pixels)]) * 1E6
			self.full_return_current_ua = np.array([self.simulation_stage_output[f'VrCProbe{x + 1}'] for x in range(self.number_of_pixels)]) * 1E6

			# load all needed COMSOL files

			return_results = np.loadtxt(Configuration().params["r_matrix_input_file_return"], delimiter=',')
			self.return_x = return_results[0, :]
			self.return_voltage_mv = return_results[1:, :]

			return_near_results = np.loadtxt(Configuration().params["r_matrix_input_file_return_near"], delimiter=',')
			self.x_return_near = return_near_results[0, :]
			self.y_return_near = return_near_results[1, :]
			self.return_near_voltage_mv = return_near_results[2:, :]

			# create a 2D and a 3D mesh for populating the potential matrices
			frame_width = Configuration().params["frame_width"]
			self.x_frame = np.arange(start=-frame_width, stop=frame_width + 1, step=4)
			self.y_frame = np.arange(start=-frame_width, stop=frame_width + 1, step=4)

			self.xx, self.yy = np.meshgrid(self.x_frame, self.y_frame)
			self.xxx, self.yyy, self.zzz = np.meshgrid(self.x_frame, self.y_frame, self.z_values)

			self.xx_element = np.zeros([self.x_frame.size * self.y_frame.size, self.number_of_pixels])
			self.yy_element = np.zeros([self.x_frame.size * self.y_frame.size, self.number_of_pixels])
			for kk in range(self.number_of_pixels):
				self.xx_element[:, kk] = np.abs(self.xx.flatten() - self.pixel_coordinates[kk, 0])
				self.yy_element[:, kk] = np.abs(self.yy.flatten() - self.pixel_coordinates[kk, 1])

			self.dist_elem = np.sqrt(self.xx_element ** 2 + self.yy_element ** 2)

			# Va_time = np.zeros([len(times), N_pixels])
			# Vr_time = np.zeros([len(times), N_pixels])

	def _determine_start_pulse(sefl):
		"""
		TODO
		A function which determines when the steady state is reached for each cycle.
		The steady state is defined as a variation smaller than 1% between the max voltages reached for each cycle.

		Returns:
			start_time_ms (float): The first pulse considered as steady that we want to analyze
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
		When we do the time dynamics analyzis (i.e. average_over_pulse_duration = False)
			- We generate time section the width of the averaging resolution
			- And we only keep the time sections that were generated by Xyce

		Returns: 
			time_sections_to_inlcude list(float): The time sections that have to be used the time averaging (whether we average over pulse duration or not)
		"""
		
		if self.average_over_pulse_duration:
			return list(self.start_time_ms)
		
		else:
			# Generate all the theoretical time points
			all_time_sections = list(np.arange(start=self.start_time_ms, stop=self.start_time_ms + self.pulse_duration, step=self.averaging_resolution_ms))
			# Get all the actual time points and convert in ms
			time_vector = self.simulation_stage_output["time"] * 1e3  
			# And keep only the sections that are not empty / do exist
			time_sections_to_include = []

			for section in all_time_sections:
				time_indices_to_include =  (time_vector >= section) & (time_vector < section + self.averaging_resolution_ms)
				# We only keep the time section that do exist in the time vector generated by Xyce
				if time_indices_to_include.sum() != 0:
					time_sections_to_include.append(section)

			return time_sections_to_include
						
	def _generate_potential_matrix_per_time_section_monopolar(self, start_time, idx_time, total_time, average_over_pulse_duration=True):
		"""
		Handles the time analysis for monopolar configuration.
			Parameters:
				start_time: 
				idx_time (int): the current time slice used for the progress bar
				total_time (int): the total number of time slice to analyze for the progress bar 
				average_over_pulse_duration (bool): For averaging or not
		"""
		# TODO
		pass 
	
	def _generate_potential_matrix_per_time_section(self, start_time, idx_time, total_time):
		"""
		Handles the time analysis for bipolar configuration.
		The time analysis for monopolar configuration is not available yet. 
			Parameters:
				start_time (float): The time where current becomes steady enough and time averaging can start 
				idx_time (int): the current time slice used for the progress bar
				total_time (int): the total number of time slice to analyze for the progress bar 
		"""

		"""
		Here is the logic behind the previous implementation: 
		Xyce will output a current simulation w.r.t time of a certain duration parametrized by: simulation_duration_sec in user_interface
		This time simulation is only relevant when the current becomes steady, which is manually determined in Nathan's code depending on the
		simulation scenario. Here it is determined in the user_interface as pulse_start_time_in_ms
		We need to either autommate it, or provide clearer instructions in user_interface.
		
		When doing time_averaging, the whole time_frame of interest (from start_time to averaging_resolution_ms) is selected and all the currents are averaged.
		When time dynamics is enabled, several smaller time frames are analyzed within the 'big' time frame of interset. 
		The size of the small time frame is determined by end_time (time_step in Nathan's code) which should be called something time_dynamics_resolution_ms.
		The currents are still averaged, but within a much smaller time frame of time_dynamics_resolution_ms. This resolution should correspond to a period
		of time where the current is rather steady and we do not lose too much information by averaging (1 to 2 ms according to Nathan).   

		- start_time is currently manually selected as the time when the curent becomes steady for a given pulse (or pulse_start_time_in_ms in user_interface)
		  Steady state can be seen in the plots "Current vs Time" outputted by the Simulation stage for an active pixel.
		  Finding an active pixel is not automated yet, the one RPSim outputs is hardcoded and is likely inactive.  
		--> start_time needs to either be automated, or manually inputted for each type of simulation from the user_interface. 
		
		- end_time (or time_step in Nathan's notebooks), corresponds to the time resolution (or time frame) for the time dynamics analysis. 
		  It should probably be renamed time_dynamics_resolution_ms and be set as a parameter in user_interface.
		  It also depends on the type of simulation we are running. According to Nahtan, something between 1 to 2 ms is good enough. 
		--> To make the code compatible with time dynamics, we have to make sure that time indices outputted by Xyce do exist: for the
		  time resolution we want, the start time and length of the time frame (another variable). This will avoid the NaN we currently get.
		"""
		
		# Start by shifting the time vector around the time section of interest
		shifted_time_vector = self.simulation_stage_output['time'] * 1E3 - start_time
		end_time = self.averaging_resolution_ms		
		
		# filter the time indices that correspond to the current time window
		time_indices_to_include = (shifted_time_vector > 1e-6) & (shifted_time_vector < end_time)
		active_current_ua = self.full_active_current_ua[:, time_indices_to_include]
		return_current_ua = self.full_return_current_ua[:, time_indices_to_include]
		time_vector = shifted_time_vector[time_indices_to_include]
	
		# Time averaging in the frame of interest. The width of the window depends on the averaging_resolution_ms
		active_current_ua = (active_current_ua[:, :-1] + active_current_ua[:, 1:]) / 2
		return_current_ua = (return_current_ua[:, :-1] + return_current_ua[:, 1:]) / 2
		Td = time_vector[1:] - time_vector[:-1]
		active_current_ua = np.sum(active_current_ua * Td, axis=1) / np.sum(Td)
		return_current_ua = np.sum(return_current_ua * Td, axis=1) / np.sum(Td)

		# initialize output structure for this time point
		voltage_3d_matrix = np.zeros(self.xx.shape + (len(self.z_values),))

		# Calculate potential matrix for each z value in this time window
		# tqdm is used to generate a progress bar
		with tqdm(total = len(self.z_values), file = sys.stdout) as pbar_z:
			for z_index, z_value in enumerate(self.z_values):
				# Progress bar 
				pbar_z.set_description(f'Processing Z-slice: {1 + z_index} of time point {idx_time + 1}/{total_time}')
				pbar_z.update(1)
				sleep(0.1)
				
				# Actual computations for the XY potential at a given z-height
				V_elem_act = np.interp(self.dist_elem, self.active_x, self.active_voltage_mv[z_index, :])
				V_elem_ret = np.interp(self.dist_elem, self.return_x, self.return_voltage_mv[z_index, :])

				V_near = self.return_near_voltage_mv[(z_index) * self.x_return_near.size: (z_index + 1) * self.x_return_near.size, :]
				myfun = interpolate.RectBivariateSpline(self.x_return_near, self.y_return_near, V_near.T)

				idx_near = (self.xx_element < np.max(self.x_return_near)) & (self.yy_element < np.max(self.y_return_near))
				V_elem_ret[idx_near] = myfun.ev(self.xx_element[idx_near], self.yy_element[idx_near])

				voltage_xy_matrix = np.matmul(V_elem_act, active_current_ua) + np.matmul(V_elem_ret, return_current_ua)
				voltage_xy_matrix = np.reshape(voltage_xy_matrix, self.xx.shape)

				# update output structures
				voltage_3d_matrix[:, :, z_index] = voltage_xy_matrix
				#return start_time, voltage_3d_matrix

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
		time_window_start = Configuration().params["pulse_start_time_in_ms"] if Configuration().params["pulse_start_time_in_ms"] else 200
		time_window_end = time_window_start + (Configuration().params["pulse_duration_in_ms"] if Configuration().params["pulse_duration_in_ms"] else 9.8)

		# find the indices that correspond to the time window
		indices_in_window = np.where((time_ms >= time_window_start) & (time_ms <= time_window_end))

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
		self.z_values = None
		self._initialize_analysis_parameters()
		# find the diodes that are on
		on_diode_data_during_stable_pulse = self._extract_on_diode_pulses()

		if Configuration().params["model"] == Models.MONOPOLAR.value:

			# define end time based on whether we average over the pulse duration or not
			end_time = self.averaging_resolution_ms
			
			# Initialize the distance matrices
			# TODO We can probably reuse this block as it looks the same from the bipolar initialization (except for step = 4)
			XX, YY = np.meshgrid(self.x_frame, self.y_frame)
			XX_elem = np.zeros([self.x_frame.size * self.y_frame.size, self.number_of_pixels])
			YY_elem = np.zeros([self.x_frame.size * self.y_frame.size, self.number_of_pixels])
			for kk in range(self.number_of_pixels):
				XX_elem[:, kk] = np.abs(XX.flatten() - self.pixel_coordinates[kk, 0])
				YY_elem[:, kk] = np.abs(YY.flatten() - self.pixel_coordinates[kk, 1])

			dist_elem = np.sqrt(XX_elem ** 2 + YY_elem ** 2)

			# Load the active and return currents out of pixel from the simulation stage 
			I_act_t = np.array([self.simulation_stage_output[f'VCProbe{x + 1}'] for x in range(self.number_of_pixels)]) * 1E6  # uA
			I_ret_t = np.array(self.simulation_stage_output[f'VCProbe{0}']) * 1E6  # uA

			# Selecting the right times indices 
			T = self.simulation_stage_output['time'] * 1E3 - self.start_time_ms # Difference with Nathan's code is start_time_ms = 666
			t_idx = (T > 0.1) & (T < self.pulse_duration) # In Nathan's bipolar it's 0.0001 
			I_act = I_act_t[:, t_idx]
			I_ret = I_ret_t[t_idx]
			T = T[t_idx]

			# TODO figure out what that block is doing
			I_act = (I_act[:, :-1] + I_act[:, 1:]) / 2
			I_ret = (I_ret[:-1] + I_ret[1:]) / 2
			Td = T[1:] - T[:-1]
			I_act = np.sum(I_act * Td, axis=1) / np.sum(Td)
			I_ret = np.sum(I_ret * Td) / np.sum(Td)

			# Iterate from 0 (the pixels) to 159 um into the retina by default
			self.z_values = [x*1 for x in range(160)]
			voltage_3d_matrix = np.zeros(XX.shape + (len(self.z_values),))

			# create voltage x-y matrix for each z plane
			with tqdm(total = len(self.z_values), file = sys.stdout) as pbar_z:
				for z_idx in range(len(self.z_values)):
					# Progress bar monopoalr 
					pbar_z.set_description(f'Processing z-slice: {1 + z_idx}')
					pbar_z.update(1)
					sleep(0.1)
					
					# TODO figure out the computations' core
					V_elem_act = np.interp(dist_elem, self.active_x, self.active_voltage_mv[z_idx, :])
					V_ret = self.V_dict_ret[(z_idx) * self.x_frame.size: (z_idx + 1) * self.x_frame.size, :]

					V = np.matmul(V_elem_act, I_act)
					V = np.reshape(V, XX.shape) + V_ret * I_ret

					voltage_3d_matrix[:, :, z_idx] = V

			# initialize return structures
			voltage_4d_matrix = np.zeros(XX.shape + (len(self.z_values),) + (1,))

			voltage_4d_matrix[:, :, :, 0] = voltage_3d_matrix

			output_dictionary = {"v(x,y,z,t)_mv": voltage_4d_matrix,
								 "2d_mesh_um": (XX, YY),
								 "3d_mesh_um": None,
								 "z_um": self.z_values,
								 "t_ms": [0],
								 "pixel_coordinates_um": self.pixel_coordinates,
								 "on_diode_data": on_diode_data_during_stable_pulse}
			#frame_width = self.x_frame.size
			frame_width = abs(self.x_frame[0])

		if Configuration().params["model"] == Models.BIPOLAR.value:

			# initialize return structures
			voltage_4d_matrix = np.zeros(self.xx.shape + (len(self.z_values),) + (len(self.time_points_to_analyze_ms),))

			output_dictionary = {"v(x,y,z,t)_mv": None,
									"2d_mesh_um": (self.xx, self.yy),
									"3d_mesh_um": (self.xxx, self.yyy, self.zzz),
									"z_um": self.z_values,
									"t_ms": self.time_points_to_analyze_ms,
									"pixel_coordinates_um": self.pixel_coordinates,
									"on_diode_data": on_diode_data_during_stable_pulse}


			# # run post-processing concurrently for all the time points
			# with Pool(multiprocessing.cpu_count()//3) as pool:
			# 	results = pool.map(self._generate_potential_matrix_per_time_section, self.time_points_to_analyze_ms)
			#
			# for result in results:
			# 	voltage_4d_matrix[:, :, :, self.time_points_to_analyze_ms.index(result[0])] = result[1]

			# iterate over time and create 3D potential matrix for each time point
			for time_point_index, time_point_value in enumerate(self.time_points_to_analyze_ms):
				# calculate matrix for this time point
				voltage_3d_matrix = self._generate_potential_matrix_per_time_section(start_time=time_point_value, \
																					idx_time=time_point_index, total_time=len(self.time_points_to_analyze_ms))
				# add result to output structure
				voltage_4d_matrix[:, :, :, time_point_index] = voltage_3d_matrix

			# prepare output
			output_dictionary["v(x,y,z,t)_mv"] = voltage_4d_matrix

			frame_width = Configuration().params["frame_width"]

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








