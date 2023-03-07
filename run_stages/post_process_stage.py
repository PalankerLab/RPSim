import multiprocessing
from multiprocessing import Pool

import numpy as np
from scipy import interpolate

from configuration.configuration_manager import Configuration
from configuration.models import Models
from configuration.stages import RunStages

from run_stages.common_run_stage import CommonRunStage
from utilities.visualization_utils import VisualizationUtils


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
			# extract analysis parameters
			number_of_pixels = Configuration().params["number_of_pixels"]
			self.start_time_ms = Configuration().params["start_time_in_ms"]
			self.pulse_duration = Configuration().params["stimulation_duration_in_ms"]
			self.time_step_ms = self.pulse_duration
			self.time_points_to_analyze_ms = [self.start_time_ms]

			if Configuration().params["analyze_time_dynamics"]:
				self.time_step_ms = 0.5
				self.time_points_to_analyze_ms = list(np.arange(start=self.start_time_ms,
																stop=self.start_time_ms + self.pulse_duration * 2,
																step=self.time_step_ms))

			# define depth resolution
			self.z_values = Configuration().params["depth_values_in_um"] if Configuration().params[
				"depth_values_in_um"] else [x * 5 for x in range(11)] + [x * 5 + 57 for x in range(20)]

			# populate arrays with simulation values
			self.full_active_current_ua = np.array([self.simulation_stage_output[f'VCProbe{x + 1}'] for x in range(number_of_pixels)]) * 1E6
			self.full_return_current_ua = np.array([self.simulation_stage_output[f'VrCProbe{x + 1}'] for x in range(number_of_pixels)]) * 1E6

			# load all needed COMSOL files
			active_results = np.loadtxt(Configuration().params["r_matrix_input_file_active"], delimiter=',')
			self.active_x = active_results[0, :]
			self.active_voltage_mv = active_results[1:, :]

			return_results = np.loadtxt(Configuration().params["r_matrix_input_file_return"], delimiter=',')
			self.return_x = return_results[0, :]
			self.return_voltage_mv = return_results[1:, :]

			return_near_results = np.loadtxt(Configuration().params["r_matrix_input_file_return_near"], delimiter=',')
			self.x_return_near = return_near_results[0, :]
			self.y_return_near = return_near_results[1, :]
			self.return_near_voltage_mv = return_near_results[2:, :]

			self.pixel_coordinates = np.loadtxt(Configuration().params["r_matrix_input_file_px_pos"], delimiter=',')

			# create a 2D and a 3D mesh for populating the potential matrices
			frame_width = Configuration().params["frame_width"]
			self.x_frame = np.arange(start=-frame_width, stop=frame_width + 1, step=4)
			self.y_frame = np.arange(start=-frame_width, stop=frame_width + 1, step=4)

			self.xx, self.yy = np.meshgrid(self.x_frame, self.y_frame)
			self.xxx, self.yyy, self.zzz = np.meshgrid(self.x_frame, self.y_frame, self.z_values)

			self.xx_element = np.zeros([self.x_frame.size * self.y_frame.size, number_of_pixels])
			self.yy_element = np.zeros([self.x_frame.size * self.y_frame.size, number_of_pixels])
			for kk in range(number_of_pixels):
				self.xx_element[:, kk] = np.abs(self.xx.flatten() - self.pixel_coordinates[kk, 0])
				self.yy_element[:, kk] = np.abs(self.yy.flatten() - self.pixel_coordinates[kk, 1])

			self.dist_elem = np.sqrt(self.xx_element ** 2 + self.yy_element ** 2)

			# Va_time = np.zeros([len(times), N_pixels])
			# Vr_time = np.zeros([len(times), N_pixels])

	def _generate_potential_matrix_per_time_section(self, start_time):

		# shift time vector to point of interest
		shifted_time_vector = self.simulation_stage_output['time'] * 1E3 - start_time

		# filter the time indices that correspond to the current time window
		time_indices_to_include = (shifted_time_vector > 0.1) & (shifted_time_vector < self.time_step_ms)
		active_current_ua = self.full_active_current_ua[:, time_indices_to_include]
		return_current_ua = self.full_return_current_ua[:, time_indices_to_include]
		time_vector = shifted_time_vector[time_indices_to_include]

		active_current_ua = (active_current_ua[:, :-1] + active_current_ua[:, 1:]) / 2
		return_current_ua = (return_current_ua[:, :-1] + return_current_ua[:, 1:]) / 2
		Td = time_vector[1:] - time_vector[:-1]
		active_current_ua = np.sum(active_current_ua * Td, axis=1) / np.sum(Td)
		return_current_ua = np.sum(return_current_ua * Td, axis=1) / np.sum(Td)

		# initialize output structure for this time point
		voltage_3d_matrix = np.zeros(self.xx.shape + (len(self.z_values),))

		# calculate potential matrix for each z value in this time window
		for z_index, z_value in enumerate(self.z_values):
			V_elem_act = np.interp(self.dist_elem, self.active_x, self.active_voltage_mv[z_index, :])
			V_elem_ret = np.interp(self.dist_elem, self.return_x, self.return_voltage_mv[z_index, :])

			V_near = self.return_near_voltage_mv[(z_index) * self.x_return_near.size: (z_index + 1) * self.x_return_near.size, :]
			myfun = interpolate.RectBivariateSpline(self.x_return_near, self.y_return_near, V_near.T)

			idx_near = (self.xx_element < np.max(self.x_return_near)) & (self.yy_element < np.max(self.y_return_near))
			V_elem_ret[idx_near] = myfun.ev(self.xx_element[idx_near], self.yy_element[idx_near])

			voltage_xy_matrix = np.matmul(V_elem_act, active_current_ua) + np.matmul(V_elem_ret, return_current_ua)
			voltage_xy_matrix = np.reshape(voltage_xy_matrix, self.xx.shape)

			# logic to only take the V_field at the pixels
			# pixelCtrIndx = np.zeros([N_pixels,2],dtype = int)
			# pixelRtnIndx = np.zeros([N_pixels,2],dtype = int)
			# for i in range(0,px_pos.shape[0],1):
			#     # for active
			#     xdistancesFromPixel = np.absolute(XX-px_pos[i,0])
			#     ydistancesFromPixel = np.absolute(YY-px_pos[i,1])
			#     pixelCtrIndx[i,:] = np.unravel_index(np.absolute(xdistancesFromPixel+ydistancesFromPixel).argmin(), XX.shape)
			#     #for return
			#     returnXOffset = 44 # horizontal distance from pixel center to return electrode (um)
			#     returnYOffset = 0 # vertical distance from pixel center to return electrode (um)
			#     xdistancesFromPixel = np.absolute(XX-(px_pos[i,0]+returnXOffset))
			#     ydistancesFromPixel = np.absolute(YY-(px_pos[i,1]+returnYOffset))
			#     pixelRtnIndx[i,:] = np.unravel_index(np.absolute(xdistancesFromPixel+ydistancesFromPixel).argmin(), XX.shape)
			# V_active = np.zeros([N_pixels])
			# V_return = np.zeros([N_pixels])
			# for j in range(0,N_pixels,1):
			#     V_active[j]=V[pixelCtrIndx[j,0],pixelCtrIndx[j,1]]
			#     V_return[j]=V[pixelRtnIndx[j,0],pixelCtrIndx[j,1]]
			# Va_time[time_idx,:] = V_active
			# Vr_time[time_idx,:] = V_return

			# update output structures
			voltage_3d_matrix[:, :, z_index] = voltage_xy_matrix
			return start_time, voltage_3d_matrix

	def run_stage(self, *args, **kwargs):
		"""
		This function holds the execution logic for the simulation stage
		:param args:
		:param kwargs:
		:return:
		"""
		if Configuration().params["model"] == Models.BIPOLAR.value:

			# initialize all the parameters we need for the post-processing analysis
			self._initialize_analysis_parameters()

			# initialize return structures
			voltage_4d_matrix = np.zeros(self.xx.shape + (len(self.z_values),) + (len(self.time_points_to_analyze_ms),))

			output_dictionary = {"v(x,y,z,t)_mv": None,
									  "2d_mesh_um": (self.xx, self.yy),
									  "3d_mesh_um": (self.xxx, self.yyy, self.zzz),
									  "z_um": self.z_values,
									  "t_ms": self.time_points_to_analyze_ms,
									  "pixel_coordinates_um": self.pixel_coordinates}

			# run post-processing concurrently for all the time points
			with Pool(multiprocessing.cpu_count()//3) as pool:
				for result in pool.map(self._generate_potential_matrix_per_time_section, self.time_points_to_analyze_ms):
					voltage_4d_matrix[:, :, :, self.time_points_to_analyze_ms.index(result[0])] = result[1]

			# # iterate over time and create 3D potential matrix for each time point
			# for time_point_index, time_point_value in enumerate(self.time_points_to_analyze_ms):
			#
			# 	# calculate metrix for this time point
			# 	voltage_3d_matrix = self._generate_potential_matrix_per_time_section(start_time=time_point_value)
			#
			# 	# add result to output structure
			# 	self.voltage_4d_matrix[:, :, :, time_point_index] = voltage_3d_matrix
			#
			# prepare output
			output_dictionary["v(x,y,z,t)_mv"] = voltage_4d_matrix
			return [output_dictionary]








