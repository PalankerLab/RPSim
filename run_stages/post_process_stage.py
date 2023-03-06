import os

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
		self.simulation_stage_output = self.outputs_container[RunStages.simulation.name][0]

	@property
	def stage_name(self):
		return RunStages.post_process.name

	def run_stage(self, *args, **kwargs):
		"""
		This function holds the execution logic for the simulation stage
		:param args:
		:param kwargs:
		:return:
		"""
		if Configuration().params["model"] == Models.BIPOLAR.value:

			# extract analysis parameters
			start_time_ms = Configuration().params["start_time_in_ms"]
			end_time_ms = start_time_ms
			time_step_ms = 0
			pulse_duration = Configuration().params["stimulation_duration_in_ms"]

			if Configuration().params["analyze_time_dynamics"]:
				end_time_ms = start_time_ms + pulse_duration * 2
				time_step_ms = 0.5
				pulse_duration = time_step_ms

			# generate time points to analyze
			time_points_to_analyze_ms = np.arange(start_time_ms, end_time_ms, time_step_ms)

			# define depth resolution
			z_values = Configuration().params["depth_values_in_um"] if Configuration().params["depth_values_in_um"] else [x * 5 for x in range(11)] + [x * 5 + 57 for x in range(20)]

			# load all needed COMSOL files
			active_results = np.loadtxt(Configuration().params["r_matrix_input_file_active"], delimiter=',')
			active_x = active_results[0, :]
			active_voltage_mv = active_results[1:, :]

			return_results = np.loadtxt(Configuration().params["r_matrix_input_file_return"], delimiter=',')
			return_x = return_results[0, :]
			return_voltage_mv = return_results[1:, :]

			return_near_results = np.loadtxt(Configuration().params["r_matrix_input_file_return_near"], delimiter=',')
			x_return_near = return_near_results[0, :]
			y_return_near = return_near_results[1, :]
			return_near_voltage_mv = return_near_results[2:, :]

			pixel_coordinates = np.loadtxt(Configuration().params["r_matrix_input_file_px_pos"], delimiter=',')
			number_of_pixels = pixel_coordinates.shape[0]

			# create a 2D and a 3D mesh for populating the potential matrices
			frame_width = Configuration().params["frame_width"]
			x_frame = np.arange(start=-frame_width, stop=frame_width + 1, step=4)
			y_frame = np.arange(start=-frame_width, stop=frame_width + 1, step=4)

			xx, yy = np.meshgrid(x_frame, y_frame)
			xxx, yyy, zzz = np.meshgrid(x_frame, y_frame, z_values)

			xx_element = np.zeros([x_frame.size * y_frame.size, number_of_pixels])
			yy_element = np.zeros([x_frame.size * y_frame.size, number_of_pixels])
			for kk in range(number_of_pixels):
				xx_element[:, kk] = np.abs(xx.flatten() - pixel_coordinates[kk, 0])
				yy_element[:, kk] = np.abs(yy.flatten() - pixel_coordinates[kk, 1])

			dist_elem = np.sqrt(xx_element ** 2 + yy_element ** 2)

			# initialize output structure
			# Va_time = np.zeros([len(times), N_pixels])
			# Vr_time = np.zeros([len(times), N_pixels])
			voltage_4d_matrix = np.zeros(xx.shape + (len(z_values),) + (len(time_points_to_analyze_ms),))

			# iterate over time and create 3D potential matrix for each time point
			for time_point_index, time_point_value in enumerate(time_points_to_analyze_ms):

				print("current start point is: {}ms".format(str(time_point_value)))

				# populate arrays with simulation values
				full_active_current_ua = np.array([self.simulation_stage_output[f'VCProbe{x + 1}'] for x in range(number_of_pixels)]) * 1E6  # uA
				full_return_current_ua = np.array([self.simulation_stage_output[f'VrCProbe{x + 1}'] for x in range(number_of_pixels)]) * 1E6  # uA

				# shift time vector to point of interest
				shifted_time_vector = self.simulation_stage_output['time'] * 1E3 - time_point_value

				# filter corresponding time values
				time_indices_to_include = (shifted_time_vector > 0.1) & (shifted_time_vector < pulse_duration)
				active_current_ua = full_active_current_ua[:, time_indices_to_include]
				return_current_ua = full_return_current_ua[:, time_indices_to_include]
				time_vector = shifted_time_vector[time_indices_to_include]

				print("adjusted time vector is: " + str(time_vector))

				active_current_ua = (active_current_ua[:, :-1] + active_current_ua[:, 1:]) / 2
				return_current_ua = (return_current_ua[:, :-1] + return_current_ua[:, 1:]) / 2
				Td = time_vector[1:] - time_vector[:-1]
				active_current_ua = np.sum(active_current_ua * Td, axis=1) / np.sum(Td)
				return_current_ua = np.sum(return_current_ua * Td, axis=1) / np.sum(Td)

				# initialize output structure for this time point
				voltage_3d_matrix = np.zeros(xx.shape + (len(z_values),))

				for z_index, z_value in enumerate(z_values):
					print("\n\nThe current depth being analyzed is: {}um".format(str(z_value)))

					V_elem = np.zeros([x_frame.size, y_frame.size, number_of_pixels * 2])
					V_elem_act = np.interp(dist_elem, active_x, active_voltage_mv[z_index, :])
					V_elem_ret = np.interp(dist_elem, return_x, return_voltage_mv[z_index, :])

					V_near = return_near_voltage_mv[(z_index) * x_return_near.size: (z_index + 1) * x_return_near.size, :]
					myfun = interpolate.RectBivariateSpline(x_return_near, y_return_near, V_near.T)

					idx_near = (xx_element < np.max(x_return_near)) & (yy_element < np.max(y_return_near))
					V_elem_ret[idx_near] = myfun.ev(xx_element[idx_near], yy_element[idx_near])

					voltage_xy_matrix = np.matmul(V_elem_act, active_current_ua) + np.matmul(V_elem_ret, return_current_ua)
					voltage_xy_matrix = np.reshape(voltage_xy_matrix, xx.shape)

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
					voltage_3d_matrix[:,:,z_index] = voltage_xy_matrix
					voltage_4d_matrix[:, :, :, time_point_index] = voltage_3d_matrix

			# prepare output
			processed_output_structure = {"v(x,y,z,t)_mv":voltage_4d_matrix,
										  "2d_mesh_um": (xx,yy),
										  "3d_mesh_um":(xxx, yyy, zzz),
										  "z_um":z_values,
										  "t_ms":time_points_to_analyze_ms,
										 "pixel_coordinates_um":pixel_coordinates}

			# visualize
			fig = VisualizationUtils.plot_3d_array(voltage_4d_matrix[:,:,:,0], "3D Potential at t=0 ms","x [um]", "y [um]",
												   "z [um]", "Potential [mV]", (xxx,yyy,zzz))
			return [processed_output_structure]








