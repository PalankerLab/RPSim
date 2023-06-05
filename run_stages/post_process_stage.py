import glob
import multiprocessing
import os

import matplotlib
from PIL import Image
import sys
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from configuration.configuration_manager import Configuration
from configuration.models import Models
from configuration.stages import RunStages

from run_stages.common_run_stage import CommonRunStage


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
			self.start_time_ms = Configuration().params["pulse_start_time_in_ms"]
			self.pulse_duration = Configuration().params["stimulation_duration_in_ms"]
			self.averaging_resolution_ms = self.pulse_duration
			self.time_points_to_analyze_ms = [self.start_time_ms]

			if Configuration().params["analyze_time_dynamics"]:
				self.averaging_resolution_ms = 1
				self.time_points_to_analyze_ms = list(np.arange(start=self.start_time_ms,
																stop=self.start_time_ms + self.pulse_duration * 2,
																step=self.averaging_resolution_ms))

			# define depth resolution
			default_depth_params =  [x*1 for x in range(150)] #[x * 5 for x in range(11)] + [x * 5 + 57 for x in range(20)]
			self.z_values = Configuration().params["depth_values_in_um"] if Configuration().params.get("depth_values_in_um") else default_depth_params

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
		time_indices_to_include = (shifted_time_vector > 0.1) & (shifted_time_vector < self.averaging_resolution_ms)
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

		if Configuration().params["model"] == Models.MONOPOLAR.value:

			pulse_width = Configuration().params["pulse_duration_ms"]
			pulse_start = Configuration().params["pulse_start_time_in_ms"]

			V_dict = np.loadtxt(Configuration().params["r_matrix_input_file_active"], delimiter=',')  # mV
			X_act = V_dict[0, :]
			V_dict_act = V_dict[1:, :]

			V_dict = np.loadtxt(Configuration().params["r_matrix_input_file_return"], delimiter=',')  # mV
			x_frame = V_dict[0, :]
			y_frame = V_dict[1, :]
			V_dict_ret = V_dict[2:, :]

			px_pos = np.loadtxt(Configuration().params["r_matrix_input_file_px_pos"], delimiter=',')  # mV
			print(px_pos)

			number_of_pixels = px_pos.shape[0]

			XX, YY = np.meshgrid(x_frame, y_frame)
			XX_elem = np.zeros([x_frame.size * y_frame.size, number_of_pixels])
			YY_elem = np.zeros([x_frame.size * y_frame.size, number_of_pixels])
			for kk in range(number_of_pixels):
				XX_elem[:, kk] = np.abs(XX.flatten() - px_pos[kk, 0])
				YY_elem[:, kk] = np.abs(YY.flatten() - px_pos[kk, 1])

			dist_elem = np.sqrt(XX_elem ** 2 + YY_elem ** 2)

			I_act_t = np.array([self.simulation_stage_output[f'VCProbe{x + 1}'] for x in range(number_of_pixels)]) * 1E6  # uA
			I_ret_t = np.array(self.simulation_stage_output[f'VCProbe{0}']) * 1E6  # uA

			T = self.simulation_stage_output['time'] * 1E3 - pulse_start
			t_idx = (T > 0.1) & (T < pulse_width)
			I_act = I_act_t[:, t_idx]
			I_ret = I_ret_t[t_idx]
			T = T[t_idx]

			I_act = (I_act[:, :-1] + I_act[:, 1:]) / 2
			I_ret = (I_ret[:-1] + I_ret[1:]) / 2
			Td = T[1:] - T[:-1]
			I_act = np.sum(I_act * Td, axis=1) / np.sum(Td)
			I_ret = np.sum(I_ret * Td) / np.sum(Td)

			self.z_values = [x*1 for x in range(151)] #[x * 5 for x in range(11)] + [x * 5 + 57 for x in range(10)]
			voltage_3d_matrix = np.zeros(XX.shape + (len(self.z_values),))

			# create voltage x-y matrix for each z plane
			for z_idx in range(len(self.z_values)):
				V_elem_act = np.interp(dist_elem, X_act, V_dict_act[z_idx, :])
				V_ret = V_dict_ret[(z_idx) * x_frame.size: (z_idx + 1) * x_frame.size, :]

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
								 "pixel_coordinates_um": px_pos}

			#frame_width = x_frame.size
			frame_width = abs(x_frame[0])

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

			# # run post-processing concurrently for all the time points
			# with Pool(multiprocessing.cpu_count()//3) as pool:
			# 	results = pool.map(self._generate_potential_matrix_per_time_section, self.time_points_to_analyze_ms)
			#
			# for result in results:
			# 	voltage_4d_matrix[:, :, :, self.time_points_to_analyze_ms.index(result[0])] = result[1]

			# iterate over time and create 3D potential matrix for each time point
			for time_point_index, time_point_value in enumerate(self.time_points_to_analyze_ms):
				# calculate metrix for this time point
				voltage_3d_matrix = self._generate_potential_matrix_per_time_section(start_time=time_point_value)
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








