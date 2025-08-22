import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings

from configuration.configuration_manager import Configuration
from run_stages.common_run_stage import CommonRunStage
from configuration.stages import RunStages
from run_stages.pattern_generation_stage import ImagePattern


class PlotResultsStage(CommonRunStage):
	"""
	This class implements the logic for the plot results stage, which handles plotting the previously obtained
	simulation results, while abiding by the structure required by the common run stage
	"""
	def __init__(self, *args):
		super().__init__(*args)
		self.simulation_results = self.outputs_container[RunStages.simulation.name][0]
		if RunStages.post_process.name in self.outputs_container:
			self.post_process_results = self.outputs_container[RunStages.post_process.name][0]

		# To plot the electrode map location we need their position.
		# The information is extracted through ImagePattern class.
		tmp = ImagePattern(pixel_size=Configuration().params["pixel_size"])
		dist_matrix = tmp.create_distance_matrix()
		self.central_electrode = tmp.determine_central_label(dist_matrix)
		self.pixel_labels = tmp.pixel_labels

		self.time_start_ms = Configuration().params["plot_time_window_start_ms"]
		self.time_end_ms = Configuration().params["plot_time_window_end_ms"]
		
		# Colors for plotting certain pixels
		self.color_center = (199/255, 35/255, 73/255, 1)
		self.color_edge = (44/255, 127/255, 173/255, 1)
		self.color_most = (250/255, 172/255, 34/255, 1)

	@property
	def stage_name(self):
		return RunStages.plot_results.name

	def run_stage(self):
		"""
		This function holds the execution logic for the plot results stage
		:param args:
		:param kwargs:
		:return:
		"""
		# initialize output
		output_figures = list()
		
		# New plotting displaying the most illuminated pixel as well 
		most_illuminated_electrode = self.outputs_container[RunStages.current_sequence.name][2] 
		# Edge electrode is still number 1, this is not automated but should be correct for all configurations
		edge_electrode = 1

		# plot diode voltage as a function of time for a center diode and an edge diode
		fig1 = plt.figure()
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'Pt{self.central_electrode}'] * 1E3, color=self.color_center, linewidth=1,label='Center')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'Pt{edge_electrode}'] * 1E3, color=self.color_edge, linewidth=1,label='Edge')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'Pt{most_illuminated_electrode}'] * 1E3, color=self.color_most, linewidth=1,label='Most illuminated')
		plt.legend(loc="best")
		plt.ylabel("Diode Voltage (mV)")
		plt.xlabel("Time (ms)")
		plt.grid()
		output_figures.append(fig1)

		# plot injected current [uA] as a function of time [ms] for a center electrode and an edge electrode
		fig2 = plt.figure()
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'VCProbe{self.central_electrode}'] * 1E6, color=self.color_center,linewidth=1, label='Center')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'VCProbe{edge_electrode}'] * 1E6, color=self.color_edge,linewidth=1, label='Edge')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'VCProbe{most_illuminated_electrode}'] * 1E6, color=self.color_most, linewidth=1, label='Most illuminated')
		plt.legend(loc="best")
		plt.ylabel("Current ($\mu$A)")
		plt.xlabel("Time (ms)")
		plt.grid()
		output_figures.append(fig2)

		# Plot the location of the electrodes
		fig3 = self.generate_electrodes_position(edge_electrode, most_illuminated_electrode) 
		output_figures.append(fig3)

		# Create an empty figure because the run manager expects four figures
		fig4 = plt.figure()
		if RunStages.post_process.name in self.outputs_container:
			# extract one diode that should be on
			time = self.post_process_results["on_diode_data"]["time_ms"]
			on_diodes = list(self.post_process_results["on_diode_data"].keys()) # Todo automate in case less than five pixels are illuminated below the threshold!
			n_diodes_above_threshold = len(on_diodes)
			if n_diodes_above_threshold > 0:
				idx = int(np.random.uniform(0, n_diodes_above_threshold))
				one_on_diode = on_diodes[idx]
			
				# plot this diode
				plt.plot(time, self.post_process_results["on_diode_data"][one_on_diode]["current"], linewidth=1)
				plt.ylabel("Current on diode {} (mV)".format(one_on_diode))
				plt.xlabel("Time (ms)")
				plt.grid()
			
				output_figures.append(fig4)
				
			else:
				warnings.warn(f"No diode were illuminated above the provided theshold {Configuration().params['on_diode_threshold_mV']}. Could not plot on diode.")

		fig5 = self.return_heat_map_currents()
		output_figures.append(fig5)

		if RunStages.post_process.name in self.outputs_container:
			fig6, fig7 = self.plot_cross_section_field()
		else:
			fig6, fig7 = None
		output_figures.extend([fig6, fig7])
		
		return output_figures

	def generate_electrodes_position(self, edge, most):
		"""
		This function generate an image of the plotted pixels. 
		Params: 
			pixel_labels  (Numpy.array): Array having the same size as implant_layout. Each entry corresponds to either 0 or the pixel label (the active and return electrodes are also labeled 0, only the photodiode is non-zero)
			central (int): The label of the central pixel 
			edge (int): The label of an edge pixel
			most (int): The label of the most illuminated pixel
		Return 
			figure (plt.fig)
		"""
		
		# Locate all the relevant positions
		mask_center = self.pixel_labels == self.central_electrode
		mask_edge = self.pixel_labels == edge 
		mask_most = self.pixel_labels == most
		mask_other = (self.pixel_labels > 0) & (self.pixel_labels != self.central_electrode) & (self.pixel_labels != edge) & (self.pixel_labels != most)

		# Assign the colors for each section (floating RGBA)
		color_background = (63/255, 35/255, 73/255, 0.33)
		color_other = (10/255,10/255,30/255,10/255)

		# Create an RGBA array with background color 
		array_shape =  (self.pixel_labels.shape[0], self.pixel_labels.shape[1], 4)
		image = np.full(array_shape, color_background, dtype=float)

		# Assign the colors of the relevant pixels
		image[mask_other] = color_other
		image[mask_center] = self.color_center
		image[mask_edge] = self.color_edge
		image[mask_most] = self.color_most

		# Prepare a nice legend
		colors = [color_background, color_other, self.color_center, self.color_edge, self.color_most]
		labels = ["Background", "Other pixels",  "Central", "Edge", "Most illuminated"]
		patches = [mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(labels)) ]
		
		# Plot the image and legend
		fig = plt.figure(figsize=(7,4)) # Plotting a legend outside of the box and saving the image is not straightforward, because the figure that was created do not take into account the outside legend. Hence the big floating space around
		plt.imshow(image)
		plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.5), loc= "center left", borderaxespad=0.)

		return fig
	
	def return_heat_map_currents(self):
		""""
		Plot a heatmap of the Simulation stage generated active currents,
		averaged over a certain time window. 

		To plot the current per pixel, this function requires the pixel label file,
		a numpy array filled with int, containing the number of the electrode at its
		corresponding location in the array.
		"""

		# Get the mask for extracting the currents to average
		time_ms = self.simulation_results['time'] * 1e3
		time_mask = (time_ms > self.time_start_ms) & (time_ms < self.time_end_ms)

		nb_pixels = self.pixel_labels.max()
		# For storing the output 
		map_avg_currents_ua = np.zeros(self.pixel_labels.shape)
		
		# Iterate over the pixels
		for pixel in range(1,nb_pixels+1):    
			# Get the active currents for the given pixel for the given time window
			# VCrProbe would be for the return currents in bipolar case
			currents = self.simulation_results[f'VCProbe{pixel}'][time_mask]
			# Select the pixel 
			pixel_mask = self.pixel_labels == pixel

			# Get the averaged current in that time window for the given pixel
			map_avg_currents_ua[pixel_mask] = np.mean(currents) * 1e6
			
		fig = plt.figure(figsize=(7,6))
		plt.imshow(map_avg_currents_ua, cmap='magma')
		plt.title(f"Averaged currents per pixel\n [{self.time_start_ms}-{self.time_end_ms} ms]", fontsize=21)
		plt.axis('off')

		cbar = plt.colorbar()
		cbar.set_label("Current [$\mu A$]", fontsize=18)
		cbar.ax.tick_params(labelsize=16)
		plt.tight_layout()

		return fig
	
	# def plot_cross_section_field(self):
	# 	"""
	# 	Plot the resulting electric field for a given z-slice, depth,
	# 	and given time window.
	# 	"""
	#
	# 	field = self.post_process_results["v(x,y,z,t)_mv"]
	# 	depth_um = Configuration().params["plot_potential_depth_um"]
	# 	existing_depths = self.post_process_results["z_um"]
	# 	existing_times = self.post_process_results["t_start_ms"]
	# 	# Extract the closest existing depth slice in the post process results
	# 	idx_depth = np.argmin(np.abs(np.array(existing_depths) - depth_um))
	# 	# Extract the closest existing time slice in the post process results
	# 	idx_time = np.argmin(np.abs(np.array(existing_times) - self.time_start_ms))
	#
	# 	# Get the correct frame size
	# 	frame_size = self.post_process_results["2d_mesh_um"]
	# 	min_x, max_x, min_y, max_y = frame_size[0].min(), frame_size[0].max(), frame_size[1].min(), frame_size[1].max()
	#
	# 	fig = plt.figure(figsize=(6, 5))
	# 	plt.imshow(field[:,:,idx_depth, idx_time], cmap='inferno', extent=(min_x, max_x, min_y, max_y))
	# 	plt.xticks(fontsize=11)
	# 	plt.yticks(fontsize=11)
	# 	plt.xlabel("x-Distance [$\mu m$]", fontsize=15)
	# 	plt.ylabel("y-Distance [$\mu m$]", fontsize=15)
	#
	# 	cbar = plt.colorbar()
	# 	cbar.set_label("Potential [$mV$]", fontsize=18)
	# 	cbar.ax.tick_params(labelsize=16)
	#
	# 	plt.title(f'Potential in the retina\n Z={existing_depths[idx_depth]} $\mu m$ [{existing_times[idx_time]:.1f}-{existing_times[idx_time] + Configuration().params["time_averaging_resolution_ms"]:.1f} ms]', fontsize=20)
	# 	plt.tight_layout()
	#
	# 	return fig

	def plot_cross_section_field(self):
		field = self.post_process_results["v(x,y,z,t)_mv"]
		depth_um = Configuration().params["plot_potential_depth_um"]
		time_window_end = Configuration().params["plot_time_window_end_ms"]

		existing_depths = self.post_process_results["z_um"]
		existing_times = self.post_process_results["t_start_ms"]

		# extract the closest existing depth slice
		idx_depth = np.argmin(np.abs(np.array(existing_depths) - depth_um))

		# get the correct frame size
		frame_size = self.post_process_results["2d_mesh_um"]
		min_x, max_x = frame_size[0].min(), frame_size[0].max()
		min_y, max_y = frame_size[1].min(), frame_size[1].max()

		# slice: xy plane over all times at this z
		xy_t = field[:, :, idx_depth, :]

		# find time index of max and min
		t_max_idx = np.argmax(xy_t.max(axis=(0, 1)))
		t_min_idx = np.argmin(xy_t.min(axis=(0, 1)))

		# compute average over all times
		xy_avg = xy_t.mean(axis=2)

		# compute average over requested time window
		idx_time_start = np.argmin(np.abs(np.array(existing_times) - self.time_start_ms))
		idx_time_end = np.argmin(np.abs(np.array(existing_times) - time_window_end))
		if idx_time_start > idx_time_end:
			idx_time_start, idx_time_end = idx_time_end, idx_time_start
		xy_avg_window = xy_t[:, :, idx_time_start:idx_time_end + 1].mean(axis=2)

		# -------- summary figure (2x2) --------
		fig_summary, axes = plt.subplots(2, 2, figsize=(12, 10))

		def add_panel(ax, data, title):
			im = ax.imshow(data, cmap="inferno", extent=(min_x, max_x, min_y, max_y))
			ax.set_title(title, fontsize=12)
			ax.set_xlabel("x-Distance [$\mu m$]")
			ax.set_ylabel("y-Distance [$\mu m$]")
			cbar = fig_summary.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
			cmin, cmax = im.get_clim()
			cbar.set_ticks([cmin, cmax])
			cbar.set_ticklabels([f"{cmin:.1f}", f"{cmax:.1f}"])
			return im

		add_panel(axes[0, 0], xy_t[:, :, t_max_idx],
				  f"Max potential\nZ={existing_depths[idx_depth]} μm, t={existing_times[t_max_idx]:.1f} ms")
		add_panel(axes[0, 1], xy_t[:, :, t_min_idx],
				  f"Min potential\nZ={existing_depths[idx_depth]} μm, t={existing_times[t_min_idx]:.1f} ms")
		add_panel(axes[1, 0], xy_avg,
				  f"Average across all times\nZ={existing_depths[idx_depth]} μm")
		add_panel(axes[1, 1], xy_avg_window,
				  f"Average {existing_times[idx_time_start]:.1f}–{existing_times[idx_time_end]:.1f} ms\nZ={existing_depths[idx_depth]} μm")

		plt.tight_layout()

		# -------- all time indices figure --------
		n_times = xy_t.shape[2]
		ncols = int(np.ceil(np.sqrt(n_times)))
		nrows = int(np.ceil(n_times / ncols))

		fig_all, axes_all = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
		axes_all = axes_all.flatten()

		for i in range(n_times):
			im = axes_all[i].imshow(xy_t[:, :, i], cmap="inferno", extent=(min_x, max_x, min_y, max_y))
			axes_all[i].set_title(f"t={existing_times[i]:.1f} ms", fontsize=9)
			axes_all[i].set_xticks([])
			axes_all[i].set_yticks([])
			cbar = fig_all.colorbar(im, ax=axes_all[i], fraction=0.046, pad=0.04)
			cmin, cmax = im.get_clim()
			cbar.set_ticks([cmin, cmax])
			cbar.set_ticklabels([f"{cmin:.1f}", f"{cmax:.1f}"])

		# remove unused subplots if any
		for j in range(n_times, len(axes_all)):
			fig_all.delaxes(axes_all[j])

		plt.tight_layout()

		return fig_summary, fig_all

