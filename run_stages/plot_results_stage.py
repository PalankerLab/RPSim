import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
		self.post_process_results = self.outputs_container[RunStages.post_process.name][0]

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

		# New! We determine the actual central pixel of any implant configuration, 
		# rather than taking electrode 99. We need the ImagePattern class to determine that value.
		tmp = ImagePattern(pixel_size = Configuration().params["pixel_size"])
		dist_matrix = tmp.create_distance_matrix()
		central_electrode = tmp.determine_central_label(dist_matrix)
		pixel_labels = tmp.pixel_labels
		
		# New plotting displaying the most illuminated pixel as well 
		most_illuminated_electrode = self.outputs_container[RunStages.current_sequence.name][2] 
		# Edge electrode is still number 1, this is not automated but should be correct for all configurations
		edge_electrode = 1

		# plot diode voltage as a function of time for a center diode and an edge diode
		fig1 = plt.figure()
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'Pt{central_electrode}'] * 1E3, color='b', linewidth=1,label='Center')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'Pt{edge_electrode}'] * 1E3, color='r', linewidth=1,label='Edge')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'Pt{most_illuminated_electrode}'] * 1E3, color='r', linewidth=1,label='Most illuminated')
		plt.legend(loc="best")
		plt.ylabel("Diode Voltage (mV)")
		plt.xlabel("Time (ms)")
		plt.grid()
		output_figures.append(fig1)

		# plot injected current [uA] as a function of time [ms] for a center electrode and an edge electrode
		fig2 = plt.figure()
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'VCProbe{central_electrode}'] * 1E6, color='b',linewidth=1, label='Center')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'VCProbe{edge_electrode}'] * 1E6, color='r',linewidth=1, label='Edge')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results[f'VCProbe{most_illuminated_electrode}'] * 1E6, color='r',linewidth=1, label='Most illuminated')
		plt.legend(loc="best")
		plt.ylabel("Current ($\mu$A)")
		plt.xlabel("Time (ms)")
		plt.grid()
		output_figures.append(fig2)

		# Plot the location of the electrodes
		fig3 = plt.figure()
		image, patches = self.generate_electrodes_position(pixel_labels, central_electrode, edge_electrode, most_illuminated_electrode) 
		plt.imshow(image)
		#plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. ) # Top right
		plt.legend(handles=patches, bbox_to_anchor=(1.6, 0.5), loc=7, borderaxespad=0. ) # Center right
		output_figures.append(fig3)
		
		# extract one diode that should be on
		time = self.post_process_results["on_diode_data"]["time_ms"]
		on_diodes = list(self.post_process_results["on_diode_data"].keys())
		one_on_diode = on_diodes[5]
		
		# plot this diode
		fig4 = plt.figure()
		plt.plot(time, self.post_process_results["on_diode_data"][one_on_diode]["current"], linewidth=1)
		plt.ylabel("Current on diode {} (mV)".format(one_on_diode))
		plt.xlabel("Time (ms)")
		plt.grid()
		output_figures.append(fig4)

		return output_figures

	def generate_electrodes_position(self, pixel_labels, central, edge, most):
		"""
		This function generate an image of the plotted pixels. 
		Params: 
			pixel_labels  (Numpy.array): Array having the same size as implant_layout. Each entry corresponds to either 0 or the pixel label (the active and return electrodes are also labeled 0, only the photodiode is non-zero)
			central (int): The label of the central pixel 
			edge (int): The label of an edge pixel
			most (int): The label of the most illuminated pixel
		Return 
			image (Numpy array (x, x, 4)): an RGBA array representing the the electrodes positions
			patches (matplotlib.patches): the legend handles
		"""
		
		# Locate all the relevant positions
		mask_center = pixel_labels == central
		mask_edge = pixel_labels == edge 
		mask_most = pixel_labels == most
		mask_other = (pixel_labels > 0) & (pixel_labels != central) & (pixel_labels != edge) & (pixel_labels != most)

		# Assign the colors for each section (floating RGBA)
		color_background = (63/255, 35/255, 73/255, 0.33)
		color_other = (10/255,10/255,30/255,10/255)
		color_center = (199/255, 35/255, 73/255, 1)
		color_edge = (44/255, 127/255, 173/255, 1)
		color_most = (250/255, 172/255, 34/255, 1)

		# Create an RGBA array with background color 
		array_shape =  (pixel_labels.shape[0], pixel_labels.shape[1], 4)
		image = np.full(array_shape, color_background, dtype=float)

		# Assign the colors of the relevant pixels
		image[mask_other] = color_other
		image[mask_center] = color_center
		image[mask_edge] = color_edge
		image[mask_most] = color_most

		# Prepare a nice legend
		colors = [color_background, color_other, color_center, color_edge, color_most]
		labels = ["Background", "Other pixels",  "Central", "Edge", "Most illuminated"]
		patches = [mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(labels)) ]

		return image, patches 

