import matplotlib.pyplot as plt

from run_stages.common_run_stage import CommonRunStage


class PlotResultsStage(CommonRunStage):
	"""
	This class implements the logic for the plot results stage, which handles plotting the previously obtained
	simulation results, while abiding by the structure required by the common run stage
	"""
	def __init__(self, *args):
		super().__init__(*args)
		self.simulation_results = self.get_stage_output_func("simulation", 0)

	def __str__(self):
		return "Plot Results Stage"

	@property
	def stage_name(self):
		return "plot_results"

	@property
	def output_file_name(self):
		return ["figure_1.png", "figure_2.png"]

	@property
	def output_as_figure(self):
		return True

	def run_stage(self):
		"""
		This function holds the execution logic for the plot results stage
		:param args:
		:param kwargs:
		:return:
		"""
		fig1 = plt.figure()
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results['Pt99'] * 1E3, color='b', linewidth=1,
				 label='Center')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results['Pt1'] * 1E3, color='r', linewidth=1,
				 label='Edge')
		# plt.xlim(768, 800)
		plt.legend(loc="lower right")
		plt.ylabel("Diode Voltage (mV)")
		plt.xlabel("Time (ms)")
		plt.grid()

		fig2 = plt.figure()
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results['VCProbe99'] * 1E6, color='b',
				 linewidth=1, label='Center')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results['VCProbe1'] * 1E6, color='r',
				 linewidth=1, label='Edge')
		# plt.xlim(950, 1120)
		plt.legend(loc="upper right")
		plt.ylabel("Current ($\mu$A)")
		plt.xlabel("Time (ms)")
		plt.grid()

		return fig1, fig2


