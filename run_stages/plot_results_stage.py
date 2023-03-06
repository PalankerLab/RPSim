import matplotlib.pyplot as plt

from run_stages.common_run_stage import CommonRunStage
from configuration.stages import RunStages


class PlotResultsStage(CommonRunStage):
	"""
	This class implements the logic for the plot results stage, which handles plotting the previously obtained
	simulation results, while abiding by the structure required by the common run stage
	"""
	def __init__(self, *args):
		super().__init__(*args)
		self.simulation_results = self.outputs_container[RunStages.simulation.name][0]

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

		# plot diode voltage as a function of time for a center diode and an edge diode
		fig1 = plt.figure()
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results['Pt99'] * 1E3, color='b', linewidth=1,label='Center')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results['Pt1'] * 1E3, color='r', linewidth=1,label='Edge')
		plt.legend(loc="best")
		plt.ylabel("Diode Voltage (mV)")
		plt.xlabel("Time (ms)")
		plt.grid()
		output_figures.append(fig1)

		# plot injected current [uA] as a function of time [ms] for a center electrode and an edge electrode
		fig2 = plt.figure()
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results['VCProbe99'] * 1E6, color='b',linewidth=1, label='Center')
		plt.plot(self.simulation_results['time'] * 1E3, self.simulation_results['VCProbe1'] * 1E6, color='r',linewidth=1, label='Edge')
		plt.legend(loc="best")
		plt.ylabel("Current ($\mu$A)")
		plt.xlabel("Time (ms)")
		plt.grid()
		output_figures.append(fig2)

		return output_figures


