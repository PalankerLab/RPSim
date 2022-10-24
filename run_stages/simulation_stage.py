import numpy as np

import PySpice
import PySpice.Unit as U

PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'xyce-serial'

from configuration.configuration_manager import Configuration

from run_stages.common_run_stage import CommonRunStage


class SimulationStage(CommonRunStage):
	"""
	This class implements the logic for the simulation stage, which executes the previously generated circuit in Xyce,
	while abiding by the structure required by the common run stage
	"""

	def __init__(self, *args):
		super().__init__(*args)
		self.circuit = self.get_stage_output_func("circuit", 0)
		self.video_sequence = self.get_stage_output_func("current_sequence", 0)

	def __str__(self):
		return "Simulation Execution Stage"

	@property
	def stage_name(self):
		return "simulation"

	@property
	def output_file_name(self):
		return ["simulation_results.pkl"]

	def run_stage(self, *args, **kwargs):
		"""
		This function holds the execution logic for the simulation stage
		:param args:
		:param kwargs:
		:return:
		"""
		# initialize simulator
		simulator = self.circuit.simulator(temperature=Configuration().params["temperature"],
										   nominal_temperature=Configuration().params["nominal_temperature"],
										   xyce_command = 'Xyce')
		# run simulation
		analysis = simulator.transient(step_time=(self.video_sequence['time_step']) @ U.u_ms,
									   end_time=(Configuration().params["simulation_duration"]) @ U.u_s)

		# select the nodes of interest to save. automatic completion of the suffixed numerical indices
		return self._format_output(analysis, sim_mode='trans',
								   nodes_select={'Pt', 'VCProbe', 'Saline', 'VrCProbe', 'rPt', 'rSaline'})

	@staticmethod
	def _strip_idx(node_name):
		"""
		This function removes the numerical suffix of a variable name to help classify.
		"""
		return node_name.rstrip(''.join(str(kk) for kk in range(10)))

	def _format_output(self, analysis, sim_mode=[], nodes_select={}, delay=0):
		"""
		This function converts the output from PySpice to a Python dictionary.
		"""
		sim_res_dict = {}

		for node in analysis.nodes.values():
			data_label = "%s" % str(node)
			if bool(nodes_select) and (self._strip_idx(data_label) not in nodes_select):
				continue
			sim_res_dict[data_label] = np.array(node)

		if sim_mode == 'trans':
			#need to record the time axis
			t = []
			for val in analysis.time:
				t.append(float(val))
			sim_res_dict['time'] = np.array(t) - delay * 1E-3

		return sim_res_dict
