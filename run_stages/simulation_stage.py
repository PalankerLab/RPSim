import numpy as np

import PySpice
import PySpice.Unit as U

PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'xyce-serial'

from configuration.configuration_manager import Configuration

from configuration.stages import RunStages
from run_stages.common_run_stage import CommonRunStage

from utilities.common_utilities import CommonUtils



class SimulationStage(CommonRunStage):
	"""
	This class implements the logic for the simulation stage, which executes the previously generated circuit in Xyce,
	while abiding by the structure required by the common run stage
	"""
	@property
	def stage_name(self):
		return RunStages.simulation.name

	def run_stage(self, *args, **kwargs):
		"""
		This function holds the execution logic for the simulation stage
		:param args:
		:param kwargs:
		:return:
		"""
		# get generated circuit and current sequence
		circuit = self.outputs_container[RunStages.circuit.name][0]
		video_sequence = self.outputs_container[RunStages.current_sequence.name][0]

		# initialize simulator
		simulator = circuit.simulator(temperature=Configuration().params["temperature"],
									  nominal_temperature=Configuration().params["nominal_temperature"],
									  xyce_command = 'Xyce')
		# run simulation
		analysis = simulator.transient(step_time=(video_sequence['time_step']) @ U.u_ms,
									   end_time=(Configuration().params["simulation_duration"]) @ U.u_s)

		# select the nodes of interest to save. automatic completion of the suffixed numerical indices
		return self._format_output(analysis, sim_mode='trans',
								   nodes_select={'Pt', 'VCProbe', 'Saline', 'VrCProbe', 'rPt', 'rSaline'})
	@staticmethod
	def _format_output(analysis, sim_mode=[], nodes_select={}, delay=0):
		"""
		This function converts the output from PySpice to a Python dictionary.
		"""
		sim_res_dict = {}

		for node in analysis.nodes.values():
			data_label = "%s" % str(node)
			if bool(nodes_select) and (CommonUtils.strip_idx(data_label) not in nodes_select):
				continue
			sim_res_dict[data_label] = np.array(node)

		if sim_mode == 'trans':
			#need to record the time axis
			t = []
			for val in analysis.time:
				t.append(float(val))
			sim_res_dict['time'] = np.array(t) - delay * 1E-3

		return sim_res_dict
