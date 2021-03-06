import PySpice
import PySpice.Unit as U
from PySpice.Spice.Netlist import Circuit
import numpy as np

PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'xyce-serial'

from configuration.configuration_manager import Configuration
from configuration.models import Models

from run_stages.common_run_stage import CommonRunStage

from circuit_building_blocks.pixel import Pixel
from circuit_building_blocks.electrode import SIROF
from circuit_building_blocks.frame_driver import FramesDriver
from circuit_building_blocks.image_driver import ImageDriver

from utilities.image_processing_utilities import isEdge


class CircuitStage(CommonRunStage):
	"""
	This class implements the logic for the circuit stage, which generates the needed circuit, while abiding by the
	structure required by the common run stage
	"""
	def __init__(self, *args):
		super().__init__(*args)
		self.circuit = None
		self.video_sequence = self.get_stage_output_func("current_sequence", 0)
		self.resistive_mesh = self.get_stage_output_func("resistive_mesh", 0)
		self.number_of_pixels = len(self.video_sequence['Frames'])
		self.simulation_results = None

		if Configuration().params.get("r_matrix_input_file_px_pos"):
			px_pos = np.loadtxt(Configuration().params["r_matrix_input_file_px_pos"], delimiter=',')
			self.is_edge = isEdge(px_pos, Configuration().params["pixel_size"])
			self.edge_factor = Configuration().params["additional_edges"] / np.sum(self.is_edge) / 6

	def __str__(self):
		return "Circuit Generation Stage"

	@property
	def stage_name(self):
		return "circuit"

	@property
	def output_file_name(self):
		return [Configuration().params["netlist_output_file"]]

	@property
	def output_as_pickle(self):
		return False

	def run_stage(self, *args, **kwargs):
		"""
		This function holds the execution logic for the circuit stage
		:param args:
		:param kwargs:
		:return:
		"""
		isBipolar = Configuration().params["model"] == Models.BIPOLAR.value
		self.circuit = Circuit('My_Circuit')

		# --------Image sequence controller----------
		# mimic an ideal switch with a continuous element
		self.circuit.model('SW1V', 'VSWITCH', Ron=1 @ U.u_Ohm, Roff=1 @ U.u_MOhm, Voff=0.45 @ U.u_V, Von=0.55 @ U.u_V)
		self.circuit.subcircuit(
			ImageDriver('ImgDr', image_T=[x * self.video_sequence['T_frame'] for x in self.video_sequence['L_images']],
						delay=2))		
		# number of output ports depends on the number of images
		image_control_ports = [f'ImgON{x + 1}' for x in range(len(self.video_sequence['L_images']))]
		self.circuit.X('ImgCtl', 'ImgDr', *tuple([self.circuit.gnd] + image_control_ports))
		for px_idx in range(1, self.number_of_pixels + 1):
			self.circuit.subcircuit(FramesDriver(f'fd{px_idx}', frame_val=self.video_sequence['Frames'][px_idx - 1],
												frame_T=self.video_sequence['T_subframes'],
												time_step=self.video_sequence['time_step'],
												Apho_factor=Configuration().params["Ipho_scaling"]
												if "Ipho_scaling" in Configuration().params.keys() else 1))
			# switch between subframes
			self.circuit.X(f'FrameCtl{px_idx}', f'fd{px_idx}', *tuple([self.circuit.gnd, f'Si{px_idx}'] +
												image_control_ports))

		# --------Photovoltaic pixels----------
		# definition of the diode
		self.circuit.model('Diode_B6', 'D', IS=Configuration().params["Isat"] @ U.u_pA, BV=22 @ U.u_V, 
						N=Configuration().params["ideality_factor"]*Configuration().params["number_of_diodes"])
		self.circuit.subcircuit(Pixel('pixel', shunt=Configuration().params["shunt_resistance"]))
		for px_idx in range(1, self.number_of_pixels + 1):
			self.circuit.X(f'Pixel{px_idx}', 'pixel', self.circuit.gnd, f'Si{px_idx}')

		# --------Electrodes----------
		self.circuit.subcircuit(SIROF('active', Vini=Configuration().params["initial_Vactive"],
									  c0=Configuration().params["sirof_active_capacitance_nF"]))
		self.circuit.subcircuit(SIROF('return',
			Vini=-Configuration().params["initial_Vactive"] / Configuration().params["return_to_active_area_ratio"],
			scaling= Configuration().params["return_to_active_area_ratio"] * (1 if isBipolar else self.number_of_pixels),
			c0=Configuration().params["sirof_active_capacitance_nF"]))
		if isBipolar:
			edge_ar = Configuration().params["return_to_active_area_ratio"] * (1 + self.edge_factor)
			self.circuit.subcircuit(SIROF('return_edge',	Vini=-Configuration().params["initial_Vactive"] / edge_ar,
				scaling=edge_ar, c0=Configuration().params["sirof_active_capacitance_nF"]))
		else:
			self.circuit.V(f'CProbe{0}', self.circuit.gnd, f'Pt{0}', 0 @U.u_V)
			self.circuit.X('Return', 'return', f'Saline{0}', f'Pt{0}')

		for px_idx in range(1, self.number_of_pixels + 1):
			#configure the active electrodes and their connections to the GND
			self.circuit.V(f'CProbe{px_idx}', f'Si{px_idx}', f'Pt{px_idx}', 0 @U.u_V)  # current probe
			self.circuit.X(f'Active{px_idx}', 'active', f'Pt{px_idx}', f'Saline{px_idx}')
			self.circuit.R(f'{px_idx}_{px_idx}', f'Saline{px_idx}', f'Saline{0}',self.resistive_mesh[px_idx - 1, px_idx - 1] @ U.u_Ohm)
			
			if isBipolar:
				#configure return electrodes and their connections to the GND
				self.circuit.V(f'rCProbe{px_idx}', self.circuit.gnd, f'rPt{px_idx}', 0@U.u_V)
				self.circuit.X(f'Return{px_idx}', 'return' + ("_edge" if self.is_edge[px_idx-1] else ""), f'rSaline{px_idx}', f'rPt{px_idx}')
				
				self.circuit.R(f'r{px_idx}_{px_idx}', f'rSaline{px_idx}', f'Saline{0}', self.resistive_mesh[px_idx+self.number_of_pixels-1, px_idx+self.number_of_pixels-1]@U.u_Ohm)
				# connections between each pair of active and return
				for cross_idx in range(1, self.number_of_pixels+1):
					self.circuit.R(f'ar{cross_idx}_{px_idx}', f'Saline{cross_idx}', f'rSaline{px_idx}',
						self.resistive_mesh[px_idx-1, cross_idx+self.number_of_pixels-1]@U.u_Ohm)

			# now define the interconnected resistor mesh
			for cross_idx in range(1, px_idx):
				#interconnection among the active electrodes
				self.circuit.R(f'{cross_idx}_{px_idx}', f'Saline{cross_idx}', f'Saline{px_idx}',
							   self.resistive_mesh[px_idx - 1, cross_idx - 1] @ U.u_Ohm)
				if isBipolar:
					#interconnection among the return electrodes
					self.circuit.R(f'r{cross_idx}_{px_idx}', f'rSaline{cross_idx}', f'rSaline{px_idx}', self.resistive_mesh[px_idx+self.number_of_pixels-1, cross_idx+self.number_of_pixels-1]@U.u_Ohm)
		
		return self.circuit

