import os
import pickle

import numpy as np

import PySpice
import PySpice.Unit as U
from PySpice.Spice.Netlist import Circuit

PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'xyce-serial'

from configuration.models import Models
from configuration.stages import RunStages
from configuration.configuration_manager import Configuration

from run_stages.common_run_stage import CommonRunStage

from circuit_building_blocks.pixel import Pixel
from circuit_building_blocks.electrode import SIROF
from circuit_building_blocks.frame_driver import FramesDriver
from circuit_building_blocks.image_driver import ImageDriver

from utilities.image_processing_utilities import is_edge, Rmat_simp


class CircuitStage(CommonRunStage):
	"""
	This class implements the logic for the circuit stage, which generates the needed circuit, while abiding by the
	structure required by the common run stage
	"""

	def __init__(self, *args):
		super().__init__(*args)
		# initialize output
		self.circuit = None

		# get previous stages output
		self.video_sequence = self.outputs_container[RunStages.current_sequence.name][0]
		self.resistive_mesh = self.outputs_container[RunStages.resistive_mesh.name][0]
		self.number_of_pixels = Configuration().params["number_of_pixels"]

		# get current model
		self.is_bipolar = Configuration().params["model"] == Models.BIPOLAR.value

		# calculate parameters based on given configuration
		self.sirof_active_capacitance_nF = Configuration().params["sirof_capacitance"] * np.pi * Configuration().params[
			"active_electrode_radius"] ** 2 * 1E-2

		if self.is_bipolar:
			self.return_to_active_area_ratio = np.sqrt(3) / 2 * \
											   (Configuration().params["pixel_size"] ** 2 - Configuration().params[
												   "photosensitive_area_edge_to_edge"] ** 2) / \
											   (np.pi * Configuration().params["active_electrode_radius"] ** 2)
		else:
			self.return_to_active_area_ratio = Configuration().params["return_to_active_area_ratio"]

		self.photosensitive_area = Configuration().params.get("photosensitive_area") 
		
		# add compensation matrix, if requested
		self.G_comp_flag = False
		Gs_new = 1 / Configuration().params['shunt_resistance'] if Configuration().params['shunt_resistance'] else 0
		if Configuration().params['r_matrix_simp_ratio'] < 1:
			if Configuration().params["model"]== Models.BIPOLAR.value:
				imag_basis = np.array(self.video_sequence["Frames"]).reshape((self.number_of_pixels*2, -1))
			if Configuration().params["model"] == Models.MONOPOLAR.value:
				imag_basis = np.array(self.video_sequence["Frames"]).reshape((self.number_of_pixels, -1))
			col_norm = np.linalg.norm(imag_basis, axis=0)
			imag_basis = imag_basis[:, col_norm > 1E-6]
			(self.resistive_mesh, self.G_comp) = Rmat_simp(Rmat=self.resistive_mesh,
														   Gs=Gs_new,
														   ratio=Configuration().params['r_matrix_simp_ratio'],
														   imag_basis=imag_basis)
			self.G_comp_flag = True

		# get edges
		if Configuration().params.get("r_matrix_input_file_px_pos") and self.is_bipolar:
			self.is_edge = is_edge(np.loadtxt(Configuration().params["r_matrix_input_file_px_pos"], delimiter=','),
								   Configuration().params["pixel_size"])
			self.edge_factor = Configuration().params["additional_edges"] / np.sum(self.is_edge) / 6

		# get initial voltage
		self._configure_initial_voltage()

	@property
	def stage_name(self):
		return RunStages.circuit.name

	def _configure_initial_voltage(self):
		self.initial_Vactive = Configuration().params["initial_Vactive"] if Configuration().params.get(
			"initial_Vactive") else 0

		if isinstance(self.initial_Vactive, tuple):
			# if tuple is provided with a bipolar design, raise an error
			if self.is_bipolar:
				raise Exception("Bipolar mode does not support importing state.")

			# else, for a monopolar design, load provided path
			full_path = os.path.join(Configuration().params["user_output_path"], self.initial_Vactive[0],
									 'simulation_results.pkl')
			with open(full_path, 'rb') as f:
				sim_res_dict = pickle.load(f)

			t = sim_res_dict['time'] * 1E3 - self.initial_Vactive[1]
			self.Vini_act = np.array([np.interp(0, t, sim_res_dict[f'Pt{x + 1}'] - sim_res_dict[f'Saline{x + 1}'])
									  for x in range(Configuration().params["number_of_pixels"])])  # V
			self.Vini_ret = np.interp(0, t, sim_res_dict[f'Pt{0}'] - sim_res_dict[f'Saline{0}'])  # V

		else:
			self.Vini_act = self.initial_Vactive * np.ones(Configuration().params["number_of_pixels"])
			self.Vini_ret = -self.initial_Vactive / self.return_to_active_area_ratio

	def run_stage(self, *args, **kwargs):
		"""
		This function holds the execution logic for the circuit stage
		:param args:
		:param kwargs:
		:return:
		"""
		self.circuit = Circuit('My_Circuit')

		# --------Image sequence controller----------
		# mimic an ideal switch with a continuous element
		self.circuit.model('SW1V', 'VSWITCH', Ron=1 @ U.u_Ohm, Roff=1 @ U.u_MOhm, Voff=0.45 @ U.u_V, Von=0.55 @ U.u_V)
		self.circuit.subcircuit(
			ImageDriver('ImgDr', image_T=[x * self.video_sequence['duration_frames_ms'] for x in self.video_sequence['nb_repetitions_frames']],
						delay=2))

		# number of output ports depends on the number of images
		image_control_ports = [f'ImgON{x + 1}' for x in range(len(self.video_sequence['nb_repetitions_frames']))]
		self.circuit.X('ImgCtl', 'ImgDr', *tuple([self.circuit.gnd] + image_control_ports))
		for px_idx in range(1, self.number_of_pixels + 1):
			self.circuit.subcircuit(FramesDriver(f'fd{px_idx}',
												 frame_val=self.video_sequence['Frames'][px_idx - 1],
												 frame_T=self.video_sequence['duration_subframes_ms'],
												 time_step=self.video_sequence['time_step'],
												 Apho_factor=Configuration().params[
													 "Ipho_scaling"] if Configuration().params.get(
													 "Ipho_scaling") else 1))
			# switch between subframes
			self.circuit.X(f'FrameCtl{px_idx}', f'fd{px_idx}', *tuple([self.circuit.gnd, f'Si{px_idx}'] +
																	  image_control_ports))

		# --------Photovoltaic pixels----------
		# definition of the diode
		self.circuit.model('Diode_B6', 'D',
						   IS=Configuration().params["Isat"] @ U.u_pA,
						   BV=22 @ U.u_V,
						   N=Configuration().params["ideality_factor"] * Configuration().params["number_of_diodes"])

		self.circuit.subcircuit(Pixel('pixel', shunt=Configuration().params["shunt_resistance"]))

		for px_idx in range(1, self.number_of_pixels + 1):
			self.circuit.X(f'Pixel{px_idx}', 'pixel', self.circuit.gnd, f'Si{px_idx}')

		# --------Electrodes----------
		self.circuit.subcircuit(
			SIROF('active',
				  c0=self.sirof_active_capacitance_nF))

		self.circuit.subcircuit(
			SIROF('return',
				  scaling=self.return_to_active_area_ratio * (1 if self.is_bipolar else self.number_of_pixels),
				  c0=self.sirof_active_capacitance_nF))

		if self.is_bipolar:
			edge_ar = self.return_to_active_area_ratio * (1 + self.edge_factor)
			self.circuit.subcircuit(SIROF('return_edge',
										  Vini=-self.initial_Vactive / edge_ar,
										  scaling=edge_ar,
										  c0=self.sirof_active_capacitance_nF))
		else:
			self.circuit.V(f'CProbe{0}', self.circuit.gnd, f'Pt{0}', 0 @ U.u_V)
			self.circuit.X('Return', f"return PARAMS: Vini={self.Vini_ret}", f'Pt{0}', f'Saline{0}')

		for px_idx in range(1, self.number_of_pixels + 1):
			# configure the active electrodes and their connections to the GND
			self.circuit.V(f'CProbe{px_idx}', f'Si{px_idx}', f'Pt{px_idx}', 0 @ U.u_V)  # current probe
			self.circuit.X(f'Active{px_idx}', f"active PARAMS: Vini="f"{self.Vini_act[px_idx - 1]}", f'Pt{px_idx}',
						   f'Saline{px_idx}')
			self.circuit.R(f'{px_idx}_{px_idx}', f'Saline{px_idx}', f'Saline{0}',
						   "{:.3e}".format(self.resistive_mesh[px_idx - 1, px_idx - 1]))

			if self.is_bipolar:
				# configure return electrodes and their connections to the GND
				self.circuit.V(f'rCProbe{px_idx}', self.circuit.gnd, f'rPt{px_idx}', 0 @ U.u_V)
				self.circuit.X(f'Return{px_idx}', 'return' + ("_edge" if self.is_edge[px_idx - 1] else ""),
							   f'rPt{px_idx}', f'rSaline{px_idx}')

				self.circuit.R(f'r{px_idx}_{px_idx}', f'rSaline{px_idx}', f'Saline{0}',
							   "{:.3e}".format(self.resistive_mesh[
												   px_idx + self.number_of_pixels - 1, px_idx + self.number_of_pixels - 1]))
				# connections between each pair of active and return
				for cross_idx in range(1, self.number_of_pixels + 1):
					R_cross = self.resistive_mesh[px_idx - 1, cross_idx + self.number_of_pixels - 1]
					if np.isnan(R_cross):
						continue
					self.circuit.R(f'ar{cross_idx}_{px_idx}', f'Saline{cross_idx}', f'rSaline{px_idx}',
						"{:.3e}".format(R_cross))

			# 

			# now define the interconnected resistor mesh
			for cross_idx in range(1, px_idx):
				R = self.resistive_mesh[px_idx - 1, cross_idx - 1]
				if np.isnan(R):
					continue
				# interconnection among the active electrodes
				self.circuit.R(f'{cross_idx}_{px_idx}', f'Saline{cross_idx}', f'Saline{px_idx}', "{:.3e}".format(R))
				if self.is_bipolar:
					# interconnection among the return electrodes
					R = self.resistive_mesh[px_idx + self.number_of_pixels - 1, cross_idx + self.number_of_pixels - 1]
					if np.isnan(R):
						continue
					self.circuit.R(f'r{cross_idx}_{px_idx}', f'rSaline{cross_idx}', f'rSaline{px_idx}',
								   "{:.3e}".format(R))

		if self.G_comp_flag:
			for comp_idx in range(self.G_comp['v_basis'].shape[1]):
				self.circuit.R(f'comp{comp_idx}', f'comp{comp_idx}', self.circuit.gnd, 1)
				for px_idx in range(1, self.number_of_pixels + 1):
					self.circuit.VCCS(f'px{px_idx}_comp{comp_idx}', self.circuit.gnd, f'comp{comp_idx}',
									f'Saline{px_idx}', f'Saline{0}',
									"{:.3e}".format(self.G_comp['v_basis'][px_idx - 1, comp_idx]))
					self.circuit.VCCS(f'comp{comp_idx}_px{px_idx}', f'Saline{px_idx}', f'Saline{0}',
									f'comp{comp_idx}', self.circuit.gnd,
									"{:.3e}".format(self.G_comp['i_basis'][px_idx - 1, comp_idx]))
					if self.is_bipolar:
							number_of_returns = self.number_of_pixels
							for ret_idx in range(1, number_of_returns + 1):
								self.circuit.VCCS(f'ret{ret_idx}_comp{comp_idx}', self.circuit.gnd, f'comp{comp_idx}',
												f'rSaline{ret_idx}', f'Saline{0}', "{:.3e}".format(self.G_comp['v_basis'][self.number_of_pixels + ret_idx-1, comp_idx]))
								self.circuit.VCCS(f'comp{comp_idx}_ret{ret_idx}', f'rSaline{ret_idx}', f'Saline{0}',
												f'comp{comp_idx}', self.circuit.gnd, "{:.3e}".format(self.G_comp['i_basis'][self.number_of_pixels + ret_idx-1, comp_idx]))
		return self.circuit
