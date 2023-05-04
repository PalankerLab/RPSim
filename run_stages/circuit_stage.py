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

from utilities.image_processing_utilities import is_edge
from utilities.matrix_utilities import Rmat_simp
import pickle as pkl


class CircuitStage(CommonRunStage):
	"""
	This class implements the logic for the circuit stage, which generates the needed circuit, while abiding by the
	structure required by the common run stage
	"""
	def __init__(self, *args):
		super().__init__(*args)
		self.circuit = None
		self.video_sequence = self.outputs_container[RunStages.current_sequence.name][0]
		self.resistive_mesh = self.outputs_container[RunStages.resistive_mesh.name][0]
		self.number_of_pixels = Configuration().params["number_of_pixels"]
		self.simulation_results = None

		self.G_comp_flag = False

		if Configuration().params["model"]== Models.MONO_DR.value:
			with open(self.resistive_mesh, 'rb') as f:
				dat = pkl.load(f)
			
			N_act = dat['N_act']
			N_ret = dat['N_ret']
			Gp = np.zeros((N_act+1, N_act+1))
			Gp[:N_act, :N_act] = dat['G'][:N_act, :N_act]
			Gp[:N_act, N_act] = np.sum(dat['G'][:N_act, N_act:], axis=1)
			Gp[N_act, :N_act] = Gp[:N_act, N_act]
			Gp[N_act, N_act] = np.sum(dat['G'][N_act:, N_act:])
			
			imag_basis = np.array(self.video_sequence["Frames"]).reshape((self.number_of_pixels, -1))
			I_tot = np.sum(imag_basis, axis=0)
			I_tot = np.reshape(I_tot, (1, -1))
			imag_basis = np.concatenate((imag_basis, -I_tot), axis=0)
			zero_idx = np.linalg.norm(imag_basis, axis=0) == 0
			imag_basis = imag_basis[:, ~zero_idx]

			Gs = 1/Configuration().params['shunt_resistance']
			Gp += np.eye(N_act+1)*Gs
			Gp[N_act, N_act] += np.eye(N_act+1)*Gs
			Gp[:N_act, N_act] -= Gs
			Gp[N_act, :N_act] -= Gs

			v_basis = np.linalg.solve(Gp, imag_basis)
			ep_pad = np.outer( np.ones(N_ret-1), v_basis[-1, :] )
			v_basis = np.concatenate((v_basis, ep_pad ), axis=0)
			(v_basis_om, _) = np.linalg.qr(v_basis)
			i_basis = (dat['G'] - dat['G_cmp']) @ v_basis_om

			self.G_comp = {}
			self.G_comp['v_basis'] = np.concatenate((dat['u'] , v_basis_om), axis=1)
			self.G_comp['i_basis'] = np.concatenate((dat['u']*dat['w'] , i_basis), axis=1)

			Smat = np.diagflat(np.sum(dat['S'], axis=0)) - np.tril(dat['S'], -1) - np.triu(dat['S'], 1)
			Smat[Smat==0] = np.nan
			self.resistive_mesh = 1/Smat

			self.number_of_returns = N_ret
			self.ret_tri_area = dat['tri_area']
		
			self.G_comp_flag = True
			


		if Configuration().params["model"]== Models.MONOPOLAR.value and Configuration().params['r_matrix_simp_ratio'] < 1:
			imag_basis = np.array(self.video_sequence["Frames"]).reshape((self.number_of_pixels, -1))
			col_norm = np.linalg.norm(imag_basis, axis=0)
			imag_basis = imag_basis[:, col_norm>1E-6]
			(self.resistive_mesh, self.G_comp) = Rmat_simp(Rmat=self.resistive_mesh, Gs=1/Configuration().params['shunt_resistance'],
						  ratio=Configuration().params['r_matrix_simp_ratio'], imag_basis=imag_basis)
			self.G_comp_flag = True

		if Configuration().params.get("r_matrix_input_file_px_pos"):
			self.is_edge = is_edge(np.loadtxt(Configuration().params["r_matrix_input_file_px_pos"], delimiter=','), Configuration().params["pixel_size"])
			self.edge_factor = Configuration().params["additional_edges"] / np.sum(self.is_edge) / 6

	@property
	def stage_name(self):
		return RunStages.circuit.name

	def run_stage(self, *args, **kwargs):
		"""
		This function holds the execution logic for the circuit stage
		:param args:
		:param kwargs:
		:return:
		"""
		is_bipolar = Configuration().params["model"] == Models.BIPOLAR.value
		is_mono_dr = Configuration().params["model"] == Models.MONO_DR.value
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
		self.circuit.subcircuit(SIROF('active', c0=Configuration().params["sirof_active_capacitance_nF"]))
		if is_mono_dr:
			self.circuit.subcircuit(SIROF('return', c0=Configuration().params["sirof_capacitance"]*1E-2), Rdc=2E3)
		else:
			self.circuit.subcircuit(SIROF('return',	
				scaling= Configuration().params["return_to_active_area_ratio"] * (1 if is_bipolar else self.number_of_pixels),
				c0=Configuration().params["sirof_active_capacitance_nF"]), Vini=Configuration.params['Vini_ret'])
		if is_bipolar:
			edge_ar = Configuration().params["return_to_active_area_ratio"] * (1 + self.edge_factor)
			self.circuit.subcircuit(SIROF('return_edge',	Vini=-Configuration().params["initial_Vactive"] / edge_ar,
				scaling=edge_ar, c0=Configuration().params["sirof_active_capacitance_nF"]))
		elif not is_mono_dr:
			self.circuit.V(f'CProbe{0}', self.circuit.gnd, f'Pt{0}', 0 @U.u_V)
			self.circuit.X('Return', f"return PARAMS: Vini={Configuration().params['Vini_ret']}", f'Pt{0}', f'Saline{0}')

		for px_idx in range(1, self.number_of_pixels + 1):
			#configure the active electrodes and their connections to the GND
			self.circuit.V(f'CProbe{px_idx}', f'Si{px_idx}', f'Pt{px_idx}', 0 @U.u_V)  # current probe
			self.circuit.X(f'Active{px_idx}', f"active PARAMS: Vini={Configuration().params['Vini_act'][px_idx-1]}", f'Pt{px_idx}', f'Saline{px_idx}')
			self.circuit.R(f'{px_idx}_{px_idx}', f'Saline{px_idx}', f'Saline{0}', "{:.3e}".format(self.resistive_mesh[px_idx - 1, px_idx - 1]))
			
			if is_bipolar:
				#configure return electrodes and their connections to the GND
				self.circuit.V(f'rCProbe{px_idx}', self.circuit.gnd, f'rPt{px_idx}', 0@U.u_V)
				self.circuit.X(f'Return{px_idx}', 'return' + ("_edge" if self.is_edge[px_idx-1] else ""), f'rPt{px_idx}', f'rSaline{px_idx}')
				
				self.circuit.R(f'r{px_idx}_{px_idx}', f'rSaline{px_idx}', f'Saline{0}',
		   						"{:.3e}".format(self.resistive_mesh[px_idx+self.number_of_pixels-1, px_idx+self.number_of_pixels-1]))
				# connections between each pair of active and return
				for cross_idx in range(1, self.number_of_pixels+1):
					self.circuit.R(f'ar{cross_idx}_{px_idx}', f'Saline{cross_idx}', f'rSaline{px_idx}',
						"{:.3e}".format(self.resistive_mesh[px_idx-1, cross_idx+self.number_of_pixels-1]))

			# now define the interconnected resistor mesh
			for cross_idx in range(1, px_idx):
				R = self.resistive_mesh[px_idx - 1, cross_idx - 1]
				if np.isnan(R):
					continue
				#interconnection among the active electrodes
				self.circuit.R(f'{cross_idx}_{px_idx}', f'Saline{cross_idx}', f'Saline{px_idx}', "{:.3e}".format(R))
				if is_bipolar:
					#interconnection among the return electrodes
					self.circuit.R(f'r{cross_idx}_{px_idx}', f'rSaline{cross_idx}', f'rSaline{px_idx}',  
		    			"{:.3e}".format(self.resistive_mesh[px_idx+self.number_of_pixels-1, cross_idx+self.number_of_pixels-1]))
			

			if is_mono_dr:
				for ret_idx in range(1, self.number_of_returns+1):
					R = self.resistive_mesh[px_idx - 1, self.number_of_pixels + ret_idx - 1]
					if np.isnan(R):
						continue
					self.circuit.R(f'ar{px_idx}_{ret_idx}', f'Saline{px_idx}', f'rSaline{ret_idx}', "{:.3e}".format(R))
			

		if is_mono_dr:
			for ret_idx in range(1, self.number_of_returns + 1):
				self.circuit.V(f'rCProbe{ret_idx}', self.circuit.gnd, f'rPt{ret_idx}', 0@U.u_V)
				self.circuit.X(f'Return{ret_idx}', f"return PARAMS: Scale={self.ret_tri_area[ret_idx-1]}", f'rPt{ret_idx}', f'rSaline{ret_idx}')
				
				R = self.resistive_mesh[self.number_of_pixels + ret_idx - 1, self.number_of_pixels + ret_idx - 1]
				if not np.isnan(R):
					self.circuit.R(f'r{ret_idx}_{ret_idx}', f'rSaline{ret_idx}', f'Saline{0}', "{:.3e}".format(R))
				# connections between each pair of return
				for cross_idx in range(1, self.number_of_returns+1):
					R = self.resistive_mesh[self.number_of_pixels + cross_idx - 1, self.number_of_pixels + ret_idx - 1]
					self.circuit.R(f'r{cross_idx}_{ret_idx}', f'rSaline{cross_idx}', f'rSaline{ret_idx}', "{:.3e}".format(R))

				
			
		# Conductance matrix compensation after thresholding:
		if self.G_comp_flag:
			for comp_idx in range(self.G_comp['v_basis'].shape[1]):
				self.circuit.R(f'comp{comp_idx}', f'comp{comp_idx}', self.circuit.gnd, 1)
				for px_idx in range(1, self.number_of_pixels + 1):
					self.circuit.VCCS(f'px{px_idx}_comp{comp_idx}', self.circuit.gnd, f'comp{comp_idx}',
		       							f'Saline{px_idx}', f'Saline{0}', "{:.3e}".format(self.G_comp['v_basis'][px_idx-1, comp_idx]))
					self.circuit.VCCS(f'comp{comp_idx}_px{px_idx}', f'Saline{px_idx}', f'Saline{0}',
		       							f'comp{comp_idx}', self.circuit.gnd, "{:.3e}".format(self.G_comp['i_basis'][px_idx-1, comp_idx]))
		
		return self.circuit

