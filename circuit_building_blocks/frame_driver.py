from PySpice.Spice.Netlist import SubCircuit
import PySpice.Unit as U
import numpy as np


# control subframes
class FramesDriver(SubCircuit):

	"""
	This class builds the frame driver in the circuit that switches difference frames within an image
	"""

	def __init__(self, name, frame_val=[[.1, 0]], frame_T=[[10, 22]], time_step=0.05,
				 Apho_factor=1):  # current in uA, time in ms
		self.__nodes__ = ('GND', 'out') + tuple(f'in{x + 1}' for x in range(len(frame_val)))
		SubCircuit.__init__(self, name, *self.__nodes__)

		#low pass filter to avoid discontinuity
		self.R('load', 'Vout', 'GND', 1 @ U.u_Ohm)
		self.C('load', 'Vout', 'GND', (time_step * 2) @ U.u_mF)

		for frame_idx in range(len(frame_val)):
			#turn on the frame at the right image
			self.VoltageControlledSwitch(f'ImgON{frame_idx + 1}', f'Vframe{frame_idx + 1}', 'Vout', \
										 f'in{frame_idx + 1}', 'GND', model='SW1V')

			#add rising/falling edges of a finite width at the transitions to avoid discontinuity
			subframe_val = frame_val[frame_idx]
			mid_val = (subframe_val[0] + subframe_val[-1]) / 2
			subframe_cumT = np.cumsum(frame_T[frame_idx])
			frame_seq = [(0, mid_val @ U.u_V), (time_step @ U.u_ms, subframe_val[0] @ U.u_V)]

			#config a pulse generator that outputs a voltage waveform resembling the irradiance as a function of time
			for subframe_idx in range(len(subframe_cumT) - 1):
				time_pt = subframe_cumT[subframe_idx]
				frame_seq.append(((time_pt - time_step) @ U.u_ms, subframe_val[subframe_idx] @ U.u_V))
				frame_seq.append(((time_pt + time_step) @ U.u_ms, subframe_val[subframe_idx + 1] @ U.u_V))
			frame_seq.append(((subframe_cumT[-1] - time_step) @ U.u_ms, subframe_val[-1] @ U.u_V))
			frame_seq.append((subframe_cumT[-1] @ U.u_ms, mid_val @ U.u_V))

			self.PieceWiseLinearVoltageSource(f'DriverV{frame_idx + 1}', f'Vframe{frame_idx + 1}', 'GND',
											  values=frame_seq, repeat_time=0)

		self.VCCS('DriveC', 'GND', 'out', 'Vout', 'GND', Apho_factor * 2 @ U.u_uS)

		return
