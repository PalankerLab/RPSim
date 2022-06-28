from PySpice.Spice.Netlist import SubCircuit
import PySpice.Unit as U
import numpy as np


# multiplex for switching between images
class ImageDriver(SubCircuit):
	"""
	This class implements the image driver that loop through different images in the sequence.
	"""
	def __init__(self, name, image_T=[64, 96, 128], delay=2):
		self.__nodes__ = tuple(['GND']) + tuple(f'out{x + 1}' for x in range(len(image_T)))
		SubCircuit.__init__(self, name, *self.__nodes__)

		#add a global delay
		self.raw_spice = f'Vdelay AllON GND PAT(1 0 0 50us 50us {delay}ms b01)'
		self.R('load', 'AllON', 'GND', 1 @ U.u_Ohm)

		#looping a sequence of potential levels to control the image multiplex 
		image_cumT = np.cumsum([0] + image_T)
		for img_idx in range(len(image_T)):
			if len(image_T) == 1:
				self.V('img1', 't1', 'GND', 2 @ U.u_V)
			else:
				self.PulseVoltageSource(f'img{img_idx + 1}', f't{img_idx + 1}', 'GND',
										initial_value=0 @ U.u_V, pulsed_value=2 @ U.u_V,
										pulse_width=image_T[img_idx] @ U.u_ms, \
										period=image_cumT[-1] @ U.u_ms, delay_time=image_cumT[img_idx] @ U.u_ms)
			self.VoltageControlledSwitch(f'delay{img_idx + 1}', f't{img_idx + 1}', f'out{img_idx + 1}', 'AllON', 'GND', model='SW1V')
			self.R(img_idx, f'out{img_idx + 1}', 'GND', 1 @ U.u_Ohm)

		return
