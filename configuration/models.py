from enum import Enum


class Models(Enum):
	"""
	This class defines the currently supported diode configurations in the array
	"""

	MONOPOLAR = "monopolar"
	BIPOLAR = "bipolar"
	MONO_DR = "monopolar_DR"