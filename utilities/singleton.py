
class Singleton(type):
	"""
	This class implements the Singleton design pattern used across the tool modules
	"""
	_instances = {}

	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
		return cls._instances[cls]
