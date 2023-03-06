import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class VisualizationUtils:

	@staticmethod
	def plot_3d_array(array, title=None, x_label=None, y_label=None, z_label=None, colorbar_label=None, mesh=None):
		"""
		Plots a 3D array in 3D using matplotlib.

		Parameters:
		array (np.ndarray): A 3D NumPy array.

		Returns:
		None
		"""
		# create a 3D figure
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		# get the shape of the array
		x_len, y_len, z_len = array.shape

		# create 3D coordinates
		if not mesh:
			xx, yy, zz = np.meshgrid(np.arange(x_len), np.arange(y_len), np.arange(z_len))
		else:
			xx, yy, zz = mesh

		# flatten the array
		array_flat = array.flatten()

		# plot the array using scatter plot
		point_cloud = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=array_flat, cmap="jet")

		# Set axis labels
		ax.set_xlabel(x_label, fontsize=14)
		ax.set_ylabel(y_label, fontsize=14)
		ax.set_zlabel(z_label, fontsize=14)

		# adjust color bar
		cb = fig.colorbar(point_cloud)
		cb.ax.set_ylabel(colorbar_label, fontsize=14)

		# Show the plot
		plt.title(title)
		plt.show()

		return fig