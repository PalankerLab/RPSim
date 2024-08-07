import os
import csv
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
from configuration.models import Models
from configuration.configuration_manager import Configuration

from configuration.stages import RunStages
from run_stages.common_run_stage import CommonRunStage

from utilities.image_processing_utilities import *


class CurrentSequenceStage(CommonRunStage):
	"""
	This class implements the logic for the current sequence stage while abiding by the structure required by the common
	run stage
	"""

	def __init__(self, *args):
		super().__init__(*args)

		# initialize video sequence structure
		self.video_sequence = {
			"name": self.output_directory_name,
			'pixel_size': Configuration().params["pixel_size"]
		}

		# Whether to look for generated patterns or pre-existing patterns
		self.is_generated = Configuration().params["generate_pattern"]

		self.multiplexed = 'multiplex' in Configuration().params and Configuration().params['multiplex']
		
		# define paths
		if self.is_generated:
			# If the patterns are generated, the source folder is in the output path
			self.image_sequence_input_folder = self.output_directory
		else:
			# When loading existing patterns, the source folder is in the input path
			self.image_sequence_input_folder = os.path.join(Configuration().params["user_input_path"], "image_sequence",self.output_directory_name)
		
		self.sequence_script_input_file = os.path.join(self.image_sequence_input_folder, "seq_time.csv")

		# initialize gif params
		self.gif_image = []
		self.gif_time = []

		# load the geometric description of the array
		with open(Configuration().params["pixel_label_input_file"], 'rb') as f:
			self.image_label = pickle.load(f)

		# extract number of pixels
		self.number_of_pixels = self.image_label.max()

		self.script = list()
		self.max_photo_current_in_ua = None

		# Variable used for plotting the most illuminated pixel at the end of the run
		self.most_illuminated_pixels = dict()

	@property
	def stage_name(self):
		return RunStages.current_sequence.name

	def run_stage(self):
		"""
		This function holds the execution logic for the current sequence stage
		:param args:
		:param kwargs:
		:return:
		"""
		# TODO: multiplex added here
		if self.multiplexed:
			list_images = self.outputs_container[RunStages.multiplexing.name][0]
		elif self.is_generated:
			#list_images = self.outputs_container["pattern_generation"][0]
			list_images = self.outputs_container[RunStages.pattern_generation.name][0]
		self._parse_script_file()

		# duration of the frames in ms
		for row in self.script:
			self.video_sequence['frame_names'].append(row[0])
			self.video_sequence['nb_repetitions_frames'].append(int(row[1]))
			row_dat = [float(x) for x in row[2:] if x]

			assert abs(sum(row_dat) - self.video_sequence['duration_frames_ms'])<1e-6, "Frames must be of the same length!" # TODO: change rounding error
			self.video_sequence['duration_subframes_ms'].append(row_dat)
			print(len(row_dat))
			print(row_dat)

		print(np.array(self.video_sequence['duration_subframes_ms']).shape)

		self.video_sequence['Frames'] = [deepcopy(self.video_sequence['duration_subframes_ms']) for _ in range(
			self.number_of_pixels)]
		
		if Models.BIPOLAR.value:
			self.video_sequence['Frames'] = np.array(self.video_sequence['Frames'][np.newaxis, :, :, :], 2, axis=0)
		print(self.video_sequence['Frames'].shape)


		
		print(np.array(self.video_sequence['Frames']).shape)
		print(len(self.video_sequence['duration_subframes_ms']))



		# Iterate on the images
		for frame_idx in range(len(self.script)):
			number_of_sub_frames = len(self.video_sequence['duration_subframes_ms'][frame_idx])
			image_stack_temp = []

			# TODO: multiplexed here
			if self.is_generated or self.multiplexed:
				list_subframes = list_images[frame_idx]
			
			# Iterate on the subframes
			for sub_frame_idx in range(number_of_sub_frames):

				if self.is_generated or self.multiplexed:
					image = list_subframes[sub_frame_idx]
				else:
					sub_frame_image_path = os.path.join(Configuration().params["user_input_path"], 'image_sequence',
														Configuration().params["video_sequence_name"],
														f"{self.video_sequence['frame_names'][frame_idx]}",
														f'Subframe{sub_frame_idx + 1}.bmp')
					image = plt.imread(sub_frame_image_path).astype(float)
				
				image = red_corners(image, self.image_label.shape[0])
				#image = red_corners(image, 2000)

				# fill in the photo-current of each pixel for each sub frame
				light_on_pixels = img2pixel(image, self.image_label)
				self.add_most_illuminated_pixel(light_on_pixels)
				for pixel_idx in range(self.number_of_pixels):
					self.video_sequence['Frames'][pixel_idx][frame_idx][sub_frame_idx] = light_on_pixels[
						pixel_idx] * self.max_photo_current_in_ua

				image_stack_temp.append(im.fromarray(np.uint8(image.round())))

			number_of_repetitions = self.video_sequence['nb_repetitions_frames'][frame_idx]
			self.gif_image += image_stack_temp * number_of_repetitions
			self.gif_time += self.video_sequence['duration_subframes_ms'][frame_idx] * number_of_repetitions

		self.gif_time = [x * 10 for x in self.gif_time]

		return [self.video_sequence, {"gif_data": self.gif_image, "gif_time": self.gif_time}, self.determine_most_illuminated(), self.image_sequence_input_folder]
	
	def _parse_script_file(self):
		"""
		This function reads the sequence definition from the csv spec file, including time information and irradiance.
		"""
		# TODO: multiplex changes here
		if self.multiplexed:
			self.script = self.outputs_container[RunStages.multiplexing.name][1]
		elif self.is_generated:
			self.script = self.outputs_container[RunStages.pattern_generation.name][1]
		else:
			# open csv file with video sequence description
			with open(self.sequence_script_input_file, 'r') as f:
				csv_file = csv.reader(f)
				for row in csv_file:
					self.script.append(row)

		# photocurrent per pixel at 1mW/mm^2
		photocurrent = Configuration().params["photosensitive_area"] * Configuration().params["light_to_current_conversion_rate"] \
			* 1E-3 / Configuration().params["number_of_diodes"]

		self.max_photo_current_in_ua = float(self.script.pop(0)[1]) * photocurrent  # uA  maximum photo-current
		self.video_sequence['duration_frames_ms'] = float(self.script.pop(0)[1])
		self.video_sequence['time_step'] = float(self.script.pop(0)[1])
		self.video_sequence['frame_names'] = []
		self.video_sequence['nb_repetitions_frames'] = []
		self.video_sequence['duration_subframes_ms'] = []
		# Removes the row containing the column names (i.e. Frame 'Repetition', 'Subframe1', ..., 'SubframeN')
		self.script = self.script[1:]

	def add_most_illuminated_pixel(self, light_on_pixels):
		"""
		This function fills the dictionary of the most illuminated pixel per subframe. 
		It is used at the end of the run for plotting purposes.  

		Params:
			light_on_pixels (np.array): contains the light on each pixel for one subframe
		"""

		idx, val = np.argmax(light_on_pixels) + 1, np.max(light_on_pixels)
		if idx in self.most_illuminated_pixels:
			self.most_illuminated_pixels[idx] += val
		else:
			self.most_illuminated_pixels[idx] = val
		
	def determine_most_illuminated(self):
		"""
		This function determines which pixels received the most amount of light across all subframes and frames. 
		It is used at the end of the run for plotting purposes.  

		Return: 
			index of the most illuminated pixel corresponding to the pixel label files
		"""
		
		# Return the dict key of the entry having the maximum value (i.e. the most illuminated pixel)
		idx_most = max(self.most_illuminated_pixels, key=self.most_illuminated_pixels.get)
		return idx_most


