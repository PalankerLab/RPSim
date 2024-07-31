import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from copy import deepcopy
import glob
from PIL import Image, ImageDraw, ImageFont
import csv

from configuration.stages import RunStages
from configuration.configuration_manager import Configuration

from run_stages.common_run_stage import CommonRunStage

class MultiplexingStage(CommonRunStage):
    """
    This class implements the automated pattern generation from a configuration file.

    Attributes:
        pixel_size (int): The pixel size in micron (we refer to pixel as the hexagonal unit with photodiode, active and return electrode)
        implant_layout (PIL.Image): The image of the pixel layout
        pixel_labels (Numpy.array): Array having the same size as implant_layout. Each entry corresponds to either 0 or the pixel label (the active and return electrodes are also labeled 0, only the photodiode is non-zero)
        
        center_x, center_y (int, int): The pixels coordinate in implant_layout corresponding the center of the central hexagon
        width, height (int, int): The size in pixels (and microns) of the implant_layout image
        
        backgroud_overlay (PIL.Image): The pixel pattern as background, and the projected image as overlay
        projected (PIL.Image): A black screen as background, and the projected image as overlay
    """

    def __init__(self, *args):
        super().__init__(*args)

        self.list_subframes_as_ndarray_after_multiplexing = []
        self.script = None
        self.dict_PIL_images_after_multiplexing = {}

        self.is_generated = Configuration().params["generate_pattern"]
        
        # define paths
        if self.is_generated:
            # If the patterns are generated, the source folder is in the output path
            self.image_sequence_input_folder = self.output_directory
        else:
            # When loading existing patterns, the source folder is in the input path
            self.image_sequence_input_folder = os.path.join(Configuration().params["user_input_path"], "image_sequence", self.output_directory_name)
            assert os.path.exists(self.image_sequence_input_folder), "Please provide image sequence input folder for multiplexing!"
        
        if self.is_generated:
            self.script = self.outputs_container[RunStages.pattern_generation.name][1]
        else:
            # open csv file with video sequence description
            with open(self.sequence_script_input_file, 'r') as f:
                csv_file = csv.reader(f)
                for row in csv_file:
                    self.script.append(row)
        
                            
    @property
    def stage_name(self):
        return RunStages.multiplexing.name
    
    def _show_multiplexed(self, original_img, img_list, frame_name, subframe_idx):
        fig, ax = plt.subplots(nrows=1, ncols=len(img_list)+1, figsize=(12, 20))
        img_list = [original_img] + img_list
        axes = ax.ravel()
        for i, img in enumerate(img_list):
            axes[i].imshow(img)
            if i == 0:
                title = "Original"
            else:
                title = f"Multiplexed {i}"
            axes[i].set_title(title)
        fig.suptitle(f"{frame_name} Subframe {subframe_idx} Multiplexed Results")
        plt.tight_layout()
        plt.show()
    

    def _check_is_black(self, img):
        # print(img[:, :, 1].sum())
        # print(img[:, :, 2].sum())
        # print(img.shape)
        # print(img[:, :, :3].sum())
        # find all png files
        green_channel = img[:, :, 1].sum().item()
        blue_channel = img[:, :, 2].sum().item()
        # print((255 * img.shape[0] * img.shape[1]))
        # print(green_channel, blue_channel)
        is_all_black = green_channel == 0 and blue_channel == 0
        is_all_white = green_channel == (255 * img.shape[0] * img.shape[1]) and blue_channel == (255 * img.shape[0] * img.shape[1])
        # print(is_all_black, is_all_white)
        return is_all_black or is_all_white
               
    def _multiplex(self, image, as_PIL=False):
        num_split = Configuration().params['num_split']
        h, w, c = image.shape
        alg = Configuration().params['alg']
        result = []
        if alg == 'vertical':
            width = w // num_split
            for i in range(0, w-width+1, width):
                start = i
                end = i + width
                mask = np.zeros_like(image, dtype=bool)
                mask[:, start:end, :] = True
                arr = Image.fromarray((mask * deepcopy(image)).astype(np.uint8))
                # need to draw red corner
                drawing_projection = ImageDraw.Draw(arr)
                drawing_projection.rectangle([start, 0, end - 1, h - 1], outline="red", width=2)
                # plt.imshow(drawing_projection)
                # plt.show()
                if as_PIL:
                    result.append(drawing_projection)
                else:
                    result.append(np.asarray(drawing_projection))
        elif alg == 'horizontal':
            height = h // num_split
            for i in range(0, h-height+1, height):
                start = i
                end = i + height
                mask = np.zeros_like(image, dtype=float)
                mask[start:end, :, :] = 1
                img_PIL = Image.fromarray((mask * deepcopy(image)).astype(np.uint8))
                # need to draw red corner
                drawing_projection = ImageDraw.Draw(img_PIL)
                drawing_projection.rectangle([0, start, w - 1, end - 1], outline="red", width=2)
                # plt.imshow(np.array(img_PIL))
                # plt.show()
                if as_PIL:
                    result.append(img_PIL)
                else:
                    result.append(np.asarray(img_PIL))
        else:
            raise NotImplementedError
        
        return result
    
    def _modify_script(self, info):
        '''
            Modifies the duration of sub-subframes after multiplexing
            Black frames remains unchanged
        '''
        modified_script = deepcopy(self.script[:4])
        max_frames = 0
        for i in range(4, len(self.script)):
            # start to modify script
            old_row = self.script[i]
            frame_name = old_row[0]
            repetition = old_row[1]
            total_frames = 0
            new_row = [frame_name, repetition]
            for idx in info[frame_name]:
                subframe_total_duration = old_row[2+idx]
                # print(info[frame_name][idx])
                num_multiplexed = info[frame_name][idx] # split into these sub-subframes
                total_frames += num_multiplexed
                if num_multiplexed > 1: # a result of multiplexing, non-black image
                    for j in range(num_multiplexed):
                        new_row.append(subframe_total_duration / num_multiplexed)
                else: # black frame, copy value
                    new_row.append(subframe_total_duration)
            modified_script.append(new_row)
            max_frames = max(max_frames, total_frames)

        modified_script[3] = [None,'Frame Repetition'] + [f'Subframe{i}' for i in range(1, max_frames+1)]
        # print(modified_script[3])
        return modified_script

    def run_stage(self, *args, **kwargs):

        if self.is_generated:
            #list_images = self.outputs_container["pattern_generation"][0]
            list_images = self.outputs_container[RunStages.pattern_generation.name][0]
        # Iterate on the images
        # {'frame1': {
            # 'subframe 1': 4, (a non-black frame splitted into 4)
            # 'subframe 2': 1 (a black frame)
        # }}
        frame_info = {} 
        for frame_idx in range(len(self.script) - 4):
            number_of_sub_frames = len(self.script[4 + frame_idx]) - 2 # first_grating_120um_64Hz_3mW_on_for_3.90625ms,32,3.90625,3.90625,3.90625,3.90625
            if self.is_generated:
                list_subframes = list_images[frame_idx]
                # print(len(list_subframes))
            
            # Iterate on the subframes
            frame_name = self.script[4 + frame_idx][0]
            frame_info[frame_name] = {}

            list_tmp_bmp = []
            list_tmp_array = []

            for sub_frame_idx in range(number_of_sub_frames):
                if self.is_generated:
                    image = list_subframes[sub_frame_idx]
                    print(type(image))

                else:
                    sub_frame_image_path = os.path.join(Configuration().params["user_input_path"], 'image_sequence',
                                                        Configuration().params["video_sequence_name"],
                                                        f"{frame_name}",
                                                        f'Subframe{sub_frame_idx + 1}.bmp')
                    image = plt.imread(sub_frame_image_path).astype(float)
                
                if self._check_is_black(image):
                    frame_info[frame_name][sub_frame_idx] = 1
                    # Save subframe
                    list_tmp_bmp.append((f'Subframe{int(sub_frame_idx+1)}', Image.fromarray(image.astype(np.uint8))))
                    list_tmp_array.append(image)
                    continue
                # print(Configuration().params)
                multiplexed_imgs_arr = self._multiplex(image, as_PIL=False) # np format
                multiplexed_imgs_PIL = [Image.fromarray(x) for x in multiplexed_imgs_arr] # PIL Image format

                for i, img_PIL in enumerate(multiplexed_imgs_PIL):
                    list_tmp_bmp.append((f'Subframe{int(sub_frame_idx+1)}_multiplexed_{i+1}', img_PIL))

                frame_info[frame_name][sub_frame_idx] = i + 1

                list_tmp_array.extend(multiplexed_imgs_arr)

                self._show_multiplexed(image, multiplexed_imgs_arr, frame_name, sub_frame_idx)
        
            # Save frame TODO: list_tmp_bmp: Subframe 1: [image1, image2, ....]
            self.dict_PIL_images_after_multiplexing[frame_name] = list_tmp_bmp
            self.list_subframes_as_ndarray_after_multiplexing.append(list_tmp_array)

        modified_script = self._modify_script(frame_info)
        # print(modified_script)

        return [self.list_subframes_as_ndarray_after_multiplexing, modified_script, self.dict_PIL_images_after_multiplexing] 
            
            
