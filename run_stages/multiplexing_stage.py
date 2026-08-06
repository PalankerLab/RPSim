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
        Multiplexing Stage
    """

    def __init__(self, *args):
        super().__init__(*args)

        self.list_subframes_as_ndarray_after_multiplexing = []
        self.script = []
        self.dict_PIL_images_after_multiplexing = {}

        self.is_generated = Configuration().params["generate_pattern"]

        # define paths
        if self.is_generated:
            # If the patterns are generated, the source folder is in the output path
            self.image_sequence_input_folder = self.output_directory
        else:
            # When loading existing patterns, the source folder is in the input path
            self.image_sequence_input_folder = os.path.join(Configuration().params["user_input_path"], "image_sequence",
                                                            self.output_directory_name)

        if self.is_generated:
            self.script = self.outputs_container[RunStages.pattern_generation.name][1]
        else:
            self.sequence_script_input_file = os.path.join(self.image_sequence_input_folder, "seq_time.csv")
            print(f"Pattern generation skipped.. Using existing files at {self.sequence_script_input_file}")
            # open csv file with video sequence description
            with open(self.sequence_script_input_file, 'r') as f:
                csv_file = csv.reader(f)
                for row in csv_file:
                    self.script.append(row)

    @property
    def stage_name(self):
        return RunStages.multiplexing.name

    def _show_multiplexed(self, original_img, img_list, frame_name, subframe_idx):
        '''
            Plot multiplexed result of original_img
            img_list: multiplexed images
        '''
        fig, ax = plt.subplots(nrows=1, ncols=len(img_list) + 1, figsize=(15, 8))
        img_list = [original_img] + img_list
        axes = ax.ravel()
        for i, img in enumerate(img_list):
            axes[i].imshow(img.astype(np.uint8), vmin=0, vmax=255)
            if i == 0:
                title = "Original"
            else:
                title = f"Multiplexed {i}"
            axes[i].set_title(title)
        fig.suptitle(f"{frame_name} Subframe {subframe_idx + 1} Multiplexed Results", y=0.7)
        plt.tight_layout()
        plt.show(block=False)

    def _check_is_black(self, img):
        '''
            Check whether a frame is black
        '''
        green_channel = img[:, :, 1].sum().item()
        blue_channel = img[:, :, 2].sum().item()
        is_all_black = green_channel == 0 and blue_channel == 0
        is_all_white = green_channel == (255 * img.shape[0] * img.shape[1]) and blue_channel == (
                    255 * img.shape[0] * img.shape[1])
        return is_all_black or is_all_white

    def _apply_mask(self, image, mask, as_PIL, w, h):
        '''Helper: apply a binary mask to image and return PIL or ndarray with red border.'''
        img_PIL = Image.fromarray((mask * deepcopy(image)).astype(np.uint8))
        ImageDraw.Draw(img_PIL).rectangle([0, 0, w - 1, h - 1], outline="red", width=2)
        return img_PIL if as_PIL else np.asarray(img_PIL)

    def _multiplex_vertical(self, image, as_PIL, h, w, num_split):
        '''Split image into num_split vertical strips.'''
        result = []
        strip_width = w // num_split
        for i in range(0, w - strip_width + 1, strip_width):
            mask = np.zeros_like(image, dtype=float)
            mask[:, i:i + strip_width, :] = 1
            result.append(self._apply_mask(image, mask, as_PIL, w, h))
        return result

    def _multiplex_horizontal(self, image, as_PIL, h, w, num_split):
        '''Split image into num_split horizontal strips.'''
        result = []
        strip_height = h // num_split
        for i in range(0, h - strip_height + 1, strip_height):
            mask = np.zeros_like(image, dtype=float)
            mask[i:i + strip_height, :, :] = 1
            result.append(self._apply_mask(image, mask, as_PIL, w, h))
        return result

    def _multiplex_checkerboard(self, image, as_PIL, h, w):
        '''Split image into 2 complementary checkerboard frames. Square size is set by
        checkerboard_square_size_um (default 200 um) in the configuration.'''
        square_size_um = Configuration().params.get('checkerboard_square_size_um', 200)
        square_px = max(1, round(square_size_um * w / Configuration().params['frame_width']))
        col_idx = np.arange(w) // square_px
        row_idx = np.arange(h) // square_px
        checkerboard = (row_idx[:, None] + col_idx[None, :]) % 2  # (h, w), values 0 or 1
        result = []
        for parity in (0, 1):
            mask = np.zeros_like(image, dtype=float)
            mask[checkerboard == parity] = 1
            result.append(self._apply_mask(image, mask, as_PIL, w, h))
        return result

    def _multiplex(self, image, as_PIL=False):
        '''
            The function that does the actual multiplexing
            image: the source image
            as_PIL: if False, return np.array, if True, return PIL.Image
            return: a list of multiplexed images
        '''
        num_split = Configuration().params['num_split'] if 'num_split' in Configuration().params else 4
        h, w, c = image.shape
        alg = Configuration().params['alg'] if 'alg' in Configuration().params else 'horizontal'
        if alg == 'vertical':
            return self._multiplex_vertical(image, as_PIL, h, w, num_split)
        elif alg == 'horizontal':
            return self._multiplex_horizontal(image, as_PIL, h, w, num_split)
        elif alg == 'checkerboard':
            return self._multiplex_checkerboard(image, as_PIL, h, w)
        else:
            raise NotImplementedError("Please provide a valid multiplex algorithm")

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
                subframe_total_duration = float(old_row[2 + idx])
                num_multiplexed = info[frame_name][idx]  # split into these sub-subframes
                total_frames += num_multiplexed
                if num_multiplexed > 1:  # a result of multiplexing, non-black image
                    for j in range(num_multiplexed):
                        new_row.append(subframe_total_duration / num_multiplexed)
                else:  # black frame, copy value
                    new_row.append(subframe_total_duration)
            modified_script.append(new_row)
            max_frames = max(max_frames, total_frames)

        modified_script[3] = [None, 'Frame Repetition'] + [f'Subframe{i}' for i in range(1, max_frames + 1)]
        return modified_script

    def run_stage(self, *args, **kwargs):

        if self.is_generated:
            list_images = self.outputs_container[RunStages.pattern_generation.name][0]
        # Iterate on the images to find max number of subframes and collect information in form below:
        # {'frame1': {
        # 'subframe 1': 4, (a non-black frame splitted into 4)
        # 'subframe 2': 1 (a black frame)
        # }}
        frame_info = {}
        for frame_idx in range(len(self.script) - 4):
            number_of_sub_frames = len(self.script[4 + frame_idx]) - 2
            if self.is_generated:
                list_subframes = list_images[frame_idx]

            frame_name = self.script[4 + frame_idx][0]
            frame_info[frame_name] = {}

            list_tmp_bmp = []
            list_tmp_array = []

            img_idx = 0
            for sub_frame_idx in range(number_of_sub_frames):
                if self.is_generated:
                    image = list_subframes[sub_frame_idx]
                else:
                    sub_frame_image_path = os.path.join(Configuration().params["user_input_path"], 'image_sequence',
                                                        Configuration().params["video_sequence_name"],
                                                        f"{frame_name}",
                                                        f'Subframe{sub_frame_idx + 1}.bmp')
                    assert os.path.exists(
                        sub_frame_idx), f"Sub frame image input directory not found for frame {frame_name} suframe {sub_frame_idx + 1}"
                    image = plt.imread(sub_frame_image_path).astype(float)

                # if black frame, no multiplex
                if self._check_is_black(image):
                    frame_info[frame_name][sub_frame_idx] = 1
                    # Save subframe
                    list_tmp_bmp.append((f'Subframe{img_idx + 1}_multiplexed', Image.fromarray(image.astype(np.uint8))))
                    list_tmp_array.append(image)
                    img_idx += 1
                    continue

                multiplexed_imgs_arr = self._multiplex(image, as_PIL=False)  # np format
                multiplexed_imgs_PIL = [Image.fromarray(x) for x in multiplexed_imgs_arr]  # PIL Image format

                for i, img_PIL in enumerate(multiplexed_imgs_PIL):
                    # plt.imshow(multiplexed_imgs_arr[i])
                    list_tmp_bmp.append((f'Subframe{img_idx + 1}_multiplexed', img_PIL))
                    img_idx += 1

                frame_info[frame_name][sub_frame_idx] = i + 1

                list_tmp_array.extend(multiplexed_imgs_arr)

                self._show_multiplexed(image, multiplexed_imgs_arr, frame_name, sub_frame_idx)

            # Save frame TODO: list_tmp_bmp: Subframe 1: [image1, image2, ....]
            self.dict_PIL_images_after_multiplexing[frame_name] = list_tmp_bmp
            self.list_subframes_as_ndarray_after_multiplexing.append(list_tmp_array)

        modified_script = self._modify_script(frame_info)

        return [self.list_subframes_as_ndarray_after_multiplexing, modified_script,
                self.dict_PIL_images_after_multiplexing]


