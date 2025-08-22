import os
import cv2
import csv
import tifffile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from configuration.stages import RunStages
from configuration.configuration_manager import Configuration

from run_stages.common_run_stage import CommonRunStage

UINT8_MAX = np.iinfo(np.uint8).max    # 255
UINT16_MAX = np.iinfo(np.uint16).max  # 65535


class MaskProjectionStage(CommonRunStage):

    def __init__(self, *args):
        super().__init__(*args)

        # initialize output values
        self.list_subframes_as_ndarray_after_masking = []
        self.dict_PIL_images_after_masking = {}

        # set flags to determine how the patterns are generated
        self.is_generated = True if RunStages.pattern_generation.name in Configuration().params["run_stages"] else False
        self.is_multiplexed = True if RunStages.multiplexing.name in Configuration().params["run_stages"] else False

        # get the input images
        self.input_projection_patterns = []
        if self.is_multiplexed:
            self.input_projection_patterns = self.outputs_container[RunStages.multiplexing.name][0]
        elif self.is_generated:
            self.input_projection_patterns = self.outputs_container[RunStages.pattern_generation.name][0]

        # define projection script path
        if self.is_generated:
            # If the patterns are generated, the source folder is in the output path
            self.image_sequence_input_folder = self.output_directory
        else:
            # When loading existing patterns, the source folder is in the input path
            self.image_sequence_input_folder = os.path.join(Configuration().params["user_input_path"], "image_sequence", self.output_directory_name)

        # get the input projection script
        if self.is_generated:
            self.input_projection_script = self.outputs_container[RunStages.pattern_generation.name][1]
        else:
            self.sequence_script_input_file = os.path.join(self.image_sequence_input_folder, "seq_time.csv")
            with open(self.sequence_script_input_file, 'r') as f:
                csv_file = csv.reader(f)
                for row in csv_file:
                    self.input_projection_script.append(row)

        # define path to mask image
        self.mask_path = os.path.join(Configuration().params["user_input_path"], 'projection_mask', 'AVG_FF_Mask.tif')

    @property
    def stage_name(self):
        return RunStages.mask_projection.name

    def run_stage(self, *args, **kwargs):
        # iterate the frames and for each frame extract its sub-frames
        # frame_info = {}
        for frame_idx in range(len(self.input_projection_script) - 4):

            # set frame name
            frame_name = self.input_projection_script[4 + frame_idx][0]

            # get number of sub-frames for the current frame
            number_of_sub_frames = len(self.input_projection_script[4 + frame_idx]) - 2

            # initialize frame info
            # frame_info[frame_name] = {}

            list_tmp_bmp = []
            list_tmp_array = []

            img_idx = 0
            for sub_frame_idx in range(number_of_sub_frames):

                # get the sub-frame name from the runtime container
                if len(self.input_projection_patterns) != 0 and self.input_projection_patterns[frame_idx] is not None:
                    subframe_image = self.input_projection_patterns[frame_idx][sub_frame_idx]

                # load the sub-frame image from the input path
                else:
                    sub_frame_image_path = os.path.join(Configuration().params["user_input_path"], 'image_sequence',
                                                        Configuration().params["video_sequence_name"],
                                                        f"{frame_name}",
                                                        f'Subframe{sub_frame_idx + 1}.bmp')

                    # check if the sub-frame image exists
                    assert os.path.exists(sub_frame_idx), f"Sub frame image input directory not found for frame {frame_name} suframe {sub_frame_idx + 1}"

                    # read the sub-frame image from path
                    subframe_image = plt.imread(sub_frame_image_path).astype(float)

                # apply a mask to the sub-frame image
                masked_subframe_image, masked_subframe_array = self._apply_mask(subframe_image)

                # store as array and PIL image
                list_tmp_array.append(masked_subframe_array)
                masked_subframe_image_PIL = [masked_subframe_image]
                list_tmp_bmp.append((f'Subframe{img_idx + 1}_masked', masked_subframe_image_PIL))

            # save modified frames
            self.dict_PIL_images_after_masking[frame_name] = list_tmp_bmp
            self.list_subframes_as_ndarray_after_masking.append(list_tmp_array)

        return [self.list_subframes_as_ndarray_after_masking, self.dict_PIL_images_after_masking]

    def _load_mask_image(self):
        # check if mask image path exists
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f"Mask image not found at {self.mask_path}. Please provide a valid mask image.")

        # load the TIFF mask image from the specified path
        tiff_mask = tifffile.imread(self.mask_path)

        # return the mask image
        return tiff_mask

    def _apply_mask(self, subframe_image):
        # load the TIFF mask (can be 16-bit or float)
        mask_image = self._load_mask_image()

        # plot original mask
        plt.figure(figsize=(6, 4))
        plt.title("Original Mask")
        plt.imshow(mask_image, cmap='gray')
        plt.colorbar()
        plt.show()

        # resize the mask to match the target image size
        img_array = np.array(subframe_image)
        target_size = (img_array.shape[1], img_array.shape[0])
        resized_mask = cv2.resize(mask_image, target_size, interpolation=cv2.INTER_NEAREST)

        # plot resized mask
        plt.figure(figsize=(6, 4))
        plt.title("Resized Mask")
        plt.imshow(resized_mask, cmap='gray')
        plt.colorbar()
        plt.show()

        # upcast image to match mask's dtype & scale
        if np.issubdtype(resized_mask.dtype, np.floating):
            img_upcast = img_array.astype(np.float32)
        elif np.issubdtype(resized_mask.dtype, np.integer):
            img_upcast = img_array.astype(np.float32) / UINT8_MAX * UINT16_MAX
        else:
            raise TypeError("Unsupported mask dtype")

        # expand mask to 3 channels if needed
        if resized_mask.ndim == 2:
            resized_mask = np.expand_dims(resized_mask, axis=2)

        # multiply the mask by the original frame in high precision
        masked = img_upcast * resized_mask

        # rescale the result back to 8-bit (0–255)
        masked_subframe_array = masked / masked.max() * UINT8_MAX
        masked_subframe_array = np.clip(masked_subframe_array, 0, UINT8_MAX).astype(np.uint8)

        # store the masked subframe image
        masked_subframe_image = Image.fromarray(masked_subframe_array)

        # plot the final masked image
        plt.figure(figsize=(6, 4))
        plt.title("Final Masked Image")
        plt.imshow(masked_subframe_array)
        plt.axis("off")
        plt.show()

        return masked_subframe_image, masked_subframe_array


