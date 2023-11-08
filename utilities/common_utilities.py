import os
import pickle
import logging
from pathlib import Path
from datetime import datetime
from distutils.dir_util import copy_tree
from configuration.configuration_manager import Configuration

import matplotlib.pyplot


class CommonUtils:
    """
    This class holds all common utilities shared between different parts of the flow
    """

    @staticmethod
    def setup_logging(logger_name):
        """
        This function sets up a logger entity with the provided name
        :param logger_name: the name of the generated logger entity
        :return:
        """
        # create a custom logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # create handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # create formatter and add it to handler
        logging_format = '[%(asctime)-5s] %(levelname)-10s %(funcName)-16s %(message)s'
        console_format = logging.Formatter(logging_format, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_format)

        # Add handler to the logger
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def add_logger_file_handle(logger, file_name):
        """
        This function adds a file handle to an existing logger entity
        """
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(logging.INFO)

        # create formatter and add it to handler
        logging_format = '[%(asctime)-5s] %(levelname)-10s %(funcName)-16s %(message)s'
        file_format = logging.Formatter(logging_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_format)

        # Add handler to the logger
        logger.addHandler(file_handler)

    @staticmethod
    def generate_output_directory(parent_directory):
        """
        This function creates a new output directory in the given parent directory and with the given prefix name
        :param parent_directory: the path of the parent directory in which the new folder should be created
        :return:
        """
        # generate unique directory name using date and time
        current_date_and_time = datetime.now()
        output_directory = os.path.join(parent_directory, current_date_and_time.strftime("%H_%M_%S-%Y_%m_%d").split(
            '.')[0])

        # create directory and parents, if they do not exist
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        # verify it was created successfully
        if not (os.path.isdir(output_directory) and os.path.exists(output_directory)):
            raise NotADirectoryError

        return output_directory

    @staticmethod
    def store_output(output, output_directory, file_name):
        """
        This function stores the given output to the specified directory as a .pkl file or as a figure
        :param output: the output to be stored
        :param output_directory: the directory in which to store the output
        :param file_name: the name of the file with the output data
        :return:
        """

        # This is the output from PatternGenerationStage, it is more complicated to store 
        # TODO Not very robust, could be improved   
        if file_name is not None and "PIL_images.bmp" in file_name:
            Path(os.path.dirname(output_directory)).mkdir(parents=True, exist_ok=True)

            for img_name, subframes in output.items():
                # Create the path and folder with the image name
                output_path = os.path.join(output_directory, img_name + "/")
                Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

                for subframe in subframes:
                    # subframe is (str, (PIL, PIL)) - with the first image the overlay, second projected
                    overlay_path = os.path.join(output_path, subframe[0] + "_overlay.bmp")
                    projected_path = os.path.join(output_path, subframe[0] + ".bmp")
                    subframe[1][0].save(overlay_path)
                    subframe[1][1].save(projected_path)
            return 

        # if the output is a directory, just copy to output folder
        if isinstance(output, str) and os.path.isdir(output):
            if output == output_directory:
                # Special case when we generate the patterns
                # The input folder does not exist as we are creating the data
                # The input is in the output folder
                return
            copy_tree(output, output_directory)
            return

        # check if need to complete file name
        if not file_name and isinstance(output, matplotlib.pyplot.Figure):
            file_name = output._suptitle.get_text()

        # define file path
        output_path = os.path.join(output_directory, file_name)

        # create directory and parents, if they do not exist
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

        # if output path is a directory and not a file, just copy to output folder
        if os.path.isdir(output_path):
            copy_tree(output_path, output_directory)
            return

        # store as .pkl file
        if ".pkl" in output_path:
            with open(output_path, 'wb') as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # store as image file
        elif ".png" in output_path or ".jpg" in output_path:
            output.savefig(output_path)

        elif ".gif" in output_path:
            output["gif_data"][0].save(output_path, format="GIF", save_all=True,append_images=output["gif_data"][1:], duration=output["gif_time"], loop=0)

        else:
            # store as regular file
            with open(output_path, 'w') as f:
                print(output, file=f)

    @staticmethod
    def load_output(file_name):
        """
        This function loads output from a file using the provided file name
        """
        # extract suffix
        suffix = os.path.basename(file_name).split('.')[-1]
        # load according to file type
        with open(file_name, 'rb') as file:
            if suffix == 'pkl' or not suffix:
                return pickle.load(file)

    @staticmethod
    def strip_idx(string):
        """
		This function removes the numerical suffix of a variable name to help classify.
		"""
        return string.rstrip(''.join(str(kk) for kk in range(10)))