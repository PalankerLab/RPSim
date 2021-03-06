import os
import pickle
import logging
from pathlib import Path
from datetime import datetime


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
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)

        # Add handler to the logger
        logger.addHandler(file_handler)

    @staticmethod
    def generate_output_directory(parent_directory, directory_prefix):
        """
        This function creates a new output directory in the given parent directory and with the given prefix name
        :param parent_directory: the path of the parent directory in which the new folder should be created
        :param directory_prefix: the prefix to use for directory name
        :return:
        """
        # generate unique directory name using date and time
        current_date_and_time = datetime.now()
        output_directory = os.path.join(parent_directory, directory_prefix + "-" + current_date_and_time.strftime(
            "%Y_%m_%d-%H_%M_%S").split('.')[0])

        # create directory and parents, if they do not exist
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        # verify it was created successfully
        if not (os.path.isdir(output_directory) and os.path.exists(output_directory)):
            raise NotADirectoryError

        return output_directory

    @staticmethod
    def store_output(output, output_directory, file_name, as_pickle=True, as_figure=False):
        """
        This function stores the given output to the specified directory as a .pkl file or as a figure
        :param output: the output to be stored
        :param output_directory: the directory in which to store the output
        :param file_name: the name of the file with the output data
        :param as_pickle: store output in .pkl format
        :param as_figure: store output as figure
        :return:
        """
        # define file name
        output_file_path = os.path.join(output_directory, file_name)

        # create directory and parents, if they do not exist
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)

        if as_pickle:
            # store using pickle
            with open(output_file_path, 'wb') as handle:
                pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif as_figure:
            output.savefig(output_file_path)
        else:
            # dump to regular file
            with open(output_file_path, 'w') as f:
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
