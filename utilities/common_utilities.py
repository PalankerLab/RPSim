import os
import csv
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot
from datetime import datetime
from distutils.dir_util import copy_tree
from configuration.configuration_manager import Configuration


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
                    overlay_path = os.path.join(output_path, subframe[0] + "_overlay.png")
                    projected_path = os.path.join(output_path, subframe[0] + ".bmp")
                    subframe[1][0].save(overlay_path)
                    subframe[1][1].save(projected_path)
            return 
        
        if file_name is not None and "multiplexed.bmp" in file_name:
            Path(os.path.dirname(output_directory)).mkdir(parents=True, exist_ok=True)

            for img_name, subframes in output.items():
                # Create the path and folder with the image name
                output_path = os.path.join(output_directory, img_name + "/")
                Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

                for subframe in subframes:
                    # subframe is (str, (PIL, PIL)) - with the first image the overlay, second projected
                    # overlay_path = os.path.join(output_path, subframe[0] + "_overlay.png")
                    projected_path = os.path.join(output_path, subframe[0] + "_multiplexed.bmp")
                    subframe[1].save(projected_path)
                    # subframe[1][1].save(projected_path)
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
        elif (".png" in output_path or ".jpg" in output_path) and output:
            output.savefig(output_path)

        elif ".gif" in output_path:
            output["gif_data"][0].save(output_path, format="GIF", save_all=True,append_images=output["gif_data"][1:], duration=output["gif_time"], loop=0)
        
        elif ".csv" in output_path:
            pd.DataFrame(output).to_csv(output_path, index=False, header=False)
        
        elif ".py" in output_path:
            # Move the script generated after creating the projection sequence
            os.rename(output, os.path.join(output_directory, file_name))

        else:
            # store as regular file
            with open(output_path, 'w') as f:
                print(output, file=f)


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


    def strip_idx(string):
        """
		This function removes the numerical suffix of a variable name to help classify.
		"""
        return string.rstrip(''.join(str(kk) for kk in range(10)))
    

def get_post_process_files(file_path, rows, rows_meta_data, stage=None):
    """
    This function is used to load the COMSOL csv files for the post process stage. 
    All post process files with commented lines start with the '%' character. 
    They contain three rows with the X, Y, and Z grids. Depending on the file, X, Y or Z grids
    may be required. The function returns a correctly formatted csv file. 
    
    Params: 
        file_path (str): Path to csv file to load
        rows (list(list(str))): the first N rows from the file to load
        rows_meta_data (list(bool)): Determines which rows are comments/meta data in the first N rows
        stage (CommonRunStage.PostProcessStage): Optional, used to load the comsol depth value directly into stage
    
    Returns: 
    data (Numpy.array) with the configurations below
    - The UCD active (common to monopolar and bipolar) & UCD return (bipolar only)
        - The first row contains the X grid 
        - All other rows contain voltages data
    - The UCD return near (bipolar only) &  EP return 2D whole (monopolar only)
        - The first row contains the X grid 
        - The second row contains the Y grid
        - All other rows contain voltages data
    """

    # The grids are the only lines that are not commented in the headers block, and not voltages data
    # Reverse the boolean list, find the index of the first True occurence, and then reverse again the indices
    last_header = len(rows_meta_data) - rows_meta_data[::-1].index(True)
    grids = [row for row, idx in zip(rows[:last_header], rows_meta_data[:last_header]) if not idx]

    data = np.loadtxt(file_path, delimiter=',', skiprows=last_header)

    if "return_near" in file_path or "return_2D" in file_path:
        xy_grids = np.array(grids[:2], dtype=np.float64)
        data = np.vstack((xy_grids, data))
    
    # For UCD active or return
    else: 
        x_grid = np.array(grids[0], dtype=np.float64)
        data = np.vstack((x_grid, data))
        if stage and len(grids) >= 3:
            # Load the z grid (or depths) directly into the post proces stage - pass through numpy, otherwise it is read as str
            stage.comsol_depth_values = np.array(grids[2], dtype=np.float64).tolist()
    
    return data

def get_resistive_mesh_files(file_path, rows_meta_data):
    """
    This function is used to load the COMSOL csv files with in the resistive mesh stage.
    rows_meta_data indicates which rows are comments and how many rows must be skipped
    when loading the data. 
    
    Params: 
        file_path (str): Path to csv file to load
        rows_meta_data (list(bool)): Determines which rows are comments/meta data in the first N rows

    Returns:
    data (Numpy.array) with the configurations below
    - The EP Rmat (monopolar only) & EP Self (monopolar only) & UCD return neighbor (bipolar only)
        - Only data, all headers are discarded 
    """
    
    # The grids are the only lines that are not commented in the headers block, and not voltages data
    # Reverse the boolean list, find the index of the first True occurence, and then reverse again the indices
    last_header = len(rows_meta_data) - rows_meta_data[::-1].index(True)
    data = np.loadtxt(file_path, delimiter=',', skiprows=last_header)

    return data

def load_csv(file_path, nb_rows_to_analyze=100, stage=None):
    """
    Loads csv files generated by COMSOL into formatted Numpy.array. 
    It is compatible with csv files with or without headers as 
    long as they contain the "%" character indicating commented lines. 
    Params:
        file_path (str): The path to the csv file to load 
        nb_rows_to_analyse (int): The number of headers rows to analyze in the csv file. 
                                    - Set to 100 as default as it is fast and has much more data rows than comments rows. 
    Returns:
        data (Numpy.array) - the configuration varies 
    """

    rows = []
    rows_meta_data = []
    # Open the CSV file
    with open(file_path, 'r', newline='') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        # Read and print the first 20 rows
        for row_number, row in enumerate(csv_reader):
            rows.append(row)
            # Check wich rows contain meta data (i.e. the ones starting with % character)
            rows_meta_data.append(any(['%' in x for x in row]))

            # Break the loop after certain number of rows
            if row_number == nb_rows_to_analyze:
                break
    
    if any(rows_meta_data):
        if ("UCD" in file_path or  "return_2D" in file_path) and not "neighbor" in file_path:
            data = get_post_process_files(file_path, rows, rows_meta_data, stage)
        elif "EP" in file_path or "neighbor" in file_path:
            data = get_resistive_mesh_files(file_path, rows_meta_data)
            
    else:
        data = np.loadtxt(file_path, delimiter=',')
        
    return data
