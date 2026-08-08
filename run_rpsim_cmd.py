"""
This is the main module of the RPSim Software Tool
"""
import os
import sys
import json
import time
import logging
import argparse
import traceback

from configuration.configuration_manager import Configuration
from run_manager import RunManager
from utilities.common_utilities import CommonUtils
from utilities.exceptions import NeededOutputNotFound
from pathlib import Path

global RPSIM_LOGGER
from run_stages.pattern_generation_stage import ProjectionSequence
from run_stages.pattern_generation_stage import Text, Grating, Rectangle, Circle, FullField, Subframe, Frame, GaussianCircle


def run_rpsim(configuration=None, run_stages=None, skip_stages=None, output_directory=None):
    """
    This is the main function of the tool which handles its overall initialization and flow
    :param configuration: a dictionary with the user's configuration parameters
    :param run_stages: the run stages that the user would like to execute
    :return:
    """
    try:

        # start runtime clock
        start_time = time.time()

        # setup logging
        RPSIM_LOGGER = CommonUtils.setup_logging(__file__)

        # parse configuration, the configuration manager and parsing is done once at the beginning of the run
        configuration_manager = Configuration(configuration)

        while next(configuration_manager):

            # remove any previous logger file handles
            RPSIM_LOGGER.handlers = [h for h in RPSIM_LOGGER.handlers if not isinstance(h, logging.FileHandler)]

            # create a new output folder for this run with the requested user prefix
            # output_directory = CommonUtils.generate_output_directory(
            #     parent_directory=os.path.join(output_directory, 'user_files', 'user_output'), add_time=False)

            # create directory and parents, if they do not exist
            Path(output_directory).mkdir(parents=True, exist_ok=True)

            # redirect logging to a file inside the newly created output directory
            CommonUtils.add_logger_file_handle(RPSIM_LOGGER, file_name=os.path.join(output_directory, 'execution.log'))

            # report the start of a new run
            RPSIM_LOGGER.info('Staring a new run')
            RPSIM_LOGGER.info("Output directory: {}".format(output_directory))

            # start a new run manager
            run_manager = RunManager(output_directory, run_stages, skip_stages)

            # update the run stages that will be executed, and report
            run_stages = run_manager.get_requested_run_stages()
            RPSIM_LOGGER.info("Requested run stages: {}".format(list(run_stages)))

            # add list of run stages to the configuration
            configuration_manager.add_parameter('run_stages', list(run_stages))

            # print current configuration to file
            config_table, calculated_table = configuration_manager.get_configuration_as_table()
            RPSIM_LOGGER.info("Running the following configuration\n====>User Inputs\n{}\n".format(config_table))
            RPSIM_LOGGER.info("\n====>Calculated Values\n{}\n".format(calculated_table))

            # save configuration to file as dictionary for bookkeeping purposes
            configuration_manager.store_configuration(output_directory=output_directory)

            # Check if the same configuration was already executed in this location;
            if run_manager.find_previous_runs:

                RPSIM_LOGGER.info(
                    "Not all stages selected for execution, searching for previous runs with the same configuration...")
                identical_configurations = configuration_manager.find_identical_configurations(
                    os.path.dirname(output_directory))

                # check if we have previous runs to rely on, if not we have a problem
                if not identical_configurations:
                    raise NeededOutputNotFound(
                        "Current run relies on previous executions, but no such executions were found. Please run full flow.")

                run_manager.initialize_missing_outputs(identical_configurations)
                RPSIM_LOGGER.info("Missing outputs were initialized successfully")

            # execute all requested run stages
            for stage in run_stages:

                # initialize stage
                run_stage = run_manager.initialize_stage(stage)

                # print initialization message
                RPSIM_LOGGER.info("Running {}".format(run_stage.__str__()))

                # run stage
                run_stage.run()

        RPSIM_LOGGER.info("Finished running all provided configurations")

    except Exception as run_error:
        # report execution errors
        RPSIM_LOGGER.info("Output directory: {}".format(output_directory))

        RPSIM_LOGGER.error(run_error)
        RPSIM_LOGGER.error("Whole error traceback:\n {}".format(traceback.format_exc()))

    finally:
        # stop runtime clock
        RPSIM_LOGGER.info("Execution time is {:.2f} minutes".format((time.time() - start_time) / 60))

        # terminate execution
        logging.shutdown()


def format_clean(value):
    if float(value).is_integer():
        return str(int(value))
    else:
        return f"{value}"


def get_rpsim_config(duration, intensity, frequency, spot_size, frame_name, averaging_resolution_ms=1, config_type='PRIMA100', geometry='Flat_rat_DG_LE'):
    generate_pattern = True

    video_sequence_name = []
    list_projections = []

    # calculate duration off
    duration_off = (1 / frequency) * 1E3 - duration

    # set up frame
    if spot_size != 0:
        # frame_content = [Subframe(duration_ms=duration, patterns=[GaussianCircle(position=(0, 0), diameter=spot_size, fill_color='white', sigma_ratio=0.4)]),
        #                  Subframe(duration_ms=duration_off, patterns=[FullField('black')])]
        frame_content = [
            Subframe(duration_ms=duration, patterns=[Circle(position=(0, 0), diameter=spot_size, fill_color='white')]),
            Subframe(duration_ms=duration_off, patterns=[FullField('black')])]
    else:
        frame_content = [Subframe(duration_ms=duration, patterns=[FullField(fill_color='white')]),
                         Subframe(duration_ms=duration_off, patterns=[FullField('black')])]
        # frame_content = [Subframe(duration_ms=duration, patterns=[Rectangle(width=8, height=8, position=(-4, 4), unit="pixel", rotation=0)]),
        #              Subframe(duration_ms=duration_off, patterns=[FullField('black')])]

    frames = [Frame(name=frame_name, repetitions=1, subframes=frame_content)]

    list_projections.append(ProjectionSequence(intensity_mW_mm2=intensity, frequency_Hz=frequency, frames=frames))
    video_sequence_name.append(f"{frame_name}")

    rpsim_config = dict()

    if config_type == 'PRIMA100':

        # geometry-defined configuration
        rpsim_config["model"] = "bipolar"
        rpsim_config["pixel_size"] = 100
        rpsim_config["pixel_size_suffix"] = ""
        rpsim_config["frame_width"] = 750
        rpsim_config["geometry"] = geometry
        rpsim_config["number_of_diodes"] = 2
        rpsim_config["sirof_capacitance"] = 6
        rpsim_config["photosensitive_area_edge_to_edge"] = 92
        rpsim_config["active_electrode_radius"] = 17
        rpsim_config["light_to_current_conversion_rate"] = 0.4
        rpsim_config["photosensitive_area"] = 4075.72

        # R matrix parameters
        rpsim_config["r_matrix_output_file"] = f'R_{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}{rpsim_config["pixel_size_suffix"]}.pkl'
        rpsim_config["r_matrix_conductivity"] = 1/3

        # dynamic simulation configuration
        rpsim_config["Ipho_scaling"] = 1
        rpsim_config["Isat"] = 0.02
        rpsim_config["ideality_factor"] = 1.14
        rpsim_config["shunt_resistance"] = 720000.0
        # shunt
        rpsim_config["initial_Vactive"] = 0
        rpsim_config["temperature"] = 37
        rpsim_config["nominal_temperature"] = 25
        rpsim_config["simulation_duration_sec"] = 1 / frequency * 6
        rpsim_config["simulation_resolution_ms"] = None

        # input paths
        rpsim_config["user_files_path"] = None
        rpsim_config["pixel_label_input_file"] = f'image_sequence/pixel_label_PS{rpsim_config["pixel_size"]}{rpsim_config["pixel_size_suffix"]}.pkl'

        # projection sequences related
        rpsim_config["video_sequence_name"] = video_sequence_name
        rpsim_config["pattern_generation"] = {"generate_pattern": generate_pattern}
        if generate_pattern:
            tmp = \
                {
                    "projection_sequences": list_projections,
                    "font_path": None,  # If set to None for, defaults to optometrist font Sloan.otf
                    "projection_sequences_stored_config": None,
                    # Used for storing the config, but part of the skipped parameters
                    "blackout_partial_diode_hexagons": False,
                    # Only takes effect for bipolar PRIMA 100-lg implants; this PRIMA100 config
                    # defaults to pixel_size_suffix="", so it's a no-op unless suffix is set to "-lg"
                    "min_diode_illumination_fraction": 0.15,
                    # min fraction of a diode's area that must be lit to count as "on"
                    "find_worst_case_diode_shift": False
                    # if True, shift the pattern laterally to the position that turns off the most diodes
                }
            rpsim_config["pattern_generation"].update(tmp)

        # define input files for monopolar arrays
        rpsim_config["monopolar"] = \
            {
                "return_to_active_area_ratio": 5.7525,  # ratio between return area and total active area
                "r_matrix_simp_ratio": 0.1,
                "r_matrix_input_file_px_pos": f'r_matrix/COMSOL_results/PS{rpsim_config["pixel_size"]}{rpsim_config["pixel_size_suffix"]}_pos.csv',
                "r_matrix_input_file_active": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_active.csv',
                "r_matrix_input_file_EP_return_2D": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_EP_return_2D-whole.csv',
                "r_matrix_input_file_diagonal": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_EP_self.csv',
                # used for resistive mesh only
                "r_matrix_input_file_non_diagonal": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_EP_Rmat.csv'
                # used for resistive mesh only
            }

        # define input files for bipolar arrays
        bipolar_dict = \
            {
                "additional_edges": 104,
                "r_matrix_simp_ratio": 0.1,
                "r_matrix_input_file_px_pos": f'r_matrix/COMSOL_results/PS{rpsim_config["pixel_size"]}{rpsim_config["pixel_size_suffix"]}_pos.csv',
                "r_matrix_input_file_active": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_active.csv',
                "r_matrix_input_file_return": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_return.csv',
                "r_matrix_input_file_return_neighbor": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_return_neighbor.csv',
                # used for resistive mesh only
            }
        if rpsim_config["model"] == 'bipolar':  # Special file existing only for the bipolar PS100 and PS75 configurations
            bipolar_dict["r_matrix_input_file_return_near"] = f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_return_near.csv'

        rpsim_config["bipolar"] = bipolar_dict

        # post-process parameters
        rpsim_config["post_process"] = \
            {
                "pulse_start_time_in_ms": (1 / frequency) * 1e3 * 5,
                "pulse_duration_in_ms": duration,
                "average_over_pulse_duration": False,
                "pulse_extra_ms": 0,  # 16,
                "time_averaging_resolution_ms": averaging_resolution_ms,
                "interpolation_resolution_ms": 1e-3,
                "multiprocessing": False,
                "cpu_to_use": None,
                "depth_values_in_um": None,
                "on_diode_threshold_mV": 50
            }

        rpsim_config["plot_results"] = \
            {
                "plot_time_window_start_ms": (1 / frequency) * 1e3 * 5,
                "plot_time_window_end_ms": [x for x in [duration]],
                "plot_potential_depth_um": 5#75
            }

    elif config_type == "MP20":
        # geometry-defined configuration
        rpsim_config["model"] = "monopolar"
        rpsim_config["pixel_size"] = 20
        rpsim_config["pixel_size_suffix"] = ""
        rpsim_config["frame_width"] = 750
        rpsim_config["geometry"] = geometry
        rpsim_config["number_of_diodes"] = 1
        rpsim_config["sirof_capacitance"] = 6
        rpsim_config["photosensitive_area_edge_to_edge"] = 16
        rpsim_config["active_electrode_radius"] = 4.5
        rpsim_config["light_to_current_conversion_rate"] = 0.5
        rpsim_config["photosensitive_area"] = 158.085252133623

        # R matrix parameters
        rpsim_config["r_matrix_output_file"] = f'R_{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}{rpsim_config["pixel_size_suffix"]}.pkl'
        rpsim_config["r_matrix_conductivity"] = 1/3

        # dynamic simulation configuration
        rpsim_config["Ipho_scaling"] = 1
        rpsim_config["Isat"] = 0.3
        rpsim_config["ideality_factor"] = 1.5
        rpsim_config["shunt_resistance"] = None
        rpsim_config["initial_Vactive"] = 0.4
        rpsim_config["temperature"] = 37
        rpsim_config["nominal_temperature"] = 25
        rpsim_config["simulation_duration_sec"] = (1 / frequency) * 6
        rpsim_config["simulation_resolution_ms"] = None

        # input paths
        rpsim_config["user_files_path"] = None
        rpsim_config["pixel_label_input_file"] = f'image_sequence/pixel_label_PS{rpsim_config["pixel_size"]}{rpsim_config["pixel_size_suffix"]}.pkl'

        # Projection sequences related
        rpsim_config["video_sequence_name"] = video_sequence_name

        rpsim_config["pattern_generation"] = {"generate_pattern": generate_pattern}
        add_projection_seq = any(generate_pattern) if type(generate_pattern) is list else generate_pattern
        if add_projection_seq:
            tmp = \
                {
                    "projection_sequences": list_projections,
                    "font_path": None,
                    "projection_sequences_stored_config": None,
                    "blackout_partial_diode_hexagons": False,  # not applicable to monopolar implants
                    "min_diode_illumination_fraction": 0.15,
                    "find_worst_case_diode_shift": False  # not applicable to monopolar implants
                }
            rpsim_config["pattern_generation"].update(tmp)

        # define input files for monopolar arrays
        rpsim_config["monopolar"] = \
            {
                "return_to_active_area_ratio": 4.0876,
                "r_matrix_simp_ratio": 0.1,#0.1,
                "r_matrix_input_file_px_pos": f'r_matrix/COMSOL_results/PS{rpsim_config["pixel_size"]}{rpsim_config["pixel_size_suffix"]}_pos.csv',
                "r_matrix_input_file_active": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_active.csv',
                "r_matrix_input_file_EP_return_2D": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_EP_return_2D-whole.csv',
                "r_matrix_input_file_diagonal": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_EP_self.csv',
                "r_matrix_input_file_non_diagonal": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_EP_Rmat.csv'
            }

        # define input files for bipolar arrays
        bipolar_dict = \
            {
                "additional_edges": 142,
                "r_matrix_simp_ratio": 0.1,
                "r_matrix_input_file_px_pos": f'r_matrix/COMSOL_results/PS{rpsim_config["pixel_size"]}{rpsim_config["pixel_size_suffix"]}_pos.csv',
                "r_matrix_input_file_active": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_active.csv',
                "r_matrix_input_file_return": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_return.csv',
                "r_matrix_input_file_return_neighbor": f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_return_neighbor.csv',
            }
        if rpsim_config["model"] == 'bipolar':
            bipolar_dict["r_matrix_input_file_return_near"] = f'r_matrix/COMSOL_results/{rpsim_config["geometry"]}/{rpsim_config["geometry"]}_PS{rpsim_config["pixel_size"]}_UCD_return_near.csv'

        rpsim_config["bipolar"] = bipolar_dict

        # post-process parameters
        rpsim_config["post_process"] = \
            {
                "pulse_start_time_in_ms": (1 / frequency) * 1e3 * 5,
                "pulse_duration_in_ms": duration,
                "average_over_pulse_duration": False,
                "pulse_extra_ms": 0,
                "time_averaging_resolution_ms": averaging_resolution_ms,
                "interpolation_resolution_ms": 1e-3,
                "multiprocessing": False,
                "cpu_to_use": None,
                "depth_values_in_um": None,
                "on_diode_threshold_mV": 50
            }

        rpsim_config["plot_results"] = \
            {
                "plot_time_window_start_ms": (1 / frequency) * 1e3 * 5,
                "plot_time_window_end_ms": [x for x in [duration]],
                "plot_potential_depth_um": 5#75
            }

    return rpsim_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RPSim with parameters.")

    parser.add_argument("--duration", type=float, required=True, help="Simulation duration (float)")
    parser.add_argument("--intensity", type=float, required=True, help="Laser intensity (float)")
    parser.add_argument("--frequency", type=float, required=True, help="Pulse frequency (float)")
    parser.add_argument("--spot_size", type=float, required=True, help="Laser spot size (float)")
    parser.add_argument("--frame_name", type=str, required=True, help="Projected frame name (str))")
    parser.add_argument("--geometry", type=str, default="Flat_rat_DG_LE", help="Geometry <Flat/Pilar>_<human/rat_dg_le/rat_rcs/pdish> (str))")
    parser.add_argument("--averaging_resolution_ms", type=float, required=True, help="Averaging resolution in ms (float))")
    parser.add_argument("--config_type", type=str, default='PRIMA100', help="Configuration type bipolar_100 or monopolar_20 (str))")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory (str)")

    args = parser.parse_args()
    print("Running RPSim cmd")
    config = get_rpsim_config(args.duration, args.intensity, args.frequency, args.spot_size, args.frame_name, args.averaging_resolution_ms, args.config_type, args.geometry)
    print(config)
    try:
        run_rpsim(configuration=config, output_directory=args.output_dir, skip_stages=["multiplexing"])
    except Exception as e:
        print("Error running RPSim cmd")
        print(e)
