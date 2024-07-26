## MP20 RAT MP20

MP20 = {}

# geometry-defined MP20
MP20["model"]                            = "monopolar"     # model geometry: monopolar or bipolar
MP20["pixel_size"]                       = 20           # pixel size
MP20["pixel_size_suffix"]                = ""         # If large format is required, use "-lg", else use ""
MP20["frame_width"]                      = 750       # implant radius in mm
MP20["geometry"]                         = "Flat_rat"  # geometry settings: HC/Pillar/Flat devices in
# pdish/rat/human
MP20["number_of_diodes"]                 = 1             # number of photo diodes per pixel
MP20["sirof_capacitance"]                = 6             # SIROF capacitance in mF/cm^2
MP20["photosensitive_area_edge_to_edge"] = 16           # edge-to-edge size of the photosensitive area
MP20["active_electrode_radius"]          = 4.5            # radius of the active electrode in um
MP20["light_to_current_conversion_rate"] = 0.5           # light to current conversion rate in A/W
MP20["photosensitive_area"]              = 158.085252133623   # total photosensitive area in um^2. Assign
# "None" for
# auto calculation

# R matrix parameters
MP20["r_matrix_output_file"]             = f'R_{MP20["geometry"]}_PS{MP20["pixel_size"]}{MP20["pixel_size_suffix"]}.pkl'
MP20["r_matrix_conductivity"]            = 1             # conductivity scaling factor of the electrolyte

# dynamic simulation MP20
MP20["Ipho_scaling"]                     = 1  # artificial scaling of photo current, useful for
                                                     # parametric sweep (e.g. S-D curve)
MP20["Isat"]                             = 0.3          # diode saturation current in pA
MP20["ideality_factor"]                  = 1.5          # ideality factor n of the diode
MP20["shunt_resistance"]                 = None         # shunt resistance in Ohm. Assign "None" if no shunt
MP20["initial_Vactive"]                  = 0.4          # Initial bias of the active electrode in V
MP20["temperature"]                      = 37
MP20["nominal_temperature"]              = 25
MP20["simulation_duration_sec"]          = 2.0             # simulation duration in seconds
MP20["simulation_resolution_ms"]         = None          # None defaults to Xyce inner value

# input paths
MP20["user_files_path"]                  = None          # If set to None, defaults to inner user_files directory
MP20["pixel_label_input_file"]           = f'image_sequence/pixel_label_PS{MP20["pixel_size"]}{MP20["pixel_size_suffix"]}.pkl'

# define input files for monopolar arrays
MP20["monopolar"] = \
    {
    "return_to_active_area_ratio": 4.0876,              # ratio between return area and total active area 
    "r_matrix_simp_ratio": 0.1,
    "r_matrix_input_file_px_pos": f'r_matrix/COMSOL_results/PS{MP20["pixel_size"]}{MP20["pixel_size_suffix"]}_pos.csv',
    "r_matrix_input_file_active": f'r_matrix/COMSOL_results/{MP20["geometry"]}/{MP20["geometry"]}_PS{MP20["pixel_size"]}_UCD_active.csv',
    "r_matrix_input_file_EP_return_2D": f'r_matrix/COMSOL_results/{MP20["geometry"]}/{MP20["geometry"]}_PS{MP20["pixel_size"]}_EP_return_2D-whole.csv',
    "r_matrix_input_file_diagonal": f'r_matrix/COMSOL_results/{MP20["geometry"]}/{MP20["geometry"]}_PS{MP20["pixel_size"]}_EP_self.csv', # used for resistive mesh only
    "r_matrix_input_file_non_diagonal": f'r_matrix/COMSOL_results/{MP20["geometry"]}/{MP20["geometry"]}_PS{MP20["pixel_size"]}_EP_Rmat.csv' # used for resistive mesh only
    }

# define input files for bipolar arrays
bipolar_dict = \
    {
    "additional_edges": 142,                                     # bipolar only: edge segments of the return
    "r_matrix_simp_ratio": 0.1,
    "r_matrix_input_file_px_pos": f'r_matrix/COMSOL_results/PS{MP20["pixel_size"]}{MP20["pixel_size_suffix"]}_pos.csv',
    "r_matrix_input_file_active": f'r_matrix/COMSOL_results/{MP20["geometry"]}/{MP20["geometry"]}_PS{MP20["pixel_size"]}_UCD_active.csv',
    "r_matrix_input_file_return": f'r_matrix/COMSOL_results/{MP20["geometry"]}/{MP20["geometry"]}_PS{MP20["pixel_size"]}_UCD_return.csv',
    "r_matrix_input_file_return_neighbor": f'r_matrix/COMSOL_results/{MP20["geometry"]}/{MP20["geometry"]}_PS{MP20["pixel_size"]}_UCD_return_neighbor.csv', # used for resistive mesh only
    }
if MP20["model"] == 'bipolar': # Special file existing only for the bipolar PS100 and PS75 MP20s
        bipolar_dict["r_matrix_input_file_return_near"] = f'r_matrix/COMSOL_results/{MP20["geometry"]}/{MP20["geometry"]}_PS{MP20["pixel_size"]}_UCD_return_near.csv'

MP20["bipolar"] = bipolar_dict
