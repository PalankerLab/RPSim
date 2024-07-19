### PRIMA PRIMA

PRIMA = {}

# geometry-defined PRIMA
PRIMA["model"]                            = "bipolar"     # model geometry: monopolar or bipolar
PRIMA["pixel_size"]                       = 100           # pixel size
PRIMA["pixel_size_suffix"]                = "-lg"         # If large format is required, use "-lg", else use ""
PRIMA["frame_width"]                      = 1000       # implant radius in mm
PRIMA["geometry"]                         = "Flat_human"  # geometry settings: HC/Pillar/Flat devices in
# pdish/rat/human
PRIMA["number_of_diodes"]                 = 2             # number of photo diodes per pixel
PRIMA["sirof_capacitance"]                = 6             # SIROF capacitance in mF/cm^2
PRIMA["photosensitive_area_edge_to_edge"] = 92           # edge-to-edge size of the photosensitive area
PRIMA["active_electrode_radius"]          = 17            # radius of the active electrode in um
PRIMA["light_to_current_conversion_rate"] = 0.4           # light to current conversion rate in A/W
PRIMA["photosensitive_area"]              = 4075.72       # total photosensitive area in um^2. Assign "None" for
# auto calculation (only works for monopolar)

# R matrix parameters
PRIMA["r_matrix_output_file"]             = f'R_{PRIMA["geometry"]}_PS{PRIMA["pixel_size"]}{PRIMA["pixel_size_suffix"]}.pkl' 
PRIMA["r_matrix_conductivity"]            = 1             # conductivity scaling factor of the electrolyte

# dynamic simulation PRIMA
PRIMA["Ipho_scaling"]                     = 1  # artificial scaling of photo current, useful for
                                                     # parametric sweep (e.g. S-D curve)
PRIMA["Isat"]                             = 0.02          # diode saturation current in pA
PRIMA["ideality_factor"]                  = 1.14          # ideality factor n of the diode
PRIMA["shunt_resistance"]                 = 790150.0     # shunt resistance in Ohm. Assign "None" if no
# shunt
PRIMA["initial_Vactive"]                  = 0          # Initial bias of the active electrode in V
PRIMA["temperature"]                      = 37
PRIMA["nominal_temperature"]              = 25
# TODO make sure that the duration is sufficiently long compared to the frequencz, i.e. at least 6 times the period
PRIMA["simulation_resolution_ms"]         = None          # None defaults to Xyce inner value

# input paths
PRIMA["user_files_path"]                  = None          # If set to None, defaults to inner user_files directory
PRIMA["pixel_label_input_file"]           = f'image_sequence/pixel_label_PS{PRIMA["pixel_size"]}{PRIMA["pixel_size_suffix"]}.pkl'

# define input files for monopolar arrays
PRIMA["monopolar"] = \
    {
    "return_to_active_area_ratio": 5.7525,              # ratio between return area and total active area 
    "r_matrix_simp_ratio": 0.1,
    "r_matrix_input_file_px_pos": f'r_matrix/COMSOL_results/PS{PRIMA["pixel_size"]}{PRIMA["pixel_size_suffix"]}_pos.csv',
    "r_matrix_input_file_active": f'r_matrix/COMSOL_results/{PRIMA["geometry"]}/{PRIMA["geometry"]}_PS{PRIMA["pixel_size"]}_UCD_active.csv',
    "r_matrix_input_file_EP_return_2D": f'r_matrix/COMSOL_results/{PRIMA["geometry"]}/{PRIMA["geometry"]}_PS{PRIMA["pixel_size"]}_EP_return_2D-whole.csv',
    "r_matrix_input_file_diagonal": f'r_matrix/COMSOL_results/{PRIMA["geometry"]}/{PRIMA["geometry"]}_PS{PRIMA["pixel_size"]}_EP_self.csv', # used for resistive mesh only
    "r_matrix_input_file_non_diagonal": f'r_matrix/COMSOL_results/{PRIMA["geometry"]}/{PRIMA["geometry"]}_PS{PRIMA["pixel_size"]}_EP_Rmat.csv' # used for resistive mesh only
    }

# define input files for bipolar arrays
bipolar_dict = \
    {
    "additional_edges": 142,                                     # bipolar only: edge segments of the return
    "r_matrix_simp_ratio": 0.1,
    "r_matrix_input_file_px_pos": f'r_matrix/COMSOL_results/PS{PRIMA["pixel_size"]}{PRIMA["pixel_size_suffix"]}_pos.csv',
    "r_matrix_input_file_active": f'r_matrix/COMSOL_results/{PRIMA["geometry"]}/{PRIMA["geometry"]}_PS{PRIMA["pixel_size"]}_UCD_active.csv',
    "r_matrix_input_file_return": f'r_matrix/COMSOL_results/{PRIMA["geometry"]}/{PRIMA["geometry"]}_PS{PRIMA["pixel_size"]}_UCD_return.csv',
    "r_matrix_input_file_return_neighbor": f'r_matrix/COMSOL_results/{PRIMA["geometry"]}/{PRIMA["geometry"]}_PS{PRIMA["pixel_size"]}_UCD_return_neighbor.csv', # used for resistive mesh only
    }
if PRIMA["model"] == 'bipolar': # Special file existing only for the bipolar PS100 and PS75 PRIMAs
        bipolar_dict["r_matrix_input_file_return_near"] = f'r_matrix/COMSOL_results/{PRIMA["geometry"]}/{PRIMA["geometry"]}_PS{PRIMA["pixel_size"]}_UCD_return_near.csv'

PRIMA["bipolar"] = bipolar_dict
