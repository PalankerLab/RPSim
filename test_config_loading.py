from omegaconf import OmegaConf
import importlib

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


config = OmegaConf.load("./configs/flat_human_bipolar_ps100.yaml")
print(config)
print(config["r_matrix_output_file"])
# def load_update_config(fname):
#     with open(fname, 'r') as fr:
#         configuration = yaml.safe_load(fr)
#         print(configuration)
#     # additional things to conmpute
#     configuration["r_matrix_output_file"] = f'R_{configuration["geometry"]}_PS{configuration["pixel_size"]}{configuration["pixel_size_suffix"]}.pkl'
#     configuration["pixel_label_input_file"] = f'image_sequence/pixel_label_PS{configuration["pixel_size"]}{configuration["pixel_size_suffix"]}.pkl'
#     configuration["video_sequence_name"] = video_sequence_name

#     configuration["pattern_generation"] = {"generate_pattern": generate_pattern}
#     add_projection_seq = any(generate_pattern) if type(generate_pattern) is list else generate_pattern
#     if add_projection_seq:
#         tmp = \
#             {
#             "projection_sequences"              : list_projections,
#             "font_path"                         : None, # If set to None for, defaults to optometrist font Sloan.otf
#             "projection_sequences_stored_config" : None # Used for storing the config, but part of the skipped parameters
#             }
#         configuration["pattern_generation"].update(tmp)





