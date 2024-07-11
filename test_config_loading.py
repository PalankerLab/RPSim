from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)
from hydra.utils import instantiate
import importlib
from pprint import pprint
def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    print(module, cls)
    return getattr(importlib.import_module(module, package=None), cls)


def load_yaml(fname):
    configuration = OmegaConf.load(fname)
    return configuration

def instantiate_obj_from_config(config):
    assert 'target' in config, "Please specify target object class!"
    return get_obj_from_str(config['target'])(**config.get('params', dict()))

def test_hydra(cfg):
    sf = instantiate(cfg.subframes)
    print(sf)
    print(type(sf))
    pprint(sf)
    return sf


if __name__ == "__main__":
    # configuration = load_yaml("./configs/flat_human_bipolar_ps100.yaml")
    # OmegaConf.resolve(configuration)
    # print(configuration)
    subf = load_yaml("./configs/test_hydra.yaml")
    OmegaConf.resolve(subf)
    video_config = load_yaml("./configs/video_sequence.yaml")
    print(video_config.subframes[1].duration_ms)
    # subframe_configs = video_config['subframes']
    subframes = instantiate(video_config.subframes)
    list_of_frames = instantiate(video_config.list_of_frames)

    list_projections = instantiate(video_config.list_projections)





