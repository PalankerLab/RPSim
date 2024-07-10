from omegaconf import OmegaConf
import importlib

def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def load_yaml(fname):
    configuration = OmegaConf.load(fname)
    return configuration

if __name__ == "__main__":
    configuration = load_yaml("./configs/flat_human_bipolar_ps100.yaml")





