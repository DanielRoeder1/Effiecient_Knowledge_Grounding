import yaml
from types import SimpleNamespace

# YAML -> SimpleNamespace
# from: https://gist.github.com/jdthorpe/313cafc6bdaedfbc7d8c32fcef799fbf
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    
    if config.paths.wandb_path is not None:
            with open(config.paths.wandb_path, 'r') as f:
                config.wandb_key = f.read()

    return config

class Loader(yaml.Loader):
    pass

def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return SimpleNamespace(**dict(loader.construct_pairs(node)))

Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)