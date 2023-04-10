import yaml
from types import SimpleNamespace
import argparse
import os

def load_args():
    args = parse_args()
    dir_path = os.path.dirname(__file__)    
    if args.config_path == 'colab':
        config_path = config_path = os.path.join(dir_path, "../config/config_colab.yaml")
    elif args.config_path == 'cluster':
        config_path = config_path = os.path.join(dir_path, "../config/config_cluster.yaml")
    config = load_config(config_path)
    if config.paths.wandb_path is not None:
            with open(config.paths.wandb_path, 'r') as f:
                config.wandb_key = f.read()
    return config

# YAML -> SimpleNamespace
# from: https://gist.github.com/jdthorpe/313cafc6bdaedfbc7d8c32fcef799fbf
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    return config

class Loader(yaml.Loader):
    pass

def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return SimpleNamespace(**dict(loader.construct_pairs(node)))

Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default = 'colab',
        help = "Provide path to config.yaml if not provided config_colab.yaml will be used"
    )
    args = parser.parse_args()
    return args