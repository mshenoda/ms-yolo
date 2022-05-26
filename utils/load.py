import yaml

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)  # dict
    return yaml_cfg
