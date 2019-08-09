import yaml
import os
from pathlib import Path

def config_get_data_path(dir, name_id):
    config_path = os.path.join(*[dir, "config.yml"])
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        assert name_id in cfg, f'config_get_data_path: {name_id} is not in the config file'
        data_path = os.path.join(*[dir, cfg[name_id]])
        assert Path(data_path).exists(), f'config_get_data_path: File {data_path} does not exist'
        return data_path

