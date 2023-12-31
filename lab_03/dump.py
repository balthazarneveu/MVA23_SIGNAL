from pathlib import Path
import json
import logging
import pickle
YAML_SUPPORT = True
YAML_NOT_DETECTED_MESSAGE = "yaml is not installed, consider installing it by pip install PyYAML"
try:
    import yaml
    from yaml.loader import SafeLoader, BaseLoader
except ImportError as e:
    YAML_SUPPORT = False
    logging.warning(f"{e}\n{YAML_NOT_DETECTED_MESSAGE}")


class Dump:
    @staticmethod
    def load_yaml(path: Path, safe_load=True) -> dict:
        assert YAML_SUPPORT, YAML_NOT_DETECTED_MESSAGE
        with open(path) as file:
            params = yaml.load(
                file, Loader=SafeLoader if safe_load else BaseLoader)
        return params

    @staticmethod
    def save_yaml(data: dict, path: Path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        assert YAML_SUPPORT, YAML_NOT_DETECTED_MESSAGE
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, **kwargs)

    @staticmethod
    def load_json(path: Path,) -> dict:
        with open(path) as file:
            params = json.load(file)
        return params

    @staticmethod
    def save_json(data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as outfile:
            json.dump(data, outfile)

    @staticmethod
    def load_pickle(path: Path,) -> dict:
        with open(path, "rb") as file:
            unpickler = pickle.Unpickler(file)
            params = unpickler.load()
            # params = pickle.load(file)
        return params

    @staticmethod
    def save_pickle(data: dict, path: Path):
        with open(path, 'wb') as outfile:
            pickle.dump(data, outfile)
