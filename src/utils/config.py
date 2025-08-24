import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = 'configs/config.yaml') -> Dict[str, Any]:
    """
    Loads configuration parameters from a YAML file.

    Args:
        config_path (str):  The path to the YAML configuration file

    Returns:
        Dict[str, Any]:  A dictionary containing the configuration parameters.
    """
    # Verify path to Configuration File exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration File not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    """
    Testing the load_config method
    """
    try:
        curr_config = load_config()
        print("Configuration successfully loaded")
        print(f"Data config: {curr_config.get('data')}\n")    # Verify Data Configuration
        print(f"Training config: {curr_config.get('training')}\n")    # Verify Training Configuration
        print(f"Model config: {curr_config.get('model')}\n")  # Verify Model Configuration
        print(f"Paths config: {curr_config.get('paths')}\n")  # Verify Paths Configuration
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred while loading config: {e}")
