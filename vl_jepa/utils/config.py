"""
Configuration management utilities
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Sanity-check a loaded config so the run fails at startup, not mid-epoch.

    Raises a ValueError listing every required key that's missing. Soft
    fields (image_transforms, evaluation, etc.) are not enforced here; this
    only covers things the training loop dereferences without a fallback.
    """
    required = [
        ('model.vision_encoder', dict),
        ('model.text_encoder', dict),
        ('model.predictor', dict),
        ('training.batch_size', int),
        ('training.gradient_accumulation_steps', int),
        ('training.num_epochs', int),
        ('training.learning_rate', (int, float)),
        ('training.weight_decay', (int, float)),
        ('training.optimizer', dict),
        ('training.scheduler', dict),
        ('data.dataset_name', str),
        ('data.data_root', str),
        ('logging', dict),
    ]
    missing = []
    bad_type = []
    for path, expected in required:
        node = config
        ok = True
        for part in path.split('.'):
            if not isinstance(node, dict) or part not in node:
                missing.append(path)
                ok = False
                break
            node = node[part]
        if ok and not isinstance(node, expected):
            bad_type.append(f"{path} (got {type(node).__name__}, expected {expected})")

    problems = []
    if missing:
        problems.append("missing keys: " + ", ".join(missing))
    if bad_type:
        problems.append("wrong types: " + "; ".join(bad_type))
    if problems:
        raise ValueError("Config validation failed -- " + " | ".join(problems))


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config saved to {save_path}")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    base_omega = OmegaConf.create(base_config)
    override_omega = OmegaConf.create(override_config)
    merged = OmegaConf.merge(base_omega, override_omega)
    
    return OmegaConf.to_container(merged, resolve=True)


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        print("Configuration:")
        print_config(config)
    else:
        print("Usage: python config.py <config_path>")
