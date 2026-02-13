"""Load configuration from YAML file."""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """Load config from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_args_with_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """Merge command line args with YAML config (CLI takes precedence)."""
    # Convert config dict to namespace
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args


def add_config_arg_to_parser(parser: argparse.ArgumentParser):
    """Add --config argument to parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config)",
    )
