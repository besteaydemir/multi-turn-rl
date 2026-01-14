#!/usr/bin/env python3
"""
Example script showing how to use SimulatorConfig from the main config file.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import RLTrainingConfig, SimulatorConfig

def main():
    print("=" * 80)
    print("EXAMPLE: Using SimulatorConfig from Config File")
    print("=" * 80)
    
    # Method 1: Load from YAML file
    print("\n[Method 1] Loading from YAML file...")
    config = RLTrainingConfig.from_yaml("example_config.yaml")
    print(f"Loaded config for experiment: {config.experiment_name}")
    print(f"Simulator max_steps: {config.simulator.max_steps}")
    print(f"Simulator temperature: {config.simulator.temperature}")
    print(f"Simulator track_action_tokens: {config.simulator.track_action_tokens}")
    
    # Method 2: Create programmatically
    print("\n[Method 2] Creating programmatically...")
    sim_config = SimulatorConfig(
        max_steps=5,
        track_action_tokens=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        min_action_tokens=15,
        max_action_tokens=80
    )
    print(f"Created SimulatorConfig:")
    print(f"  max_steps: {sim_config.max_steps}")
    print(f"  temperature: {sim_config.temperature}")
    print(f"  top_p: {sim_config.top_p}")
    print(f"  min_action_tokens: {sim_config.min_action_tokens}")
    
    # Method 3: Use in full config
    print("\n[Method 3] Using in full RLTrainingConfig...")
    full_config = RLTrainingConfig(
        experiment_name="custom_experiment",
        simulator=sim_config  # Use custom simulator config
    )
    print(f"Created full config with custom simulator settings")
    
    # Print full summary
    print("\n[Full Config Summary]")
    full_config.print_summary()
    
    # Save to file
    print("\n[Saving Config]")
    output_path = "custom_config.yaml"
    full_config.to_yaml(output_path)
    print(f"Saved to {output_path}")
    
    print("\n" + "=" * 80)
    print("HOW TO USE WITH EPISODESIMULATOR:")
    print("=" * 80)
    print("""
# Load config
config = RLTrainingConfig.from_yaml("example_config.yaml")

# Create simulator with config
simulator = EpisodeSimulator(
    model=model,
    processor=processor,
    config=config.simulator,  # Pass the simulator config
    device=config.training.device
)

# All simulator parameters are now managed through the config file!
    """)

if __name__ == "__main__":
    main()
