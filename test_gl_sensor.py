#!/usr/bin/env python3
"""Diagnose GL rendering issue with sensors."""
import os

os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json"
os.environ.pop("DISPLAY", None)
os.environ["MAGNUM_LOG"] = "verbose"
os.environ["MAGNUM_GPU_VALIDATION"] = "on"

print(f"LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH', '(not set)')}")

import habitat_sim
print(f"habitat_sim {habitat_sim.__version__}")

# Check CUDA
import torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Driver: {torch.version.cuda}")

# Try with explicit sensor like HabitatEnv does
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
sim_cfg.gpu_device_id = 0
sim_cfg.enable_physics = False

color_spec = habitat_sim.CameraSensorSpec()
color_spec.uuid = "color_sensor"
color_spec.sensor_type = habitat_sim.SensorType.COLOR
color_spec.resolution = [480, 640]  # [H, W]
color_spec.position = [0.0, 1.5, 0.0]
color_spec.hfov = 90.0

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [color_spec]

print("Creating simulator with color sensor...")
config = habitat_sim.Configuration(sim_cfg, [agent_cfg])

try:
    sim = habitat_sim.Simulator(config)
    print("Simulator created!")
    agent = sim.initialize_agent(0)
    obs = sim.get_sensor_observations()
    print(f"Observation keys: {list(obs.keys())}")
    if "color_sensor" in obs:
        print(f"Color shape: {obs['color_sensor'].shape}, dtype: {obs['color_sensor'].dtype}")
    sim.close()
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
