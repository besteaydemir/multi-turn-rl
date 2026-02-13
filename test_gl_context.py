#!/usr/bin/env python3
"""Quick habitat-sim GL context test."""
import os, sys

# Runtime fix: ensure system NVIDIA GL libs are findable
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
    "/dss/dsshome1/06/di38riq/habitat-sim/10_nvidia.json")
os.environ.pop("DISPLAY", None)

# Add system GL to LD_LIBRARY_PATH
sys_gl = "/usr/lib/x86_64-linux-gnu"
conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
ld = os.environ.get("LD_LIBRARY_PATH", "")
new_ld = f"{sys_gl}:{conda_lib}:{ld}" if ld else f"{sys_gl}:{conda_lib}"
os.environ["LD_LIBRARY_PATH"] = new_ld

print(f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")
print(f"__EGL_VENDOR_LIBRARY_FILENAMES={os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']}")

import habitat_sim
print(f"habitat_sim version: {habitat_sim.__version__}")

cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
cfg.gpu_device_id = 0
cfg.enable_physics = False

agent_cfg = habitat_sim.agent.AgentConfiguration()
config = habitat_sim.Configuration(cfg, [agent_cfg])

print("Creating simulator...")
sim = habitat_sim.Simulator(config)
print("SUCCESS: Simulator created!")

obs = sim.get_sensor_observations()
print(f"Observation keys: {list(obs.keys())}")
if "color_sensor" in obs:
    print(f"Color sensor shape: {obs['color_sensor'].shape}")

sim.close()
print("DONE - GL context works!")
