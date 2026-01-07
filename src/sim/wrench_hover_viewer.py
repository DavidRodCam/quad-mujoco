import argparse
from pathlib import Path
import numpy as np
import yaml
import mujoco
import mujoco.viewer


def load_config(path):
  with open(path, "r") as f:
    return yaml.safe_load(f)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  args = parser.parse_args()

  cfg = load_config(args.config)

  xml_path = Path(cfg["model"]["xml_path"])
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)

  dt = cfg["sim"]["dt"]
  model.opt.timestep = dt

  # Actuator ids
  thrust_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
  tau_x_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tau_x")
  tau_y_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tau_y")
  tau_z_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tau_z")

  m = float(np.sum(model.body_mass))
  g = float(-model.opt.gravity[2])
  T_hover = m * g

  print(f"[INFO] Hover thrust = {T_hover:.2f} N")

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      data.ctrl[thrust_id] = T_hover
      data.ctrl[tau_x_id] = 0.0
      data.ctrl[tau_y_id] = 0.0
      data.ctrl[tau_z_id] = 0.0

      mujoco.mj_step(model, data)
      viewer.sync()


if __name__ == "__main__":
  main()
