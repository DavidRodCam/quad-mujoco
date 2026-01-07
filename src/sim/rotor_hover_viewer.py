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

  model.opt.timestep = float(cfg["sim"]["dt"])

  u_ids = []
  for name in ["u1", "u2", "u3", "u4"]:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
      raise RuntimeError(f"Missing actuator {name}")
    u_ids.append(aid)

  m = float(np.sum(model.body_mass))
  g = float(-model.opt.gravity[2])
  u_hover = (m * g) / 4.0
  print(f"[INFO] ui_hover = {u_hover:.2f} N each")

  with mujoco.viewer.launch_passive(model, data) as viewer:
    # set a nice starting view
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90

    while viewer.is_running():
      for aid in u_ids:
        data.ctrl[aid] = u_hover

      mujoco.mj_step(model, data)

      # camera follows the quad
      viewer.cam.lookat[:] = data.qpos[0:3]

      viewer.sync()



if __name__ == "__main__":
  main()
