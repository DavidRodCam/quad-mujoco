import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
import mujoco
import matplotlib.pyplot as plt


@dataclass
class SimConfig:
  dt: float
  duration: float
  seed: int


@dataclass
class OutputConfig:
  plot_path: str


@dataclass
class ModelConfig:
  xml_path: str


@dataclass
class Config:
  sim: SimConfig
  output: OutputConfig
  model: ModelConfig


def load_config(path: str) -> Config:
  with open(path, "r") as f:
    raw = yaml.safe_load(f)

  sim = raw.get("sim", {})
  out = raw.get("output", {})
  model = raw.get("model", {})

  return Config(
    sim=SimConfig(
      dt=float(sim.get("dt", 0.01)),
      duration=float(sim.get("duration", 10.0)),
      seed=int(sim.get("seed", 0)),
    ),
    output=OutputConfig(
      plot_path=str(out.get("plot_path", "assets/figures/m1_z_vs_time.png"))
    ),
    model=ModelConfig(
      xml_path=str(model.get("xml_path", "assets/models/quad_wrench.xml"))
    )
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  args = parser.parse_args()

  cfg = load_config(args.config)
  np.random.seed(cfg.sim.seed)

  xml_path = Path(cfg.model.xml_path)
  if not xml_path.exists():
    raise FileNotFoundError(f"Model XML not found: {xml_path}")

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  model.opt.timestep = cfg.sim.dt
  data = mujoco.MjData(model)

  # Find actuator indices by name
  thrust_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
  tau_x_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tau_x")
  tau_y_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tau_y")
  tau_z_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tau_z")

  if min(thrust_id, tau_x_id, tau_y_id, tau_z_id) < 0:
    raise RuntimeError("Missing one or more actuators: thrust/tau_x/tau_y/tau_z")

  # Compute mg from model mass (total mass)
  m = float(np.sum(model.body_mass))
  g = float(-model.opt.gravity[2])  # gravity is [0,0,-9.81], so g = 9.81
  T_hover = m * g

  steps = int(cfg.sim.duration / cfg.sim.dt)
  t = np.arange(steps) * cfg.sim.dt
  z = np.zeros(steps, dtype=np.float64)

  # Apply constant hover thrust, zero torques
  for i in range(steps):
    data.ctrl[thrust_id] = T_hover
    data.ctrl[tau_x_id] = 0.0
    data.ctrl[tau_y_id] = 0.0
    data.ctrl[tau_z_id] = 0.0

    mujoco.mj_step(model, data)

    # Freejoint position is qpos[0:3] of the quad body (x,y,z)
    z[i] = float(data.qpos[2])

  plot_path = Path(cfg.output.plot_path)
  plot_path.parent.mkdir(parents=True, exist_ok=True)

  plt.figure()
  plt.plot(t, z)
  plt.xlabel("time [s]")
  plt.ylabel("z [m]")
  plt.title(f"M1 Stage A: constant thrust (T = {T_hover:.2f} N) z vs time")
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(plot_path, dpi=200)
  print(f"[OK] Saved plot -> {plot_path}")
  print(f"[INFO] Total mass = {m:.3f} kg, hover thrust = {T_hover:.3f} N")


if __name__ == "__main__":
  main()
