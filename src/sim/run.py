import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
import mujoco
import matplotlib.pyplot as plt


# Minimal MuJoCo model (not the quad yet): a free-falling body with a free joint.
# We just need a reliable "z vs time" signal to prove MuJoCo runs end-to-end.
_XML = """
<mujoco model="m0_freebody">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass
class SimConfig:
  dt: float
  duration: float
  seed: int


@dataclass
class OutputConfig:
  plot_path: str


@dataclass
class Config:
  sim: SimConfig
  output: OutputConfig


def load_config(path: str) -> Config:
  with open(path, "r") as f:
    raw = yaml.safe_load(f)

  sim = raw.get("sim", {})
  out = raw.get("output", {})

  return Config(
    sim=SimConfig(
      dt=float(sim.get("dt", 0.01)),
      duration=float(sim.get("duration", 10.0)),
      seed=int(sim.get("seed", 0)),
    ),
    output=OutputConfig(
      plot_path=str(out.get("plot_path", "assets/figures/m0_z_vs_time.png"))
    ),
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, help="Path to YAML config")
  args = parser.parse_args()

  cfg = load_config(args.config)
  np.random.seed(cfg.sim.seed)

  # Build model & data
  model = mujoco.MjModel.from_xml_string(_XML)
  model.opt.timestep = cfg.sim.dt
  data = mujoco.MjData(model)

  steps = int(cfg.sim.duration / cfg.sim.dt)
  t = np.arange(steps) * cfg.sim.dt
  z = np.zeros(steps, dtype=np.float64)

  for i in range(steps):
    mujoco.mj_step(model, data)
    # Free joint position is in qpos[0:3] (x,y,z)
    z[i] = float(data.qpos[2])

  # Save plot
  plot_path = Path(cfg.output.plot_path)
  plot_path.parent.mkdir(parents=True, exist_ok=True)

  plt.figure()
  plt.plot(t, z)
  plt.xlabel("time [s]")
  plt.ylabel("z [m]")
  plt.title("M0 sanity check: z vs time (free body)")
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(plot_path, dpi=200)
  print(f"[OK] Saved plot -> {plot_path}")


if __name__ == "__main__":
  main()
