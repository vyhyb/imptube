from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass
class EnvironmentalConditions:
    temperature: float       # °C
    rel_humidity: float      # %
    atm_pressure: float = 101325.0  # Pa


@dataclass
class BoundaryConditions:
    further_mic_dist: float  # m, distance of the further mic from the sample
    closer_mic_dist: float   # m, distance of the closer mic from the sample
    freq_limit: float        # Hz, upper frequency limit of the measurement


@dataclass
class MeasurementResults:
    freqs: np.ndarray                           # Hz
    alpha: np.ndarray                           # absorption coefficient
    reflection_factor: np.ndarray | None = None # complex reflection factor
    surface_impedance: np.ndarray | None = None # complex surface impedance (Pa·s/m)
    env: EnvironmentalConditions | None = None
    bc: BoundaryConditions | None = None

    def save(self, path: str) -> None:

        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        data = {
            "freqs": self.freqs,
            "alpha": self.alpha,
            "has_reflection_factor": self.reflection_factor is not None,
            "has_surface_impedance": self.surface_impedance is not None,
            "has_env": self.env is not None,
            "has_bc": self.bc is not None,
        }

        if self.reflection_factor is not None:
            data["reflection_factor"] = self.reflection_factor
        if self.surface_impedance is not None:
            data["surface_impedance"] = self.surface_impedance
        if self.env is not None:
            data["env_temperature"] = self.env.temperature
            data["env_rel_humidity"] = self.env.rel_humidity
            data["env_atm_pressure"] = self.env.atm_pressure
        if self.bc is not None:
            data["bc_further_mic_dist"] = self.bc.further_mic_dist
            data["bc_closer_mic_dist"] = self.bc.closer_mic_dist
            data["bc_freq_limit"] = self.bc.freq_limit

        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str) -> "MeasurementResults":

        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        with np.load(path, allow_pickle=False) as data:
            env = None
            if bool(data["has_env"]):
                env = EnvironmentalConditions(
                    temperature=float(data["env_temperature"]),
                    rel_humidity=float(data["env_rel_humidity"]),
                    atm_pressure=float(data["env_atm_pressure"]),
                )

            bc = None
            if bool(data["has_bc"]):
                bc = BoundaryConditions(
                    further_mic_dist=float(data["bc_further_mic_dist"]),
                    closer_mic_dist=float(data["bc_closer_mic_dist"]),
                    freq_limit=float(data["bc_freq_limit"]),
                )

            return cls(
                freqs=data["freqs"],
                alpha=data["alpha"],
                reflection_factor=(
                    data["reflection_factor"]
                    if bool(data["has_reflection_factor"])
                    else None
                ),
                surface_impedance=(
                    data["surface_impedance"]
                    if bool(data["has_surface_impedance"])
                    else None
                ),
                env=env,
                bc=bc,
            )