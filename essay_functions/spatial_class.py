
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TemporalDatas:
    name : str
    data_arr : list
    path : str = field(init=0)
    times : np.ndarray = field(init=0)
    N : int = field(init=0)
    N_u : int = field(init=0)
    final_time : float = field(init=0)
    time_step : np.ndarray = field(init=0)
    u : np.ndarray = field(init=0)
    u_prime : np.ndarray = field(init=0)
    kinetic_energy: np.ndarray = field(init=0)
    variance: np.ndarray = field(init=0)
    u_rms: np.ndarray = field(init=0)
    turb_int: np.ndarray = field(init=0)
    diss_coef: float = field(init=0)
    flat_coef: float = field(init=0)
    u_bar_s : np.ndarray = field(init=0)
    u_prime_bar_s : np.ndarray = field(init=0)
    u_bar_t : float = field(init=0)
    cov : float = field(init=0)
    corr_coef: float = field(init=0)
    positions: list = field(init=0)