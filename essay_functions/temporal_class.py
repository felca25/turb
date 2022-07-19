from dataclasses import dataclass, field

import numpy as np


@dataclass
class TemporalData:
    
    path: str
    name: str
    index: str
    u: np.ndarray
    times: np.ndarray
    final_time: float = field(init=0)
    time_step: float = field(init=0)
    N: int = field(init=0)
    u_bar_t: float = field(init=0)
    u_prime: np.ndarray = field(init=0)
    kinetic_energy: float = field(init=0)
    variance: float = field(init=0)
    u_rms: float = field(init=0)
    turb_int: float = field(init=0)
    diss_coef: float = field(init=0)
    flat_coef: float = field(init=0)
    u_x_pdf : np.ndarray = field(init=0)
    u_pdf: np.ndarray = field(init=0)
    u_prime_x_pdf : np.ndarray = field(init=0)
    u_prime_pdf: np.ndarray = field(init=0)
    