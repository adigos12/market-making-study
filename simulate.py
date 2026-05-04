from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# frozen=True makes Panel immutable after construction,
# protecting ground-truth parameters from accidental mutation.
@dataclass(frozen=True)
class Panel:
    """
    A simulated panel of trading sessions under known parameters.

    Attributes
    ----------
    mu_true : float
        True informed-trader fraction. In [0, 1].
    theta_true : float
        True prior probability of the high state V_H. In [0, 1].
    V_H, V_L : float
        High and low asset values, with V_H > V_L.
    trades : pandas.DataFrame
        Trade-level data. Columns are populated incrementally as
        the simulator is built; final schema includes:
            session_id, trade_idx, V_true, Z_true,
            direction, size, aggressive, delta_p
    """
    mu_true: float
    theta_true: float
    V_H: float
    V_L: float
    trades: pd.DataFrame

    @property
    def K(self) -> int:
        """Number of independent sessions, derived from the trades frame."""
        return self.trades["session_id"].nunique()

    def __post_init__(self) -> None:
        """Validate parameter bounds at construction."""
        if not 0 <= self.mu_true <= 1:
            raise ValueError(f"mu_true must be in [0, 1], got {self.mu_true}")
        if not 0 <= self.theta_true <= 1:
            raise ValueError(f"theta_true must be in [0, 1], got {self.theta_true}")
        if self.V_H <= self.V_L:
            raise ValueError(f"V_H ({self.V_H}) must be strictly greater than V_L ({self.V_L})")
        # TODO: schema validation on `trades` columns once all are populated.