"""
simulate.py
-----------

Synthetic data generator for the Glosten-Milgrom (1985) two-state
market making model. Produces panels of trading sessions with known
ground-truth parameters, suitable for benchmarking the MLE and EM
estimators developed elsewhere in this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
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
        Trade-level data. Columns:
            session_id : int     (which session this trade belongs to)
            trade_idx  : int     (within-session index)
            V_true     : float   (the realised state for the session)
            Z_true     : int     (1 if informed, 0 otherwise)
            direction  : int     (+1 buy, -1 sell)
            size       : float   (order size)
            aggressive : int     (1 if aggressive, 0 otherwise)
            delta_p    : float   (price impact, currently a stub)

    The number of sessions K is derived dynamically from `trades`
    via the `K` property to prevent inconsistency.
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