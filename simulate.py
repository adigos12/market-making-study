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
        High and low asset values.
    K : int
        Number of independent sessions in the panel.
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
    """
    mu_true: float
    theta_true: float
    V_H: float
    V_L: float
    K: int
    trades: pd.DataFrame