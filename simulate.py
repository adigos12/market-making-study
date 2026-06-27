from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class FeatureConfig:
    """
    Parameters governing the trade-level feature distributions.

    These are nuisance parameters of the simulator: they control how
    informed and uninformed trades differ in their observable features,
    but they are NOT the model parameters (mu, theta) we later estimate.

    Attributes
    ----------
    size_mu_informed, size_mu_uninformed : float
        Mean of the underlying normal for the lognormal size distribution.
        Informed traders default to a higher mean (larger orders).
    size_sigma : float
        Std of the underlying normal, shared across both classes.
        Holding this fixed shifts the size distribution between classes
        without reshaping it, keeping the separation interpretable.
    p_aggressive_informed, p_aggressive_uninformed : float
        Bernoulli probability that an order is "aggressive".
    """
    size_mu_informed: float = 4.0
    size_mu_uninformed: float = 3.0
    size_sigma: float = 0.6
    p_aggressive_informed: float = 0.7
    p_aggressive_uninformed: float = 0.3

    def __post_init__(self) -> None:
        if self.size_sigma <= 0:
            raise ValueError(f"size_sigma must be positive, got {self.size_sigma}")
        for name, p in [
            ("p_aggressive_informed", self.p_aggressive_informed),
            ("p_aggressive_uninformed", self.p_aggressive_uninformed),
        ]:
            if not 0 <= p <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {p}")


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

def simulate_session(
    mu: float,
    theta: float,
    n_trades: int,
    V_H: float,
    V_L: float,
    session_id: int = 0,
    rng: np.random.Generator | None = None,
    features: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Simulate a single trading session under the Glosten-Milgrom model.

    Returns a DataFrame with one row per trade, columns:
        session_id, trade_idx, V_true, Z_true, direction, size, aggressive.
    """
    if rng is None:
        rng = np.random.default_rng()
    if features is None:
        features = FeatureConfig()

    # 1. Realised state for this session.
    V_true = V_H if rng.random() < theta else V_L

    # 2. Informed indicator per trade: Z_i ~ Bernoulli(mu).
    Z = rng.random(n_trades) < mu

    # 3. Direction.
    informed_buys = (V_true == V_H)
    uninformed_buys = rng.random(n_trades) < 0.5
    is_buy = np.where(Z, informed_buys, uninformed_buys)
    direction = np.where(is_buy, 1, -1)

    # 4. Size: lognormal, mean depends on Z.
    size_mu_per_trade = np.where(
        Z, features.size_mu_informed, features.size_mu_uninformed
    )
    size = rng.lognormal(mean=size_mu_per_trade, sigma=features.size_sigma)

    # 5. Aggressive: Bernoulli, probability depends on Z.
    p_agg = np.where(
        Z, features.p_aggressive_informed, features.p_aggressive_uninformed
    )
    aggressive = (rng.random(n_trades) < p_agg).astype(int)

    return pd.DataFrame({
        "session_id": session_id,
        "trade_idx": np.arange(n_trades),
        "V_true": V_true,
        "Z_true": Z.astype(int),
        "direction": direction,
        "size": size,
        "aggressive": aggressive,
    })


def simulate_panel(
    mu: float,
    theta: float,
    K: int,
    n_trades_per_session: int,
    V_H: float,
    V_L: float,
    seed: int | None = None,
    features: FeatureConfig | None = None,
) -> Panel:
    """
    Simulate a panel of K independent trading sessions and return as a Panel.
    """
    rng = np.random.default_rng(seed)
    if features is None:
        features = FeatureConfig()

    session_frames = [
        simulate_session(
            mu=mu,
            theta=theta,
            n_trades=n_trades_per_session,
            V_H=V_H,
            V_L=V_L,
            session_id=k,
            rng=rng,
            features=features,
        )
        for k in range(K)
    ]

    trades = pd.concat(session_frames, ignore_index=True)

    return Panel(
        mu_true=mu,
        theta_true=theta,
        V_H=V_H,
        V_L=V_L,
        trades=trades,
    )