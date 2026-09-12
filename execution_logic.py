"""
Quantitative Trading & Risk Execution Layer
============================================
This module implements the execution and automated risk management layer,
consuming statistical time series forecasts from `modeling_engine.py`:

1. Mock Portfolio Tracker: Manages cash, asset inventory, position states, and trade logging.
2. Dynamic Risk Thresholds: Calculates rolling historical volatility baselines (95th percentile of absolute returns).
3. Execution Logic Rules: Evaluates GARCH 5-step volatility forecast against risk thresholds to issue BUY, SELL, or HOLD signals.
4. Transaction Logging: Captures trade details in structured DataFrames.
5. Integration Bridge: Line-by-line streaming runner to simulate live market execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

# Suppress benign estimation warnings for clean console presentation
warnings.filterwarnings("ignore")
try:
    from arch.utility.exceptions import DataScaleWarning
    warnings.simplefilter("ignore", DataScaleWarning)
except ImportError:
    pass

# Import statistical models from the modeling engine
from modeling_engine import (
    compute_log_returns,
    fetch_price_data,
    fit_arima_mean,
    fit_garch_volatility,
    forecast_mean_and_volatility,
)


class PositionState(str, Enum):
    CASH = "CASH"
    LONG = "LONG"


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeRecord:
    timestamp: str
    price: float
    signal: str
    order_size: float
    executed_balance: float
    portfolio_value: float
    reason: str


# =====================================================================
# 1. Mock Portfolio Tracker
# =====================================================================

class MockPortfolioTracker:
    """
    Simulates a realistic paper trading portfolio account.
    
    Attributes:
        initial_cash (float): Starting balance (default: $10,000.00).
        cash (float): Current available cash balance.
        units (float): Current asset units held in inventory.
        position_state (PositionState): Current state ('CASH' or 'LONG').
        trade_log (List[TradeRecord]): Chronological ledger of all executed transactions.
        equity_curve (List[Dict[str, Any]]): Time series of portfolio valuations.
    """

    def __init__(self, initial_cash: float = 10000.0) -> None:
        self.initial_cash: float = float(initial_cash)
        self.cash: float = float(initial_cash)
        self.units: float = 0.0
        self.position_state: PositionState = PositionState.CASH
        self.trade_log: List[TradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate mark-to-market portfolio value: Cash + Units * Price."""
        return float(self.cash + (self.units * current_price))

    def execute_buy(self, timestamp: Any, price: float, reason: str = "") -> Optional[TradeRecord]:
        """
        Execute an all-in BUY order: converts 100% of available cash into asset units.
        """
        if self.position_state == PositionState.LONG or self.cash <= 0:
            return None

        order_size = self.cash / price
        self.units = order_size
        self.cash = 0.0
        self.position_state = PositionState.LONG

        port_val = self.get_portfolio_value(price)
        record = TradeRecord(
            timestamp=str(timestamp),
            price=round(float(price), 4),
            signal=SignalType.BUY.value,
            order_size=round(float(order_size), 6),
            executed_balance=round(float(self.cash), 2),
            portfolio_value=round(float(port_val), 2),
            reason=reason,
        )
        self.trade_log.append(record)
        return record

    def execute_sell(self, timestamp: Any, price: float, reason: str = "") -> Optional[TradeRecord]:
        """
        Execute an all-out SELL order: converts 100% of asset units into cash.
        """
        if self.position_state == PositionState.CASH or self.units <= 0:
            return None

        order_size = self.units
        cash_proceeds = self.units * price
        self.cash += cash_proceeds
        self.units = 0.0
        self.position_state = PositionState.CASH

        port_val = self.get_portfolio_value(price)
        record = TradeRecord(
            timestamp=str(timestamp),
            price=round(float(price), 4),
            signal=SignalType.SELL.value,
            order_size=round(float(order_size), 6),
            executed_balance=round(float(self.cash), 2),
            portfolio_value=round(float(port_val), 2),
            reason=reason,
        )
        self.trade_log.append(record)
        return record

    def record_snapshot(
        self,
        timestamp: Any,
        price: float,
        signal: SignalType,
        vol_forecast: float,
        threshold: float,
    ) -> Dict[str, Any]:
        """Record an hourly state snapshot for equity tracking and analytics."""
        port_val = self.get_portfolio_value(price)
        snapshot = {
            "timestamp": str(timestamp),
            "price": price,
            "portfolio_value": port_val,
            "cash": self.cash,
            "units": self.units,
            "position": self.position_state.value,
            "signal": signal.value,
            "vol_forecast": vol_forecast,
            "threshold": threshold,
            "return_pct": ((port_val - self.initial_cash) / self.initial_cash) * 100.0,
        }
        self.equity_curve.append(snapshot)
        return snapshot

    def get_trade_history_df(self) -> pd.DataFrame:
        """Return the transaction ledger as a formatted pandas DataFrame."""
        if not self.trade_log:
            return pd.DataFrame(
                columns=[
                    "Timestamp",
                    "Price",
                    "Signal Type",
                    "Order Size",
                    "Executed Balance",
                    "Portfolio Value",
                    "Reason",
                ]
            )
        df = pd.DataFrame([asdict(t) for t in self.trade_log])
        df.rename(
            columns={
                "timestamp": "Timestamp",
                "price": "Price",
                "signal": "Signal Type",
                "order_size": "Order Size",
                "executed_balance": "Executed Balance",
                "portfolio_value": "Portfolio Value",
                "reason": "Reason",
            },
            inplace=True,
        )
        return df

    def get_performance_metrics(self, current_price: float) -> Dict[str, Any]:
        """Calculate comprehensive portfolio performance metrics."""
        end_val = self.get_portfolio_value(current_price)
        total_pnl = end_val - self.initial_cash
        total_return_pct = (total_pnl / self.initial_cash) * 100.0

        # Calculate max drawdown from equity curve
        if self.equity_curve:
            vals = [s["portfolio_value"] for s in self.equity_curve]
            peak = np.maximum.accumulate(vals)
            dd = (peak - vals) / peak
            max_dd_pct = float(np.max(dd)) * 100.0 if len(dd) > 0 else 0.0
        else:
            max_dd_pct = 0.0

        return {
            "initial_cash": self.initial_cash,
            "current_cash": self.cash,
            "units": self.units,
            "position_state": self.position_state.value,
            "ending_value": end_val,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd_pct,
            "total_trades": len(self.trade_log),
        }


# =====================================================================
# 2. Dynamic Risk Thresholds
# =====================================================================

def calculate_risk_threshold(
    returns: pd.Series,
    percentile: float = 95.0,
    window: Optional[int] = None,
) -> float:
    """
    Calculate the dynamic risk / regime shift threshold from historical returns.
    
    The baseline is defined as the Nth percentile (default: 95th) of the absolute
    log returns |r_t|. When predicted volatility exceeds this empirical ceiling,
    the market is considered in an extreme-risk / structural shock regime.

    Parameters:
        returns: pd.Series of log returns.
        percentile: Percentile cutoff (default 95.0 for a 5% tail shock).
        window: Optional rolling window. If provided, uses the most recent N observations.

    Returns:
        float: Threshold standard deviation value.
    """
    if window is not None and len(returns) >= window:
        sample = returns.iloc[-window:]
    else:
        sample = returns

    abs_returns = np.abs(sample.dropna())
    if len(abs_returns) == 0:
        return 0.01  # Safe default fallback

    threshold = float(np.percentile(abs_returns, percentile))
    return threshold


# =====================================================================
# 3. Execution Logic Rules
# =====================================================================

def evaluate_execution_signal(
    current_position: PositionState,
    volatility_forecast: np.ndarray | float,
    risk_threshold: float,
    current_price: float,
) -> Tuple[SignalType, str]:
    """
    Evaluate the 5-step GARCH volatility forecast against risk thresholds
    to generate an actionable quantitative execution signal:

    - BUY Signal: If position is 'CASH' and predicted volatility is low/stable
      (<= 95th percentile threshold), deploy cash into the asset.
    - SELL/EXIT Signal: If position is 'LONG' and predicted volatility spikes
      above the 95th percentile threshold, immediately exit to CASH to preserve capital.
    - HOLD Signal: If conditions are not triggered, maintain current position state.

    Parameters:
        current_position: PositionState ('CASH' or 'LONG').
        volatility_forecast: 5-step GARCH volatility forecast array (or summary float).
        risk_threshold: Historical 95th percentile risk threshold.
        current_price: Current market price.

    Returns:
        Tuple[SignalType, str]: The generated signal and the underlying quantitative rationale.
    """
    # Evaluate expected forward volatility across the forecast horizon
    if isinstance(volatility_forecast, np.ndarray):
        forward_vol = float(np.mean(volatility_forecast))
    else:
        forward_vol = float(volatility_forecast)

    # Execution Rule 1: BUY when in CASH and risk is low / regime is calm
    if current_position == PositionState.CASH and forward_vol <= risk_threshold:
        reason = (
            f"Favorable Volatility Regime: 5-step GARCH vol ({forward_vol:.6f}) <= "
            f"95th pct threshold ({risk_threshold:.6f}). Entering LONG at ${current_price:,.2f}."
        )
        return SignalType.BUY, reason

    # Execution Rule 2: SELL / EXIT when LONG and risk spikes (Regime Shift / High Fear)
    if current_position == PositionState.LONG and forward_vol > risk_threshold:
        reason = (
            f"Extreme Risk Spike Detected: 5-step GARCH vol ({forward_vol:.6f}) > "
            f"95th pct threshold ({risk_threshold:.6f}). Capital preservation EXIT to CASH at ${current_price:,.2f}."
        )
        return SignalType.SELL, reason

    # Execution Rule 3: HOLD
    if current_position == PositionState.LONG:
        reason = (
            f"Holding LONG: 5-step GARCH vol ({forward_vol:.6f}) remains within risk limits "
            f"(<= {risk_threshold:.6f})."
        )
    else:
        reason = (
            f"Remaining in CASH: Elevated volatility ({forward_vol:.6f}) exceeds entry threshold "
            f"({risk_threshold:.6f})."
        )

    return SignalType.HOLD, reason


# =====================================================================
# 4 & 5. Integration Bridge & Streaming Simulator
# =====================================================================

def step_execution_pipeline(
    portfolio: MockPortfolioTracker,
    timestamp: Any,
    current_price: float,
    rolling_returns: pd.Series,
    volatility_forecast: np.ndarray,
    percentile: float = 95.0,
    window: int = 150,
) -> Tuple[SignalType, Optional[TradeRecord], Dict[str, Any]]:
    """
    Processes a single incoming streaming tick/bar update:
    1. Computes the dynamic risk threshold.
    2. Evaluates execution rules against GARCH volatility forecast.
    3. Executes mock portfolio transactions if triggered.
    4. Records the audit snapshot.
    """
    # 1. Compute dynamic risk threshold
    threshold = calculate_risk_threshold(rolling_returns, percentile=percentile, window=window)
    vol_metric = float(np.mean(volatility_forecast))

    # 2. Evaluate signal
    signal, reason = evaluate_execution_signal(
        current_position=portfolio.position_state,
        volatility_forecast=volatility_forecast,
        risk_threshold=threshold,
        current_price=current_price,
    )

    # 3. Execute order
    trade_executed = None
    if signal == SignalType.BUY:
        trade_executed = portfolio.execute_buy(timestamp=timestamp, price=current_price, reason=reason)
    elif signal == SignalType.SELL:
        trade_executed = portfolio.execute_sell(timestamp=timestamp, price=current_price, reason=reason)

    # 4. Record snapshot
    snapshot = portfolio.record_snapshot(
        timestamp=timestamp,
        price=current_price,
        signal=signal,
        vol_forecast=vol_metric,
        threshold=threshold,
    )

    return signal, trade_executed, snapshot


def simulate_streaming_execution(
    symbol: str = "BTC-USD",
    period: str = "60d",
    interval: str = "1h",
    estimation_window: int = 150,
    simulation_steps: int = 40,
    percentile: float = 95.0,
    initial_cash: float = 10000.0,
) -> Tuple[MockPortfolioTracker, pd.DataFrame]:
    """
    Simulates a streaming environment line-by-line, sequentially updating
    time series models and executing quantitative risk rules.

    Parameters:
        symbol: Target ticker symbol (e.g. 'BTC-USD', 'TSLA').
        period: Historical lookback period.
        interval: Sampling interval.
        estimation_window: Number of historical bars used to estimate ARIMA/GARCH.
        simulation_steps: Number of forward streaming bars to simulate.
        percentile: Risk cutoff percentile (default 95.0).
        initial_cash: Initial paper trading balance (default $10,000.00).

    Returns:
        Tuple[MockPortfolioTracker, pd.DataFrame]: Portfolio object and trade history.
    """
    print("\n" + "=" * 80)
    print("   QUANTITATIVE TRADING & RISK EXECUTION LAYER   ".center(80))
    print(f"   Target: {symbol} | Stream Steps: {simulation_steps} | Initial Cash: ${initial_cash:,.2f}   ".center(80))
    print("=" * 80)

    # Ingest data using modeling_engine
    print("\n[STREAM INIT] Ingesting market feed via modeling_engine...")
    prices = fetch_price_data(symbol=symbol, period=period, interval=interval)
    returns = compute_log_returns(prices)

    total_bars = len(returns)
    if total_bars < estimation_window + simulation_steps:
        simulation_steps = max(10, total_bars - estimation_window)

    start_idx = total_bars - simulation_steps
    portfolio = MockPortfolioTracker(initial_cash=initial_cash)

    print(f"[STREAM READY] Total Ingested: {total_bars} bars.")
    print(f"[STREAM START] Replaying streaming bars from index {start_idx} to {total_bars}...\n")
    print(
        f"{'Step':<5} | {'Timestamp':<24} | {'Price':>10} | {'GARCH Vol':>10} | {'95% Thresh':>10} | "
        f"{'Signal':^7} | {'Position':^6} | {'Port Value':>12}"
    )
    print("-" * 102)

    for step_count, current_idx in enumerate(range(start_idx, total_bars)):
        ts = prices.index[current_idx]
        current_price = float(prices.iloc[current_idx])

        # Slice rolling history up to current_idx
        history_window = returns.iloc[current_idx - estimation_window : current_idx]

        # Fast model update: ARIMA + GARCH forecast
        try:
            arima_res, residuals = fit_arima_mean(history_window, order=(1, 0, 1))
            garch_res, _, scale_factor = fit_garch_volatility(residuals, p=1, q=1, rescale=True)
            forecast = forecast_mean_and_volatility(
                arima_res=arima_res,
                garch_res=garch_res,
                scale_factor=scale_factor,
                steps=5,
            )
            vol_forecast = forecast.volatility_forecast
        except Exception:
            # Robust fallback if GARCH convergence fails on particular slice
            vol_forecast = np.full(5, history_window.std())

        signal, trade, snapshot = step_execution_pipeline(
            portfolio=portfolio,
            timestamp=ts,
            current_price=current_price,
            rolling_returns=history_window,
            volatility_forecast=vol_forecast,
            percentile=percentile,
            window=estimation_window,
        )

        ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)[:16]
        vol_str = f"{snapshot['vol_forecast']:.5f}"
        thresh_str = f"{snapshot['threshold']:.5f}"
        port_val_str = f"${snapshot['portfolio_value']:,.2f}"

        # Color-coded style highlight on trade events
        signal_display = signal.value
        if signal == SignalType.BUY:
            signal_display = f"\033[92m{signal.value}\033[0m"
        elif signal == SignalType.SELL:
            signal_display = f"\033[91m{signal.value}\033[0m"

        print(
            f"{step_count+1:<5} | {ts_str:<24} | ${current_price:>9.2f} | {vol_str:>10} | {thresh_str:>10} | "
            f"{signal_display:^16} | {portfolio.position_state.value:^6} | {port_val_str:>12}"
        )

        if trade:
            action_desc = "BOUGHT" if trade.signal == "BUY" else "SOLD"
            print(
                f"  >>> [TRADE EXECUTED] {action_desc} {trade.order_size:,.4f} units @ ${trade.price:,.2f} | "
                f"Cash Left: ${trade.executed_balance:,.2f} | Port Value: ${trade.portfolio_value:,.2f}"
            )
            print(f"      Reason: {trade.reason}")

    # Final Summary Report
    print("\n" + "=" * 80)
    print("                     PAPER TRADING PERFORMANCE TEARSHEET                      ".center(80))
    print("=" * 80)

    last_price = float(prices.iloc[-1])
    perf = portfolio.get_performance_metrics(last_price)
    trades_df = portfolio.get_trade_history_df()

    print(f"Initial Starting Cash : ${perf['initial_cash']:,.2f}")
    print(f"Ending Portfolio Value: ${perf['ending_value']:,.2f}")
    print(f"Net Realized/Unr PnL  : ${perf['total_pnl']:+,.2f} ({perf['total_return_pct']:+.2f}%)")
    print(f"Maximum Drawdown      : {perf['max_drawdown_pct']:.2f}%")
    print(f"Total Trades Executed : {perf['total_trades']}")
    print(f"Ending Inventory State: {perf['position_state']} (Cash: ${perf['current_cash']:,.2f}, Units: {perf['units']:,.6f})")

    print("\n" + "-" * 80)
    print("                              EXECUTED TRANSACTIONS LEDGER                    ".center(80))
    print("-" * 80)
    if not trades_df.empty:
        print(trades_df.to_string(index=False))
    else:
        print("No transactions executed during this simulation period (Conditions not triggered).")

    print("\n" + "=" * 80 + "\n")
    return portfolio, trades_df


if __name__ == "__main__":
    import sys

    asset = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    simulate_streaming_execution(symbol=asset, simulation_steps=35)
