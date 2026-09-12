"""
Unit tests for the Quantitative Trading & Risk Execution Layer (`execution_logic.py`)
"""

import numpy as np
import pandas as pd

from execution_logic import (
    MockPortfolioTracker,
    PositionState,
    SignalType,
    calculate_risk_threshold,
    evaluate_execution_signal,
)


def test_portfolio_initialization():
    port = MockPortfolioTracker(initial_cash=10000.0)
    assert port.cash == 10000.0
    assert port.units == 0.0
    assert port.position_state == PositionState.CASH
    assert len(port.trade_log) == 0
    assert port.get_portfolio_value(current_price=50000.0) == 10000.0


def test_portfolio_buy_execution():
    port = MockPortfolioTracker(initial_cash=10000.0)
    trade = port.execute_buy(
        timestamp="2026-09-12 10:00:00",
        price=50000.0,
        reason="Low Volatility Regime",
    )
    assert trade is not None
    assert trade.signal == SignalType.BUY.value
    assert trade.price == 50000.0
    assert trade.order_size == 0.2  # 10,000 / 50,000
    assert port.cash == 0.0
    assert port.units == 0.2
    assert port.position_state == PositionState.LONG
    assert port.get_portfolio_value(current_price=50000.0) == 10000.0
    assert port.get_portfolio_value(current_price=60000.0) == 12000.0


def test_portfolio_sell_execution():
    port = MockPortfolioTracker(initial_cash=10000.0)
    port.execute_buy(timestamp="2026-09-12 10:00:00", price=50000.0)
    
    # Sell when price appreciated to 55,000
    trade = port.execute_sell(
        timestamp="2026-09-12 11:00:00",
        price=55000.0,
        reason="Volatility Spike Exit",
    )
    assert trade is not None
    assert trade.signal == SignalType.SELL.value
    assert trade.price == 55000.0
    assert trade.order_size == 0.2
    assert port.cash == 11000.0  # 0.2 * 55,000
    assert port.units == 0.0
    assert port.position_state == PositionState.CASH
    assert port.get_portfolio_value(current_price=55000.0) == 11000.0

    # Verify transaction ledger
    df = port.get_trade_history_df()
    assert len(df) == 2
    assert list(df["Signal Type"]) == ["BUY", "SELL"]
    assert list(df["Price"]) == [50000.0, 55000.0]


def test_duplicate_order_guards():
    port = MockPortfolioTracker(initial_cash=10000.0)
    
    # Cannot sell when in CASH
    trade_sell = port.execute_sell(timestamp="2026-09-12 10:00", price=50000.0)
    assert trade_sell is None
    
    # Buy succeeds
    port.execute_buy(timestamp="2026-09-12 10:00", price=50000.0)
    
    # Cannot buy again when already LONG
    trade_buy2 = port.execute_buy(timestamp="2026-09-12 11:00", price=51000.0)
    assert trade_buy2 is None


def test_risk_threshold_calculation():
    # Symmetric returns
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.05, -0.05])
    threshold = calculate_risk_threshold(returns, percentile=90.0)
    # 90th percentile of [0.01, 0.01, 0.02, 0.02, 0.05, 0.05]
    expected = np.percentile([0.01, 0.01, 0.02, 0.02, 0.05, 0.05], 90.0)
    assert np.isclose(threshold, expected)


def test_evaluate_execution_signal():
    threshold = 0.005

    # Case 1: In CASH and Vol <= Threshold -> BUY
    sig, _ = evaluate_execution_signal(
        current_position=PositionState.CASH,
        volatility_forecast=np.array([0.003, 0.003, 0.004]),
        risk_threshold=threshold,
        current_price=100.0,
    )
    assert sig == SignalType.BUY

    # Case 2: In CASH and Vol > Threshold -> HOLD (Wait in Cash)
    sig, _ = evaluate_execution_signal(
        current_position=PositionState.CASH,
        volatility_forecast=np.array([0.008, 0.007, 0.009]),
        risk_threshold=threshold,
        current_price=100.0,
    )
    assert sig == SignalType.HOLD

    # Case 3: In LONG and Vol <= Threshold -> HOLD (Stay Long)
    sig, _ = evaluate_execution_signal(
        current_position=PositionState.LONG,
        volatility_forecast=np.array([0.004, 0.004, 0.004]),
        risk_threshold=threshold,
        current_price=100.0,
    )
    assert sig == SignalType.HOLD

    # Case 4: In LONG and Vol > Threshold -> SELL (Capital Preservation Exit)
    sig, _ = evaluate_execution_signal(
        current_position=PositionState.LONG,
        volatility_forecast=np.array([0.009, 0.008, 0.010]),
        risk_threshold=threshold,
        current_price=100.0,
    )
    assert sig == SignalType.SELL
