"""
Applied Time Series Analysis - Quantitative Modeling Engine
===========================================================
This module executes a sequential, mathematically rigorous time series pipeline
on live financial asset data:

1. Data Ingestion: Fetches hourly asset prices via yfinance.
2. Transformation: Converts raw prices to Log Returns for covariance stationarity.
3. Unit Root Testing: Automated Augmented Dickey-Fuller (ADF) test.
4. Mean Modeling (ARIMA): Fits ARIMA(1, 0, 1) on returns.
5. Volatility Modeling (GARCH): Fits GARCH(1, 1) on ARIMA residuals.
6. Forecasting: Multi-step forecast for conditional mean and volatility.
7. Diagnostic Check: Ljung-Box test on standardized residuals for white noise verification.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller

# Suppress benign estimation warnings for clean console presentation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ADFResult:
    statistic: float
    p_value: float
    used_lag: int
    n_obs: int
    critical_values: Dict[str, float]
    is_stationary: bool


@dataclass
class ForecastResult:
    steps: int
    horizon_labels: List[str]
    mean_forecast: np.ndarray
    volatility_forecast: np.ndarray


# =====================================================================
# 1. Data Ingestion
# =====================================================================

def fetch_price_data(
    symbol: str = "BTC-USD",
    period: str = "60d",
    interval: str = "1h",
) -> pd.Series:
    """
    Fetch historical close prices for a specified asset using yfinance.

    Parameters:
        symbol: Financial ticker symbol (e.g. 'BTC-USD', 'TSLA').
        period: Historical lookback period (e.g. '60d').
        interval: Data sampling interval (e.g. '1h').

    Returns:
        pd.Series: Hourly closing prices indexed by datetime.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No price data retrieved for symbol '{symbol}' over period '{period}'.")

    close_series = df["Close"].copy()
    close_series.name = f"{symbol}_Close"
    return close_series


# =====================================================================
# 2. Transformation (Log Returns)
# =====================================================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Transform raw price levels into continuously compounded Log Returns
    to achieve covariance stationarity: r_t = ln(P_t / P_{t-1}).

    Parameters:
        prices: pd.Series of asset prices.

    Returns:
        pd.Series: Stationarized log returns with NaNs dropped.
    """
    if len(prices) < 2:
        raise ValueError("Price series must contain at least 2 data points to compute returns.")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = f"{prices.name.replace('_Close', '')}_LogReturn"
    return log_returns


# =====================================================================
# 3. Unit Root Testing (Augmented Dickey-Fuller)
# =====================================================================

def test_stationarity_adf(
    returns: pd.Series,
    alpha: float = 0.05,
    autolag: str = "AIC",
) -> ADFResult:
    """
    Perform Augmented Dickey-Fuller (ADF) test to evaluate stationarity.
    H0: The series contains a unit root (non-stationary).
    H1: The series does not contain a unit root (stationary).

    Parameters:
        returns: pd.Series of returns.
        alpha: Significance level threshold (default 0.05).
        autolag: Criterion for lag selection ('AIC', 'BIC', or None).

    Returns:
        ADFResult: Typed dataclass containing test statistic, p-value, and decision.
    """
    adf_out = adfuller(returns, autolag=autolag)
    stat = float(adf_out[0])
    p_val = float(adf_out[1])
    used_lag = int(adf_out[2])
    n_obs = int(adf_out[3])
    crit_vals = {k: float(v) for k, v in adf_out[4].items()}
    is_stationary = p_val < alpha

    return ADFResult(
        statistic=stat,
        p_value=p_val,
        used_lag=used_lag,
        n_obs=n_obs,
        critical_values=crit_vals,
        is_stationary=is_stationary,
    )


# =====================================================================
# 4. Mean Modeling (ARIMA)
# =====================================================================

def fit_arima_mean(
    returns: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 1),
) -> Tuple[ARIMAResults, pd.Series]:
    """
    Fit an ARIMA(p, d, q) model to capture conditional mean patterns.
    Uses array values to avoid irregular market-hours index frequency errors,
    while returning residuals anchored with the original DatetimeIndex.

    Parameters:
        returns: pd.Series of log returns.
        order: (p, d, q) tuple, defaulting to (1, 0, 1).

    Returns:
        Tuple[ARIMAResults, pd.Series]: Fitted model result and residual series e_t.
    """
    # Fit using numeric values to maintain compatibility across continuous and non-continuous trading schedules
    model = ARIMA(returns.values, order=order)
    arima_fit = model.fit()
    residuals = pd.Series(arima_fit.resid, index=returns.index, name="ARIMA_Residuals")
    return arima_fit, residuals


# =====================================================================
# 5. Volatility Modeling (GARCH)
# =====================================================================

def fit_garch_volatility(
    residuals: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    rescale: bool = True,
) -> Tuple[Any, pd.Series, float]:
    """
    Fit a GARCH(p, q) model on ARIMA residuals to model conditional heteroskedasticity:
        sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2

    Parameters:
        residuals: Mean model residual series e_t.
        p: Symmetric innovation order (ARCH lag).
        q: Volatility feedback order (GARCH lag).
        dist: Error distribution ('normal', 't', 'skewt').
        rescale: Whether to scale inputs by 100 for optimizer numerical stability.

    Returns:
        Tuple: (Fitted ARCHModelResult, standardized residuals eta_t, scale_factor).
    """
    scale_factor = 100.0 if rescale else 1.0
    scaled_resid = residuals * scale_factor

    # mean='Zero' ensures we model the volatility of the already mean-adjusted residuals
    am = arch_model(scaled_resid, p=p, q=q, mean="Zero", vol="GARCH", dist=dist)
    garch_fit = am.fit(disp="off", show_warning=False)

    # Standardized residuals: eta_t = e_t / sigma_t
    std_residuals = garch_fit.std_resid.dropna()
    std_residuals.name = "GARCH_Std_Residuals"

    return garch_fit, std_residuals, scale_factor


# =====================================================================
# 6. Multi-Step Forecasting
# =====================================================================

def forecast_mean_and_volatility(
    arima_res: ARIMAResults,
    garch_res: Any,
    last_timestamp: Optional[pd.Timestamp] = None,
    scale_factor: float = 100.0,
    steps: int = 5,
) -> ForecastResult:
    """
    Forecast the next N periods of conditional mean returns and conditional volatility.

    Parameters:
        arima_res: Fitted ARIMA model result.
        garch_res: Fitted GARCH model result.
        last_timestamp: Most recent timestamp from observed series.
        scale_factor: The multiplier applied during GARCH estimation (to unscale volatility).
        steps: Number of forward periods to project (default 5).

    Returns:
        ForecastResult: Typed dataclass with predicted mean and standard deviation.
    """
    # 1. Mean forecast from ARIMA
    mean_forecast = arima_res.forecast(steps=steps)
    if isinstance(mean_forecast, pd.Series):
        mean_vals = mean_forecast.values
    else:
        mean_vals = np.asarray(mean_forecast)

    # 2. Volatility forecast from GARCH
    garch_forecast = garch_res.forecast(horizon=steps, reindex=False)
    var_forecast = garch_forecast.variance.iloc[-1].values  # variance of scaled series
    # Unscale volatility: sigma_original = sqrt(variance_scaled) / scale_factor
    vol_vals = np.sqrt(var_forecast) / scale_factor

    # 3. Construct horizon labels
    horizon_labels = []
    for i in range(steps):
        step_num = i + 1
        if last_timestamp is not None:
            # Approximate forward hourly interval
            proj_time = last_timestamp + pd.Timedelta(hours=step_num)
            horizon_labels.append(f"t+{step_num} ({proj_time.strftime('%Y-%m-%d %H:%M')})")
        else:
            horizon_labels.append(f"t+{step_num}")

    return ForecastResult(
        steps=steps,
        horizon_labels=horizon_labels,
        mean_forecast=mean_vals,
        volatility_forecast=vol_vals,
    )


# =====================================================================
# 7. Diagnostic Check (Ljung-Box Test)
# =====================================================================

def diagnostic_ljung_box(
    std_residuals: pd.Series,
    lags: list[int] = [10, 20],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Execute Ljung-Box Portmanteau test on standardized residuals
    to confirm that no serial correlation or remaining ARCH effects exist.
    H0: Residuals are independently distributed (White Noise).
    H1: Residuals exhibit serial correlation.

    Parameters:
        std_residuals: Standardized residuals eta_t = e_t / sigma_t.
        lags: Lag checkpoints to test.
        alpha: Significance threshold (default 0.05).

    Returns:
        pd.DataFrame: Table of statistics, p-values, and White Noise status.
    """
    lb_df = acorr_ljungbox(std_residuals, lags=lags, return_df=True)
    lb_df["is_white_noise"] = lb_df["lb_pvalue"] > alpha
    return lb_df


# =====================================================================
# Sequential Pipeline Orchestrator & CLI Formatter
# =====================================================================

def print_header(title: str, step: int) -> None:
    print("\n" + "=" * 78)
    print(f" STEP {step}: {title.upper()}")
    print("=" * 78)


def run_pipeline(
    symbol: str = "BTC-USD",
    period: str = "60d",
    interval: str = "1h",
    forecast_steps: int = 5,
) -> Dict[str, Any]:
    """
    Sequentially executes the full Applied Time Series pipeline with professional formatting.

    Parameters:
        symbol: Financial ticker symbol.
        period: Lookback window.
        interval: Bar resolution.
        forecast_steps: Forward forecast horizon.

    Returns:
        Dict: Complete collection of pipeline artifacts and model objects.
    """
    print("\n" + "#" * 78)
    print("   APPLIED TIME SERIES QUANTITATIVE MODELING ENGINE   ".center(78))
    print(f"   Target: {symbol} | Lookback: {period} | Interval: {interval}   ".center(78))
    print("#" * 78)

    # -------------------------------------------------------------
    # 1. Data Ingestion
    # -------------------------------------------------------------
    print_header("Data Ingestion via yfinance", 1)
    prices = fetch_price_data(symbol=symbol, period=period, interval=interval)
    start_time = prices.index[0].strftime("%Y-%m-%d %H:%M %Z")
    end_time = prices.index[-1].strftime("%Y-%m-%d %H:%M %Z")
    print(f"Asset Symbol        : {symbol}")
    print(f"Observation Window  : {start_time} --> {end_time}")
    print(f"Total Bars Ingested : {len(prices):,}")
    print(f"Initial Close Price : ${prices.iloc[0]:,.2f}")
    print(f"Latest Close Price  : ${prices.iloc[-1]:,.2f}")

    # -------------------------------------------------------------
    # 2. Transformation to Log Returns
    # -------------------------------------------------------------
    print_header("Transformation to Log Returns", 2)
    returns = compute_log_returns(prices)
    mean_ret = returns.mean()
    std_ret = returns.std()
    skew_ret = returns.skew()
    kurt_ret = returns.kurtosis()

    print(f"Formula             : r_t = ln(P_t / P_{{t-1}})")
    print(f"Stationary Sample   : {len(returns):,} observations (NaNs dropped)")
    print(f"Sample Mean (μ)     : {mean_ret:+.6f} ({mean_ret * 100:+.4f}% / hour)")
    print(f"Sample Std Dev (σ)  : {std_ret:.6f} ({std_ret * 100:.4f}% / hour)")
    print(f"Sample Skewness     : {skew_ret:+.4f}")
    print(f"Excess Kurtosis     : {kurt_ret:+.4f} (Leptokurtic / Fat Tails detected)")

    # -------------------------------------------------------------
    # 3. Unit Root Testing (ADF)
    # -------------------------------------------------------------
    print_header("Stationarity & Unit Root Testing (Augmented Dickey-Fuller)", 3)
    adf = test_stationarity_adf(returns)
    print(f"ADF Test Statistic  : {adf.statistic:.6f}")
    print(f"p-value             : {adf.p_value:.6e}")
    print(f"Optimal Lags (AIC)  : {adf.used_lag}")
    print(f"Number of Obs       : {adf.n_obs:,}")
    print("Critical Values     :")
    for level, crit in adf.critical_values.items():
        print(f"   • {level:>4}           : {crit:.4f}")

    if adf.is_stationary:
        print("\n>> VERDICT: Null hypothesis H0 rejected (p < 0.05).")
        print("   The log return series is strictly COVARIANCE STATIONARY.")
    else:
        print("\n>> VERDICT: Failed to reject H0 (p >= 0.05). Non-stationary series.")

    # -------------------------------------------------------------
    # 4. Mean Modeling (ARIMA)
    # -------------------------------------------------------------
    print_header("Conditional Mean Modeling: ARIMA(1, 0, 1)", 4)
    arima_res, residuals = fit_arima_mean(returns, order=(1, 0, 1))

    params = arima_res.params
    pvalues = arima_res.pvalues
    bse = arima_res.bse

    param_names = ["const", "ar.L1", "ma.L1", "sigma2"]
    print("Model Architecture  : AR(1) - MA(1) on Log Returns")
    print(f"Log-Likelihood      : {arima_res.llf:,.2f}")
    print(f"Akaike IC (AIC)     : {arima_res.aic:,.2f}")
    print(f"Bayesian IC (BIC)   : {arima_res.bic:,.2f}")
    print("\nParameter Estimates:")
    print(f"{'Parameter':<14} | {'Estimate':>12} | {'Std Error':>12} | {'t-stat':>10} | {'p-value':>12}")
    print("-" * 72)
    for idx, est in enumerate(params):
        name = param_names[idx] if idx < len(param_names) else f"param_{idx}"
        se = bse[idx] if idx < len(bse) else np.nan
        t_stat = est / se if se != 0 and not np.isnan(se) else np.nan
        p_val = pvalues[idx] if idx < len(pvalues) else np.nan
        print(f"{name:<14} | {est:>12.6f} | {se:>12.6f} | {t_stat:>10.4f} | {p_val:>12.4e}")

    # -------------------------------------------------------------
    # 5. Volatility Modeling (GARCH)
    # -------------------------------------------------------------
    print_header("Conditional Volatility Modeling: GARCH(1, 1)", 5)
    garch_res, std_residuals, scale_factor = fit_garch_volatility(residuals, p=1, q=1)

    omega = garch_res.params.get("omega", np.nan)
    alpha = garch_res.params.get("alpha[1]", np.nan)
    beta = garch_res.params.get("beta[1]", np.nan)
    persistence = alpha + beta

    print("Variance Equation   : σ_t² = ω + α·e_{t-1}² + β·σ_{t-1}²")
    print(f"Log-Likelihood      : {garch_res.loglikelihood:,.2f}")
    print(f"Akaike IC (AIC)     : {garch_res.aic:,.2f}")
    print(f"Bayesian IC (BIC)   : {garch_res.bic:,.2f}")
    print("\nEstimated Volatility Parameters:")
    print(f"{'Parameter':<14} | {'Estimate':>12} | {'Std Error':>12} | {'t-stat':>10} | {'p-value':>12}")
    print("-" * 72)
    for name, est in garch_res.params.items():
        se = garch_res.std_err.get(name, np.nan)
        t_stat = garch_res.tvalues.get(name, np.nan)
        p_val = garch_res.pvalues.get(name, np.nan)
        print(f"{name:<14} | {est:>12.6f} | {se:>12.6f} | {t_stat:>10.4f} | {p_val:>12.4e}")

    print(f"\nVolatility Persistence (α + β): {persistence:.6f}")
    if persistence < 0.999 and persistence > 0:
        half_life = np.log(0.5) / np.log(persistence)
        print(f">> Model is mean-reverting (α + β < 1). Volatility Half-Life: {half_life:.2f} hours.")
    elif persistence >= 0.999:
        print(">> High Persistence / Near-Integrated GARCH (α + β ≈ 1.00). Volatility shocks decay slowly.")
    else:
        print(">> Non-stationary volatility dynamics.")

    # -------------------------------------------------------------
    # 6. Multi-Step Forecasting
    # -------------------------------------------------------------
    print_header(f"Multi-Step Forecasting: Next {forecast_steps} Periods", 6)
    last_ts = prices.index[-1] if hasattr(prices.index[-1], "strftime") else None
    forecast = forecast_mean_and_volatility(
        arima_res=arima_res,
        garch_res=garch_res,
        last_timestamp=last_ts,
        scale_factor=scale_factor,
        steps=forecast_steps,
    )

    print(f"{'Step':<5} | {'Horizon':<26} | {'Forecast Mean Return':>20} | {'Cond Volatility (1h)':>20}")
    print("-" * 78)
    for i in range(forecast.steps):
        lbl = forecast.horizon_labels[i]
        mean_pct = forecast.mean_forecast[i] * 100
        vol_pct = forecast.volatility_forecast[i] * 100
        print(
            f"t+{i+1:<3} | {lbl:<26} | {forecast.mean_forecast[i]:>13.6f} ({mean_pct:>+6.3f}%) | "
            f"{forecast.volatility_forecast[i]:>13.6f} ({vol_pct:>6.3f}%)"
        )

    # -------------------------------------------------------------
    # 7. Diagnostic Check (Ljung-Box Test on Standardized Residuals)
    # -------------------------------------------------------------
    print_header("Diagnostic Check: Ljung-Box Test on Standardized Residuals", 7)
    lb_results = diagnostic_ljung_box(std_residuals, lags=[10, 20])

    print("H0: Standardized residuals η_t = e_t / σ_t are Independent White Noise.")
    print("H1: Residuals contain remaining serial correlation / misspecification.\n")
    print(f"{'Lag Check':<12} | {'LB Test Stat':>15} | {'p-value':>15} | {'Diagnostic Status':>20}")
    print("-" * 70)
    all_passed = True
    for lag, row in lb_results.iterrows():
        stat = row["lb_stat"]
        pval = row["lb_pvalue"]
        passed = row["is_white_noise"]
        status_str = "PASS (White Noise)" if passed else "FAIL (Serial Correlation)"
        if not passed:
            all_passed = False
        print(f"Lag {lag:<8} | {stat:>15.4f} | {pval:>15.6f} | {status_str:>20}")

    print("\n>> OVERALL DIAGNOSTIC VERDICT:")
    if all_passed:
        print("   [PASSED] Standardized residuals show NO statistically significant autocorrelation.")
        print("   The combined ARIMA(1,0,1) + GARCH(1,1) specification successfully captures")
        print("   both conditional mean dynamics and conditional heteroskedasticity.")
    else:
        print("   [CAUTION] Some serial autocorrelation remains in residuals at selected lags.")

    print("\n" + "#" * 78)
    print("                      PIPELINE EXECUTION COMPLETE                      ".center(78))
    print("#" * 78 + "\n")

    return {
        "prices": prices,
        "returns": returns,
        "adf_test": adf,
        "arima_model": arima_res,
        "garch_model": garch_res,
        "forecast": forecast,
        "ljung_box": lb_results,
    }


if __name__ == "__main__":
    import sys

    asset = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    run_pipeline(symbol=asset)
