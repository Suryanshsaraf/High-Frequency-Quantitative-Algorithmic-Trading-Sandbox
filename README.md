# High-Frequency Quantitative Algorithmic Trading Sandbox (ATSA)

[![Pair-Programmed with Antigravity](https://img.shields.io/badge/Pair--Programmed%20with-Antigravity%20IDE-4285F4?logo=google&logoColor=white)](https://github.com/Suryanshsaraf/High-Frequency-Quantitative-Algorithmic-Trading-Sandbox)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fully automated, event-driven quantitative trading sandbox that ingests real-time financial market data, identifies structural regime shifts, calculates institutional risk metrics, and simulates high-frequency paper trading.

---

## Architecture & Modules

```mermaid
flowchart LR
    A[Real-Time Ingestion] --> B[Microstructure & Regime Detection]
    B --> C[Statistical Alpha & Time Series]
    C --> D[Risk & Position Sizing]
    D --> E[LOB Matching Engine & Paper Execution]
```

### 1. Applied Time Series Modeling Engine (`modeling_engine.py`)
A modular statistical modeling pipeline executing sequential time series econometrics on live financial data:

1. **Data Ingestion**: Pulls 60 days of hourly (`1h`) asset prices from Yahoo Finance (`yfinance`).
2. **Log Return Transformation**: Stationarizes raw prices into continuously compounded log returns:
   $$r_t = \ln(P_t / P_{t-1})$$
3. **Unit Root Testing (ADF)**: Evaluates covariance stationarity with automated Augmented Dickey-Fuller tests and AIC lag selection.
4. **Conditional Mean Modeling (ARIMA)**: Fits $\text{ARIMA}(1, 0, 1)$ to capture autocorrelation and conditional mean structure:
   $$r_t = c + \phi_1 r_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t$$
5. **Conditional Volatility Modeling (GARCH)**: Extracts mean model residuals $e_t$ and fits a $\text{GARCH}(1, 1)$ process to capture volatility clustering and conditional heteroskedasticity:
   $$\sigma_t^2 = \omega + \alpha_1 e_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$
6. **Multi-Period Forecasting**: Projects the next 5 periods of expected mean return and conditional volatility.
7. **Diagnostic Verification (Ljung-Box)**: Evaluates standardized residuals $\eta_t = e_t / \sigma_t$ for white noise behavior at multiple lag checkpoints ($L = 10, 20$).

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Suryanshsaraf/High-Frequency-Quantitative-Algorithmic-Trading-Sandbox.git
cd High-Frequency-Quantitative-Algorithmic-Trading-Sandbox

# Create virtual environment & install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Time Series Modeling Engine
```bash
# Run on default volatile asset (BTC-USD)
python3 modeling_engine.py

# Run on custom equity or crypto asset (e.g., TSLA, ETH-USD)
python3 modeling_engine.py TSLA
```

### Programmatic Usage
```python
from modeling_engine import run_pipeline

results = run_pipeline(symbol="BTC-USD", period="60d", interval="1h", forecast_steps=5)

# Access artifacts
returns = results["returns"]
adf_stats = results["adf_test"]
arima_model = results["arima_model"]
garch_model = results["garch_model"]
forecast = results["forecast"]
```

---

## Author
- [Suryansh Saraf](https://github.com/Suryanshsaraf)

