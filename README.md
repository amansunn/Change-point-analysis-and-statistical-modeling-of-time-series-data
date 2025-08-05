# Your Project Name

Add your project description here.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

## Key Assumptions & Limitations

### Causality vs Correlation
- Identified temporal associations don't imply causation
- External factors not in model may influence prices

### Data Limitations
- Daily prices might miss intraday volatility
- Event impact periods defined as ±5 trading days

### Model Assumptions
- Stationarity assumption for ARIMA components
- Normal distribution of residuals

### Timeframe
- Analysis covers 2014-2023 period
- Pre-2014 data excluded due to different market dynamics
