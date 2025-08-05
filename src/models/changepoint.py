import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

def load_oil_prices(file_path):
    """Load and preprocess Brent crude oil prices"""
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df['LogReturn'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
    return df.dropna()

def bayesian_changepoint_model(returns_series, n_changepoints=2):
    """Bayesian changepoint detection with volatility regimes
    
    Args:
        returns_series: Array of log returns
        n_changepoints: Maximum number of change points to detect
        
    Returns:
        model: PyMC3 model
        trace: InferenceData with sampling results
    """
    with pm.Model() as model:
        # Prior for number of changepoints (Poisson distribution)
        n_cp = pm.Poisson('n_cp', mu=n_changepoints, testval=n_changepoints)
        
        # Uniform prior on changepoint positions
        cp = pm.Uniform('cp', 0, len(returns_series),
                      shape=n_changepoints, transform=None)
        
        # Sort changepoints chronologically
        cp_sorted = pm.Deterministic('cp_sorted', pm.math.sort(cp))
        
        # Segment parameters (mean and volatility)
        mu = pm.Normal('mu', mu=0, sigma=0.2, shape=n_changepoints+1)
        sigma = pm.HalfNormal('sigma', sigma=0.1, shape=n_changepoints+1)
        
        # Construct regime switches
        regime_idx = np.zeros(len(returns_series))
        for i in range(n_changepoints):
            regime_idx = pm.math.switch(cp_sorted[i] <= np.arange(len(returns_series)),
                                      i+1, regime_idx)
            
        # Vectorized likelihood
        likelihood = pm.Normal('returns',
                             mu=mu[regime_idx.astype(int)],
                             sigma=sigma[regime_idx.astype(int)],
                             observed=returns_series)
        
        # Improved sampling with NUTS
        trace = pm.sample(2000, tune=2000, cores=4, target_accept=0.9,
                        return_inferencedata=True)
        
    return model, trace

def visualize_changepoints(prices_df, trace, events_df=None):
    """Visualize detected changepoints with regime statistics and events"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot price series with changepoints
    ax1.plot(prices_df['Date'], prices_df['Price'], label='Brent Price')
    
    # Get changepoints from trace
    cp_samples = trace.posterior['cp_sorted'].values.reshape(-1, trace.posterior.dims['cp_sorted_dim_0'])
    cp_dates = [prices_df['Date'].iloc[int(np.median(cp))] for cp in cp_samples.T]
    
    # Calculate regime statistics
    regimes = []
    prev_idx = 0
    for cp in np.median(cp_samples, axis=0):
        regime_data = prices_df.iloc[prev_idx:int(cp)]
        regimes.append({
            'start': regime_data['Date'].iloc[0],
            'end': regime_data['Date'].iloc[-1],
            'mean_price': regime_data['Price'].mean(),
            'volatility': regime_data['LogReturn'].std()
        })
        prev_idx = int(cp)
    
    # Add regime annotations
    for i, regime in enumerate(regimes):
        ax1.text(regime['start'], ax1.get_ylim()[0],
                f"Regime {i+1}\n${regime['mean_price']:.2f}±{regime['volatility']:.2f}",
                ha='left', va='bottom', backgroundcolor='white')
    
    # Plot volatility with regime means
    vol = prices_df['LogReturn'].rolling(5).std()
    ax2.plot(prices_df['Date'], vol, label='Volatility')
    ax2.axhline(vol.mean(), color='grey', linestyle='--', label='Overall Avg')
    
    # Add events if provided
    if events_df is not None:
        events_df['Date'] = pd.to_datetime(events_df['Date'])
        for _, event in events_df.iterrows():
            ax1.annotate(event['Event'], xy=(event['Date'], ax1.get_ylim()[1]),
                        xytext=(-20, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", color='black'),
                        rotation=90, ha='right', va='top',
                        backgroundcolor='white')
    
    ax1.set_title('Brent Oil Prices with Detected Regime Changes')
    ax2.set_title('Volatility Regimes with Statistical Significance')
    ax2.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax2.set_ylabel('Daily Volatility')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data
    prices = load_oil_prices('data/raw/BrentOilPrices.csv')
    events = pd.read_csv('data/raw/events.csv')
    
    # Run analysis with 3 potential change points
    model, trace = bayesian_changepoint_model(prices['LogReturn'].values, n_changepoints=3)
    
    # Generate statistical report
    print("Change Point Analysis Report")
    print("============================")
    print(az.summary(trace, var_names=["cp_sorted", "mu", "sigma"]))
    
    # Visualization with events
    visualize_changepoints(prices, trace, events_df=events)
    
    # Save results
    trace.to_netcdf("results/change_point_trace.nc")