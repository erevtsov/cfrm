#!/usr/bin/env python3

"""
Generates tables:
  results/ols_k5.csv
  results/ols_k5_hac.csv
  results/nls_k5.csv
  results/beta_vs_k.csv
  results/beta_vs_k_<TICKER>.csv
  results/expected_drag_by_regime.csv
  results/drag_forecast_models.csv
  results/drag_forecast_models.tex
  results/panel_with_ms.csv
  results/oos_drag_forecast.csv
  results/oos_drag_summary.csv
  results/oos_drag_summary.tex

Generates figures (curated):
  figures/predictor_distributions.(png|pdf)
  figures/spy_short_energy_regime_annotated.(png|pdf)
  figures/beta_vs_k.(png|pdf)
  figures/{TICKER}/rolling_beta_{TICKER}.(png|pdf)
  figures/{TICKER}/drag_timeseries_{TICKER}_k5.(png|pdf)
  figures/{TICKER}/drag_heatmap_{TICKER}.png
  figures/UPRO/drag_vs_vixslope_UPRO.(png|pdf)
  figures/drag_forecast_sensitivity.(png|pdf)
  figures/drag_forecast_sensitivity_energy.(png|pdf)
  figures/how_predictable_is_drag_in_sample.(png|pdf)
  figures/UPRO/pred_vs_real_drag_UPRO_k5.(png|pdf)
  figures/oos_R2_vs_k.(png|pdf)
  figures/oos_hitrate_vs_k.(png|pdf)
  figures/UPRO/calibration_UPRO_k5.(png|pdf)
"""

import os
import time
import warnings
from pathlib import Path
from typing import Optional  # noqa: F401

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------- Config ----------------------------

START = '2010-01-01'
END = '2025-12-31'

UNDERLYING = 'SPY'
LETF_LIST = ['SSO', 'UPRO', 'SDS', 'SPXU']
ANCHOR = 'UPRO'

VIX_TICKERS = ['^VIX', '^VIX3M']
VIX_SYNONYMS = {'^VIX': ['^VIX'], '^VIX3M': ['^VIX3M', '^VXMT']}

R_ANNUAL = 0.05
F_MAP = {'SSO': 0.009, 'UPRO': 0.009, 'SDS': 0.009, 'SPXU': 0.009}
K_LIST = (1, 5, 10, 22, 66)

os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ------------------------- Figure saving ------------------------


def safesave(fname: str, *args, **kwargs) -> None:
    """
    Save a matplotlib figure to a file, creating parent directories if needed.

    Args:
        fname: File path where the figure will be saved
        *args: Additional positional arguments passed to plt.savefig()
        **kwargs: Additional keyword arguments passed to plt.savefig()
    """
    p = Path(fname)
    if str(p.parent) != '.':
        p.parent.mkdir(parents=True, exist_ok=True)
    return plt.savefig(str(p), *args, **kwargs)


# ------------------------- Helpers: IO --------------------------


def _extract_adj_close(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """
    Extract adjusted close price from a DataFrame with different possible column naming conventions.

    Args:
        df: DataFrame containing price data
        ticker: Ticker symbol for the security

    Returns:
        pd.Series: Series of adjusted close prices, or None if not found
    """
    if df is None or df.empty:
        return None
    for col in ('Adj Close', 'Close'):
        if col in df.columns:
            ser = df[col].dropna()
            if not ser.empty:
                return ser
    if isinstance(df.columns, pd.MultiIndex):
        for col0 in ('Adj Close', 'Close'):
            key = (col0, ticker)
            if key in df.columns:
                ser = df[key].dropna()
                if not ser.empty:
                    return ser
            for c in df.columns:
                if isinstance(c, tuple) and c[0] == col0 and ticker.upper() in str(c[1]).upper():
                    ser = df[c].dropna()
                    if not ser.empty:
                        return ser
    return None


def _fred_fetch(series: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch economic data from FRED (Federal Reserve Economic Data) API.

    Args:
        series: FRED series ID (e.g., 'VIXCLS' for VIX)
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame with 'Date' index and 'Adj Close' column, or None if fetch fails
    """
    try:
        from pandas_datareader import data as pdr

        df = pdr.DataReader(series, 'fred', start=start, end=end)
        if isinstance(df, pd.DataFrame) and series in df.columns:
            out = df.rename(columns={series: 'Adj Close'}).dropna()
            out.index.name = 'Date'
            return out
    except Exception:
        return None
    return None


def download_series(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical price data for a given ticker from various sources.

    Tries multiple data sources in order:
    1. yfinance (Yahoo Finance)
    2. FRED (for VIX data)
    3. Stooq (via pandas-datareader)

    Args:
        ticker: Ticker symbol (e.g., 'SPY', '^VIX')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame with 'Date' index and 'Adj Close' column

    Raises:
        RuntimeError: If all data source attempts fail
    """
    symbols_to_try = VIX_SYNONYMS.get(ticker, [ticker])

    # yfinance first
    try:
        import yfinance as yf

        for sym in symbols_to_try:
            for i in range(5):
                try:
                    df = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False, threads=False)
                    ser = _extract_adj_close(df, sym)
                    if ser is not None and not ser.empty:
                        return (
                            pd.DataFrame({'Adj Close': ser})
                            if isinstance(ser, pd.Series)
                            else pd.DataFrame({'Adj Close': ser.squeeze()})
                        )
                except Exception as e:
                    raise e
                time.sleep(min(2.0**i, 8.0))
    except Exception:
        pass

    # FRED fallback for VIX
    fred_map = {'^VIX': 'VIXCLS', '^VIX3M': 'VIX3M'}
    if ticker in fred_map:
        fred_df = _fred_fetch(fred_map[ticker], start, end)
        if fred_df is not None and not fred_df.empty:
            return fred_df

    # Stooq via pandas-datareader (non-caret)
    if not ticker.startswith('^'):
        try:
            from pandas_datareader import data as pdr

            for sym in (ticker, f'{ticker}.US'):
                try:
                    tmp = pdr.DataReader(sym, 'stooq', start=start, end=end)
                    if isinstance(tmp, pd.DataFrame) and 'Close' in tmp and not tmp['Close'].dropna().empty:
                        ser = tmp['Close'].sort_index().dropna()
                        return pd.DataFrame({'Adj Close': ser})
                except Exception as e:
                    raise e
        except Exception as e:
            raise e

    raise RuntimeError(
        f'Failed to fetch {ticker}. '
        f'As a last resort, place CSV at data/{ticker.replace("^", "_caret_")}.csv with columns Date,Adj Close.'
    )


def ensure_prices(tickers: list[str], start: str = START, end: str = END) -> dict[str, pd.DataFrame]:
    """
    Ensure price data is available for given tickers, downloading if necessary.

    Checks local cache first, then downloads if needed. Handles multiple tickers
    and ensures no duplicate dates in the returned data.

    Args:
        tickers: List of ticker symbols
        start: Start date in 'YYYY-MM-DD' format (default: from config)
        end: End date in 'YYYY-MM-DD' format (default: from config)

    Returns:
        Dict mapping tickers to their respective price DataFrames with 'Date' index
        and 'Adj Close' column
    """
    out = {}
    for t in tickers:
        path = f'data/{t.replace("^", "_caret_")}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
        else:
            df = download_series(t, start, end)
            df.reset_index().rename(columns={'index': 'Date'}).to_csv(path, index=False)
        out[t] = df[~df.index.duplicated(keep='first')].sort_index()
    return out


def align_adj_close(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align multiple price series on their common dates using an inner join.

    Args:
        dfs: Dictionary mapping tickers to their price DataFrames

    Returns:
        pd.DataFrame: Single DataFrame with tickers as columns, aligned on common dates
        with all NaN values dropped
    """
    tmp = None
    for k, df in dfs.items():
        d = df[['Adj Close']].rename(columns={'Adj Close': k})
        tmp = d if tmp is None else tmp.join(d, how='inner')
    return tmp.dropna()


# --------------------- k-day features --------------------------


def realized_var_quiz_exact(simple_ret: pd.Series, k: int) -> pd.Series:
    """
    Calculate rolling realized variance over a k-day window.

    Implements the exact formula from the quiz:
    RV_t = Σ_{i=0}^{k-1} (r_{t-i} - μ_t)^2
    where μ_t is the mean return over the window.

    Args:
        simple_ret: Series of simple returns
        k: Window size for rolling calculation

    Returns:
        pd.Series: Rolling realized variance, shifted to align with the end of each window
    """
    m = simple_ret.rolling(k).mean()
    ss = ((simple_ret - m) ** 2).rolling(k).sum()
    return ss.shift(-(k - 1))


def build_xyv(df_prices: pd.DataFrame, S_col: str, L_col: str, k: int) -> pd.DataFrame:
    """
    Construct features and target variables for regression analysis.

    Creates:
    - x: k-day log return of S (predictor)
    - y: k-day log return of L (target)
    - v: k-day realized variance of S (predictor)

    Args:
        df_prices: DataFrame containing price data
        S_col: Column name for the short-term instrument (e.g., SPY)
        L_col: Column name for the leveraged ETF
        k: Lookahead period in days

    Returns:
        pd.DataFrame: DataFrame with columns 'x', 'y', 'v' and a datetime index
    """
    logS = np.log(df_prices[S_col])
    logL = np.log(df_prices[L_col])
    x = logS.shift(-k) - logS
    y = logL.shift(-k) - logL
    R_S = df_prices[S_col].pct_change()
    v = realized_var_quiz_exact(R_S, k)
    return pd.DataFrame({'x': x, 'y': y, 'v': v}).dropna()


# ------------------------- OLS / NLS ---------------------------


def ols_y_on_X(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    Perform ordinary least squares (OLS) regression of y on X.

    Args:
        y: 1D array of dependent variable observations
        X: 2D array of independent variables (should include constant if needed)

    Returns:
        tuple: (coefficients, stats) where:
            - coefficients: array of estimated coefficients
            - stats: dictionary containing:
                - n: number of observations
                - R2: coefficient of determination
                - SSE: sum of squared errors
    """
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    resid = y - yhat
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum(resid**2))
    r2 = 1.0 - (sse / max(sst, 1e-12))
    return b, {'n': int(len(y)), 'R2': float(r2), 'SSE': sse}


def fit_ols_hac(df_k: pd.DataFrame, lags: int = 4) -> dict[str, float | int]:
    """
    Fit OLS regression with HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors.

    The model specification is: y ~ 1 + x + v

    Args:
        df_k: DataFrame containing columns 'x', 'y', and 'v'
        lags: Number of lags to use in the HAC correction (default: 4)

    Returns:
        Dictionary containing:
        - Regression coefficients (alpha_OLS, beta_OLS, theta_OLS)
        - Standard errors (alpha_se, beta_se, theta_se)
        - t-statistics (alpha_t, beta_t, theta_t)
        - Model fit statistics (R2, SSE, n, hac_lags)
    """
    X = sm.add_constant(df_k[['x', 'v']].copy(), has_constant='add')
    y = df_k['y']
    model = sm.OLS(y, X)
    res = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return {
        'alpha_OLS': res.params.get('const', np.nan),
        'beta_OLS': res.params.get('x', np.nan),
        'theta_OLS': res.params.get('v', np.nan),
        'alpha_se': res.bse.get('const', np.nan),
        'beta_se': res.bse.get('x', np.nan),
        'theta_se': res.bse.get('v', np.nan),
        'alpha_t': res.tvalues.get('const', np.nan),
        'beta_t': res.tvalues.get('x', np.nan),
        'theta_t': res.tvalues.get('v', np.nan),
        'R2': res.rsquared,
        'SSE': float((res.resid**2).sum()),
        'n': len(df_k),
        'hac_lags': lags,
    }


def f_theory(beta: float, x: np.ndarray, v: np.ndarray, r: float, f: float, k: int) -> np.ndarray:
    """
    Theoretical model for the relationship between LETF returns and underlying returns.

    Implements the formula:
    f(beta, x, v) = beta*x - 0.5*beta*(beta-1)*v + ((1-beta)*r - f)*dT

    Where:
    - beta: Leverage factor
    - x: Log returns of the underlying
    - v: Realized variance of the underlying
    - r: Risk-free rate (annualized)
    - f: Expense ratio (annualized)
    - k: Holding period in days
    - dT: Time in years (k/252)

    Args:
        beta: Leverage factor
        x: Array of log returns of the underlying
        v: Array of realized variance of the underlying
        r: Annual risk-free rate
        f: Annual expense ratio
        k: Holding period in days

    Returns:
        Array of theoretical LETF log returns
    """
    dT = k / 252.0
    return beta * x - 0.5 * beta * (beta - 1.0) * v + ((1.0 - beta) * r - f) * dT


def sse_beta(beta: float, y: np.ndarray, x: np.ndarray, v: np.ndarray, r: float, f: float, k: int) -> float:
    """
    Calculate sum of squared errors between actual and theoretical LETF returns.

    This is a helper function used for NLS (Nonlinear Least Squares) optimization.

    Args:
        beta: Leverage factor to evaluate
        y: Array of actual LETF log returns
        x: Array of underlying log returns
        v: Array of realized variance of the underlying
        r: Annual risk-free rate
        f: Annual expense ratio
        k: Holding period in days

    Returns:
        Sum of squared errors between actual and theoretical returns
    """
    resid = y - f_theory(float(beta), x, v, r, f, k)
    return float(np.sum(resid * resid))


def nls_beta(
    y: np.ndarray, x: np.ndarray, v: np.ndarray, r: float, f: float, k: int, bracket: tuple[float, float] = (-5.0, 5.0)
) -> tuple[float, float]:
    """
    Estimate beta using Nonlinear Least Squares (NLS) with a bounded search.

    Tries to minimize the sum of squared errors between actual LETF returns
    and the theoretical model predictions.

    Args:
        y: Array of actual LETF log returns
        x: Array of underlying log returns
        v: Array of realized variance of the underlying
        r: Annual risk-free rate
        f: Annual expense ratio
        k: Holding period in days
        bracket: Tuple of (min, max) bounds for beta search

    Returns:
        Tuple of (optimal_beta, sse) where:
        - optimal_beta: Estimated leverage factor that minimizes SSE
        - sse: Sum of squared errors at optimal beta
    """
    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(
            lambda b: sse_beta(b, y, x, v, r, f, k), bounds=bracket, method='bounded', options={'xatol': 1e-8}
        )
        return float(res.x), float(res.fun)
    except Exception:
        grid = np.linspace(bracket[0], bracket[1], 2001)
        vals = [sse_beta(b, y, x, v, r, f, k) for b in grid]
        i = int(np.argmin(vals))
        return float(grid[i]), float(vals[i])


# ------------------------- Multiscale summary ------------------


def try_ceemdan_imfs(x: np.ndarray, max_imf: int | None = None, seed: int = 1337) -> np.ndarray | None:
    """
    Attempt to perform CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)
    decomposition on the input signal.

    CEEMDAN is an advanced signal processing technique that decomposes a signal into
    Intrinsic Mode Functions (IMFs) with different frequency components.

    Args:
        x: 1D input signal to decompose
        max_imf: Maximum number of IMFs to extract (None for automatic)
        seed: Random seed for reproducibility

    Returns:
        2D numpy array where each row is an IMF, or None if PyEMD import fails
    """
    try:
        from PyEMD import CEEMDAN

        ce = CEEMDAN(trials=50, noise_width=0.2, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imfs = ce.ceemdan(x, max_imf=max_imf)
        return imfs
    except Exception:
        return None


def _hilbert_amp_phase(imf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the analytic signal and extract amplitude and phase using Hilbert transform.

    Args:
        imf: 1D array representing an Intrinsic Mode Function (IMF)

    Returns:
        Tuple of (amplitude, phase) where:
        - amplitude: Instantaneous amplitude (envelope) of the signal
        - phase: Instantaneous phase of the signal in radians
    """
    from scipy.signal import hilbert

    z = hilbert(imf)
    return np.abs(z), np.angle(z)


def _inst_freq(phase: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Compute the instantaneous frequency from the instantaneous phase.

    The instantaneous frequency is the time derivative of the phase,
    normalized by 2π and the sampling frequency.

    Args:
        phase: Instantaneous phase in radians
        fs: Sampling frequency in Hz (default: 1.0)

    Returns:
        Array of instantaneous frequencies in Hz, clipped to be positive
    """
    up = np.unwrap(phase)
    dphi = np.diff(up, prepend=up[0])
    freq = (dphi / (2 * np.pi)) * fs
    return np.clip(freq, 1e-6, None)


def _group_imfs_by_period(freqs: list[np.ndarray], thresholds: tuple[float, float] = (10, 30)) -> list[int]:
    """
    Group IMFs into frequency bands based on their median period.

    IMFs are classified into three categories:
    - Short-term: period ≤ thresholds[0] days
    - Medium-term: thresholds[0] < period ≤ thresholds[1] days
    - Long-term: period > thresholds[1] days

    Args:
        freqs: List of arrays containing instantaneous frequencies for each IMF
        thresholds: Tuple of (short_threshold, medium_threshold) in days

    Returns:
        List of band indices (0=short, 1=medium, 2=long) for each IMF
    """
    labels = []
    for f in freqs:
        valid = f[np.isfinite(f) & (f > 0)]
        medf = np.median(valid) if valid.size else np.nan
        period = 1.0 / medf if (isinstance(medf, float) and medf > 0) else np.inf
        if period <= thresholds[0]:
            labels.append(0)  # short
        elif period <= thresholds[1]:
            labels.append(1)  # medium
        else:
            labels.append(2)  # long
    return labels


def _band_ts_from_imfs(imfs: np.ndarray, bands: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """
    Combine IMFs into frequency bands and compute band-specific energy and phase.

    For each frequency band (short/medium/long), this function:
    1. Sums the squared amplitudes (energy) of all IMFs in the band
    2. Computes the phase of the combined analytic signal

    Args:
        imfs: 2D array where each row is an IMF
        bands: List of band indices (0/1/2) for each IMF

    Returns:
        Dictionary mapping band index to a dict with:
        - 'E': Energy time series for the band
        - 'phase': Combined phase time series for the band
    """
    B = {0: {'E': None, 'phase': None}, 1: {'E': None, 'phase': None}, 2: {'E': None, 'phase': None}}
    for j in range(imfs.shape[0]):
        amp, ph = _hilbert_amp_phase(imfs[j])
        E = amp**2
        b = bands[j]
        B[b]['E'] = E if B[b]['E'] is None else B[b]['E'] + E
        comp = amp * np.exp(1j * ph)
        B[b]['phase'] = comp if B[b]['phase'] is None else B[b]['phase'] + comp
    for b in B:
        if B[b]['phase'] is not None:
            B[b]['phase'] = np.angle(B[b]['phase'])
        else:
            N = imfs.shape[1]
            B[b]['phase'] = np.full(N, np.nan)
        if B[b]['E'] is None:
            B[b]['E'] = np.zeros(imfs.shape[1])
    return B


def multiscale_summary(prices: pd.DataFrame, ticker: str) -> dict[str, pd.Series]:
    ret = prices[ticker].pct_change().dropna()
    imfs = try_ceemdan_imfs(ret.values, max_imf=None, seed=1337)
    if imfs is not None:
        imf_freqs = []
        for j in range(imfs.shape[0]):
            _, ph = _hilbert_amp_phase(imfs[j])
            imf_freqs.append(_inst_freq(ph, fs=1.0))
        bands = _group_imfs_by_period(imf_freqs, thresholds=(10, 30))
        bands_ts = _band_ts_from_imfs(imfs, bands)
        idx_tail = prices.index[-imfs.shape[1] :]
        E_short = pd.Series(bands_ts[0]['E'], index=idx_tail).reindex(prices.index)
        E_med = pd.Series(bands_ts[1]['E'], index=idx_tail).reindex(prices.index)
        E_long = pd.Series(bands_ts[2]['E'], index=idx_tail).reindex(prices.index)
        phase_short = pd.Series(bands_ts[0]['phase'], index=idx_tail).reindex(prices.index)
        phase_med = pd.Series(bands_ts[1]['phase'], index=idx_tail).reindex(prices.index)
        phase_long = pd.Series(bands_ts[2]['phase'], index=idx_tail).reindex(prices.index)
        return dict(
            method='ceemdan',
            E_short=E_short,
            E_med=E_med,
            E_long=E_long,
            phase_short=phase_short,
            phase_med=phase_med,
            phase_long=phase_long,
        )

    # Proxy path (rolling vars)
    rv5 = ret.rolling(5).var().bfill()
    rv20 = ret.rolling(20).var().bfill()
    rv60 = ret.rolling(60).var().bfill()
    E_short = rv5.reindex(prices.index).bfill()
    E_med = rv20.reindex(prices.index).bfill()
    E_long = rv60.reindex(prices.index).bfill()
    Na = np.nan * np.zeros(len(prices.index))
    return dict(
        method='proxy',
        E_short=E_short,
        E_med=E_med,
        E_long=E_long,
        phase_short=pd.Series(Na, index=prices.index),
        phase_med=pd.Series(Na, index=prices.index),
        phase_long=pd.Series(Na, index=prices.index),
    )


# ------------------------- Regimes & targets -------------------


def vix_regimes(vix_df: pd.DataFrame) -> pd.DataFrame:
    df = vix_df.copy()
    df['slope'] = df['^VIX3M'] - df['^VIX']
    terc = np.nanpercentile(df['^VIX'].values, [33.3, 66.6])

    def level(v):
        if np.isnan(v):
            return np.nan
        if v <= terc[0]:
            return 'low'
        if v <= terc[1]:
            return 'mid'
        return 'high'

    def slope_state(s):
        if np.isnan(s):
            return np.nan
        if s <= 0:
            return 'backwardation'
        if s <= 2:
            return 'flat'
        return 'contango'

    df['vix_level'] = df['^VIX'].apply(level)
    df['vix_slope_state'] = df['slope'].apply(slope_state)
    return df


def target_beta_from_ticker(t: str) -> float:
    t = t.upper()
    if 'UPRO' in t:
        return 3.0
    if 'SSO' in t:
        return 2.0
    if 'SPXU' in t:
        return -3.0
    if 'SDS' in t:
        return -2.0
    return float('nan')


# --------------------------- Plots -----------------------------


def plot_predictor_distributions(vix_df: pd.DataFrame):
    df = vix_df.copy()
    df['vix_level'] = df['^VIX']
    df['vix_slope'] = df['^VIX3M'] - df['^VIX']
    df = df.dropna(subset=['vix_level', 'vix_slope'])
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(df['vix_level'], bins=50, alpha=0.7)
    axes[0].set_title('Distribution of VIX Level')
    axes[0].set_xlabel('VIX')
    axes[0].set_ylabel('Frequency')
    axes[1].hist(df['vix_slope'], bins=50, alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='Backwardation (slope < 0)')
    axes[1].set_title('Distribution of VIX Term-Structure Slope')
    axes[1].set_xlabel('VIX3M - VIX')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    fig.suptitle('Distributions of Key VIX Predictors', fontsize=12)
    fig.tight_layout()
    safesave('figures/predictor_distributions.png', dpi=160)
    safesave('figures/predictor_distributions.pdf', dpi=160)
    plt.close()


def plot_short_energy_timeseries_annotated(spy_sum: dict[str, pd.Series], regimes: pd.DataFrame, top_n_spikes: int = 5):
    df_plot = pd.DataFrame({'short_energy': spy_sum['E_short'], 'state': regimes['vix_slope_state']}).dropna()
    if df_plot.empty:
        return
    x = df_plot.index
    y = df_plot['short_energy'].values
    st = df_plot['state'].values
    fig, ax = plt.subplots(figsize=(10, 3))
    current_state = None
    block_start = None
    for idx in range(len(x)):
        state_here = st[idx]
        if state_here != current_state:
            if current_state is not None:
                ax.axvspan(
                    block_start,
                    x[idx - 1],
                    alpha=0.12,
                    color=(
                        'red' if current_state == 'backwardation' else 'orange' if current_state == 'flat' else 'green'
                    ),
                )
            current_state = state_here
            block_start = x[idx]
    if current_state is not None:
        ax.axvspan(
            block_start,
            x[-1],
            alpha=0.12,
            color=('red' if current_state == 'backwardation' else 'orange' if current_state == 'flat' else 'green'),
        )
    ax.plot(x, y, linewidth=1.0, color='black')
    spike_idx_sorted = np.argsort(y)[-top_n_spikes:]
    for i in spike_idx_sorted:
        spike_date = x[i]
        spike_val = y[i]
        ax.scatter([spike_date], [spike_val], color='black', zorder=5)
        ax.text(spike_date, spike_val, spike_date.strftime('%Y-%m-%d'), fontsize=7, rotation=45, ha='left', va='bottom')
    proxy = [matplotlib.lines.Line2D([0], [0], color=c, lw=6, alpha=0.3) for c in ('red', 'orange', 'green')]
    ax.legend(proxy, ['backwardation', 'flat', 'contango'], loc='upper right', fontsize=8, frameon=False)
    ax.set_title('SPY short-horizon energy (annotated)')
    ax.set_ylabel('Short-band energy')
    ax.set_xlabel('Date')
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2)
    fig.savefig('figures/spy_short_energy_regime_annotated.png', dpi=160)
    fig.savefig('figures/spy_short_energy_regime_annotated.pdf', dpi=160)
    plt.close()


def plot_rolling_beta(prices: pd.DataFrame, regimes: pd.DataFrame, letf: str, window: int = 30):
    spy_ret = prices[UNDERLYING].pct_change()
    letf_ret = prices[letf].pct_change()
    df = pd.DataFrame({'spy_ret': spy_ret, 'letf_ret': letf_ret, 'state': regimes['vix_slope_state']}).dropna()
    if df.empty or df.shape[0] < window + 5:
        return
    num = (df['spy_ret'] * df['letf_ret']).rolling(window).sum()
    den = (df['spy_ret'] ** 2).rolling(window).sum()
    rolling_beta = num / den
    plot_df = pd.DataFrame({'beta': rolling_beta, 'state': df['state']}).dropna()
    if plot_df.empty:
        return
    x = plot_df.index
    st = plot_df['state'].values
    plt.figure(figsize=(10, 3))
    ax = plt.gca()
    current_state = None
    block_start = None
    for idx in range(len(x)):
        this_state = st[idx]
        if this_state != current_state:
            if current_state is not None:
                ax.axvspan(
                    block_start,
                    x[idx - 1],
                    alpha=0.12,
                    color=(
                        'red' if current_state == 'backwardation' else 'orange' if current_state == 'flat' else 'green'
                    ),
                )
            current_state = this_state
            block_start = x[idx]
    if current_state is not None:
        ax.axvspan(
            block_start,
            x[-1],
            alpha=0.12,
            color=('red' if current_state == 'backwardation' else 'orange' if current_state == 'flat' else 'green'),
        )
    ax.plot(x, plot_df['beta'].values, linewidth=1.0, label=f'rolling β ({window}d)')
    tgt = target_beta_from_ticker(letf)
    if np.isfinite(tgt):
        ax.axhline(tgt, linestyle='--', linewidth=1.0, alpha=0.6, label=f'target {tgt:.0f}x')
    ax.set_title(f'Rolling realized leverage β(t) — {letf}')
    ax.set_ylabel('β(t)')
    ax.set_xlabel('Date')
    plt.tight_layout()
    safesave(f'figures/{letf}/rolling_beta_{letf}.png', dpi=160)
    safesave(f'figures/{letf}/rolling_beta_{letf}.pdf', dpi=160)
    plt.close()


def plot_drag_timeseries(prices: pd.DataFrame, regimes: pd.DataFrame, letf: str, k: int = 5):
    beta_target = target_beta_from_ticker(letf)
    if not np.isfinite(beta_target):
        return
    feats = build_xyv(prices[[UNDERLYING, letf]].rename(columns={UNDERLYING: 'S', letf: 'L'}), 'S', 'L', k).copy()
    feats['drag'] = feats['y'] - beta_target * feats['x']
    feats['state'] = regimes['vix_slope_state']
    feats = feats.dropna(subset=['drag', 'state'])
    if feats.empty:
        return
    x = feats.index
    y = feats['drag'].values
    st = feats['state'].values
    plt.figure(figsize=(10, 3))
    ax = plt.gca()
    current_state = None
    block_start = None
    for idx in range(len(x)):
        this_state = st[idx]
        if this_state != current_state:
            if current_state is not None:
                ax.axvspan(
                    block_start,
                    x[idx - 1],
                    alpha=0.12,
                    color=(
                        'red' if current_state == 'backwardation' else 'orange' if current_state == 'flat' else 'green'
                    ),
                )
            current_state = this_state
            block_start = x[idx]
    if current_state is not None:
        ax.axvspan(
            block_start,
            x[-1],
            alpha=0.12,
            color=('red' if current_state == 'backwardation' else 'orange' if current_state == 'flat' else 'green'),
        )
    ax.plot(x, y, linewidth=1.0)
    ax.axhline(0, color='gray', lw=1)
    ax.set_title(f'{letf} realized {k}-day drag over time')
    ax.set_ylabel('drag = y - β_target·x')
    ax.set_xlabel('Date')
    plt.tight_layout()
    safesave(f'figures/{letf}/drag_timeseries_{letf}_k{k}.png', dpi=160)
    safesave(f'figures/{letf}/drag_timeseries_{letf}_k{k}.pdf', dpi=160)
    plt.close()


def plot_drag_heatmap(rows_drag: list[dict], letf: str):
    ddf = pd.DataFrame(rows_drag).pivot(index='k', columns='state', values='median_drag')
    plt.figure(figsize=(6, 3))
    im = plt.imshow(ddf.values, aspect='auto')
    plt.xticks(range(ddf.shape[1]), ddf.columns)
    plt.yticks(range(ddf.shape[0]), ddf.index)
    plt.colorbar(im, fraction=0.046)
    plt.title(f'Median drag (y - β_target·x): {letf}')
    plt.tight_layout()
    safesave(f'figures/{letf}/drag_heatmap_{letf}.png', dpi=160)
    plt.close()


def plot_drag_vs_vixslope_scatter(prices: pd.DataFrame, regimes: pd.DataFrame, letf: str, k_list: list[int]):
    beta_target = target_beta_from_ticker(letf)
    if not np.isfinite(beta_target):
        return
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    for k in k_list:
        feats = build_xyv(prices[[UNDERLYING, letf]].rename(columns={UNDERLYING: 'S', letf: 'L'}), 'S', 'L', k)
        feats['vix_slope'] = regimes['^VIX3M'] - regimes['^VIX']
        feats['drag'] = feats['y'] - beta_target * feats['x']
        sub = feats.dropna(subset=['vix_slope', 'drag'])
        if len(sub) < 10:
            continue
        ax.scatter(sub['vix_slope'], sub['drag'], alpha=0.25, s=10, label=f'k={k}d')
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1, linestyle=':')
    ax.set_xlabel('VIX3M - VIX')
    ax.set_ylabel('drag (y - β_target·x)')
    ax.set_title(f'Drag vs VIX slope — {letf}')
    ax.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    safesave(f'figures/{letf}/drag_vs_vixslope_{letf}.png', dpi=160)
    safesave(f'figures/{letf}/drag_vs_vixslope_{letf}.pdf', dpi=160)
    plt.close()


def plot_forecast_sensitivity(forecast_table_csv: str):
    try:
        df = pd.read_csv(forecast_table_csv)
    except Exception:
        return
    if df.empty:
        return
    plt.figure(figsize=(6, 4))
    for ticker in df['ticker'].unique():
        sub = df[df['ticker'] == ticker].sort_values('k')
        plt.plot(sub['k'], sub['coef_vix_level'], marker='o', label=ticker)
    plt.axhline(0, color='gray', lw=1)
    plt.xlabel('k (days)')
    plt.ylabel('Sensitivity of drag to VIX level')
    plt.title('Forecasted drag sensitivity to VIX level')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    safesave('figures/drag_forecast_sensitivity.png', dpi=160)
    safesave('figures/drag_forecast_sensitivity.pdf', dpi=160)
    plt.close()


def plot_forecast_sensitivity_energy(forecast_table_csv: str):
    """
    Plots sensitivity of drag to the 'short_energy_spy' coefficient.
    """
    try:
        df = pd.read_csv(forecast_table_csv)
    except Exception:
        return
    if df.empty or 'coef_short_energy' not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    for ticker in df['ticker'].unique():
        sub = df[df['ticker'] == ticker].sort_values('k')
        plt.plot(sub['k'], sub['coef_short_energy'], marker='o', label=ticker)
    plt.axhline(0, color='gray', lw=1)
    plt.xlabel('k (days)')
    plt.ylabel('Sensitivity of drag to Short Energy')
    plt.title('Forecasted drag sensitivity to Short Energy')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    safesave('figures/drag_forecast_sensitivity_energy.png', dpi=160)
    safesave('figures/drag_forecast_sensitivity_energy.pdf', dpi=160)
    plt.close()


def plot_in_sample_R2(forecast_table_csv: str):
    """
    Show how well our forecast model fits (in-sample R^2)
    as a function of horizon k, per LETF.
    This corresponds to the "How predictable is drag?" chart.
    """
    try:
        df = pd.read_csv(forecast_table_csv)
    except Exception:
        return
    if df.empty:
        return

    plt.figure(figsize=(6, 4))
    for ticker in df['ticker'].unique():
        sub = df[df['ticker'] == ticker].sort_values('k')
        plt.plot(sub['k'], sub['R2_in_sample'], marker='o', label=ticker)
    plt.axhline(0, color='gray', lw=1)
    plt.xlabel('Horizon k (trading days)')
    plt.ylabel('In-sample R² of drag forecast')
    plt.title('How predictable is drag? (In-Sample $R^2$)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    safesave('figures/how_predictable_is_drag_in_sample.png', dpi=160)
    safesave('figures/how_predictable_is_drag_in_sample.pdf', dpi=160)
    plt.close()


def plot_predicted_vs_realized_drag(
    prices: pd.DataFrame, spy_sum: dict[str, pd.Series], vix: pd.DataFrame, letf: str, k: int
):
    beta_target = target_beta_from_ticker(letf)
    if not np.isfinite(beta_target):
        return
    feats = build_xyv(prices[[UNDERLYING, letf]].rename(columns={UNDERLYING: 'S', letf: 'L'}), 'S', 'L', k).copy()
    feat_daily = pd.DataFrame(index=prices.index)
    feat_daily['vix_level'] = vix['^VIX']
    feat_daily['vix_slope'] = vix['^VIX3M'] - vix['^VIX']
    feat_daily['short_energy_spy'] = spy_sum['E_short']
    tmp = feats.join(feat_daily, how='left')
    tmp['drag'] = tmp['y'] - beta_target * tmp['x']
    cols = ['drag', 'vix_level', 'vix_slope', 'short_energy_spy']
    tmp2 = tmp.dropna(subset=cols).copy()
    if len(tmp2) < 30:
        return
    X = sm.add_constant(tmp2[['vix_level', 'vix_slope', 'short_energy_spy']], has_constant='add')
    yvec = tmp2['drag']
    model = sm.OLS(yvec, X).fit()
    tmp2['pred_drag'] = model.predict(X)
    plt.figure(figsize=(4, 4))
    plt.scatter(tmp2['drag'], tmp2['pred_drag'], alpha=0.3, s=10)
    lo = np.nanmin([tmp2['drag'].min(), tmp2['pred_drag'].min()])
    hi = np.nanmax([tmp2['drag'].max(), tmp2['pred_drag'].max()])
    plt.plot([lo, hi], [lo, hi], color='gray', lw=1)
    plt.xlabel('Realized drag')
    plt.ylabel('Predicted drag')
    plt.title(f'{letf} predicted vs realized drag (k={k}d)')
    plt.tight_layout()
    safesave(f'figures/{letf}/pred_vs_real_drag_{letf}_k{k}.png', dpi=160)
    safesave(f'figures/{letf}/pred_vs_real_drag_{letf}_k{k}.pdf', dpi=160)
    plt.close()


def plot_beta_vs_k_triple(dfb: pd.DataFrame, letf: str, is_anchor: bool):
    """
    Plots target β (flat), structural β (NLS), and realized β (OLS slope on x) vs horizon k.
    Expects dfb columns: ['k', 'beta_nls', 'beta_ols_x'].
    """
    tgt = target_beta_from_ticker(letf)

    plt.figure(figsize=(6.5, 4))
    # realized beta from OLS slope on x
    if 'beta_ols_x' in dfb.columns:
        plt.plot(dfb['k'], dfb['beta_ols_x'], marker='o', label='realized β (OLS)')

    # structural beta from NLS
    if 'beta_nls' in dfb.columns:
        plt.plot(dfb['k'], dfb['beta_nls'], marker='s', label='structural β (NLS)')

    # target beta (flat)
    if np.isfinite(tgt):
        plt.axhline(tgt, linestyle='--', linewidth=1.0, alpha=0.8, label=f'target β = {tgt:.0f}x')

    plt.title(f'{letf}: β vs horizon k')
    plt.xlabel('k (days)')
    plt.ylabel('β')
    plt.grid(alpha=0.3)
    plt.legend()

    # save (per-ticker + anchor copies)
    out_png = f'figures/{letf}/beta_vs_k_{letf}.png'
    out_pdf = f'figures/{letf}/beta_vs_k_{letf}.pdf'
    safesave(out_png, dpi=160)
    safesave(out_pdf, dpi=160)
    if is_anchor:
        safesave('figures/beta_vs_k.png', dpi=160)
        safesave('figures/beta_vs_k.pdf', dpi=160)
    plt.close()


def plot_asymmetry_bars(asym_csv: str):
    """
    Plots the asymmetry bar chart.
    """
    try:
        df = pd.read_csv(asym_csv)
        df = df[df['state'].isin(['backwardation', 'flat', 'contango'])]
        if df.empty:
            return

        # Aggregate AI by median across bands
        pivot_data = df.groupby(['state', 'asset'])['AI'].median().unstack()

        # Ensure correct order
        cats = ['backwardation', 'flat', 'contango']
        pivot_data = pivot_data.reindex(cats)

        ax = pivot_data.plot(kind='bar', figsize=(7, 4), rot=0, title='Asymmetric Volatility (Down vs Up) by VIX Slope')
        ax.set_ylabel('AI (median across bands)')
        ax.set_xlabel('VIX Slope State')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        safesave('figures/asymmetry_bars.png', dpi=160)
        safesave('figures/asymmetry_bars.pdf', dpi=160)
        plt.close()
    except Exception as e:
        print(f'Failed to plot asymmetry bars: {e}')


def build_asymmetry_rows(prices, regimes, spy_sum, letf_summaries: dict[str, dict[str, pd.Series]]):
    """
    Builds the dataframe for the asymmetry analysis.
    """
    asym_rows_all = []
    cats = ['backwardation', 'flat', 'contango']
    spy_ret = prices[UNDERLYING].pct_change().fillna(0.0)

    for state in ['all'] + cats:
        mask_state = (
            np.full(len(prices.index), True) if state == 'all' else (regimes['vix_slope_state'] == state).values
        )
        up_mask = (spy_ret.values > 0) & mask_state
        down_mask = (spy_ret.values < 0) & mask_state

        if up_mask.sum() < 10 or down_mask.sum() < 10:
            continue

        # SPY AI
        for band_name, E_series in [
            ('short', spy_sum['E_short']),
            ('medium', spy_sum['E_med']),
            ('long', spy_sum['E_long']),
        ]:
            if E_series is None:
                continue
            E_vals = E_series.values
            up_v = np.nanmedian(E_vals[up_mask])
            dn_v = np.nanmedian(E_vals[down_mask])
            ai_val = (dn_v - up_v) / (dn_v + up_v + 1e-12)
            asym_rows_all.append({'asset': UNDERLYING, 'band': band_name, 'state': state, 'AI': float(ai_val)})

        # LETF AI
        for letf, letf_sum in letf_summaries.items():
            for band_name, E_series in [
                ('short', letf_sum['E_short']),
                ('medium', letf_sum['E_med']),
                ('long', letf_sum['E_long']),
            ]:
                if E_series is None:
                    continue
                E_vals = E_series.values
                up_v = np.nanmedian(E_vals[up_mask])
                dn_v = np.nanmedian(E_vals[down_mask])
                ai_val = (dn_v - up_v) / (dn_v + up_v + 1e-12)
                asym_rows_all.append({'asset': letf, 'band': band_name, 'state': state, 'AI': float(ai_val)})
    return asym_rows_all


# --------------------- OOS walk-forward evaluation ---------------------


def _ensure_dir_for_fig(path: str):
    p = Path(path).parent
    p.mkdir(parents=True, exist_ok=True)


def walkforward_oos(
    prices: pd.DataFrame,
    vix: pd.DataFrame,
    spy_sum: dict[str, pd.Series],
    letf_list: list[str],
    k_list: list[int],
    min_train_days: int = 750,
) -> pd.DataFrame:
    """
    Walk-forward expanding-window OOS evaluation for drag:
      drag_tk = y - beta_target * x
    Model: drag_tk ~ const + vix_level_t + vix_slope_t + short_energy_spy_t

    Returns DataFrame:
      date, ticker, k, y_true, y_pred
    """
    feat_daily = pd.DataFrame(index=prices.index)
    feat_daily['vix_level'] = vix['^VIX']
    feat_daily['vix_slope'] = vix['^VIX3M'] - vix['^VIX']
    feat_daily['short_energy_spy'] = spy_sum['E_short']

    rows = []
    for letf in letf_list:
        beta_target = target_beta_from_ticker(letf)
        if not np.isfinite(beta_target):
            continue

        for k in k_list:
            feats = build_xyv(
                prices[[UNDERLYING, letf]].rename(columns={UNDERLYING: 'S', letf: 'L'}), 'S', 'L', k
            ).copy()

            tmp = feats.join(feat_daily, how='left')
            tmp['drag'] = tmp['y'] - beta_target * tmp['x']
            tmp = tmp.dropna(subset=['drag', 'vix_level', 'vix_slope', 'short_energy_spy']).copy()
            if len(tmp) < (min_train_days + 30):
                continue

            dates = tmp.index
            for i in range(min_train_days, len(tmp)):
                train = tmp.iloc[:i]
                test = tmp.iloc[i : i + 1]

                # Force intercept consistently for both train and test
                X_tr = sm.add_constant(train[['vix_level', 'vix_slope', 'short_energy_spy']], has_constant='add')
                y_tr = train['drag'].to_numpy(dtype=float, copy=False)

                beta = np.linalg.lstsq(X_tr.values, y_tr, rcond=None)[0]  # (4,)

                X_te = sm.add_constant(test[['vix_level', 'vix_slope', 'short_energy_spy']], has_constant='add')
                y_hat = float((X_te.values @ beta).ravel()[0])
                y_true = float(test['drag'].iloc[0])

                rows.append(
                    {
                        'date': dates[i],
                        'ticker': letf,
                        'k': int(k),
                        'y_true': y_true,
                        'y_pred': y_hat,
                    }
                )

    oos = (
        pd.DataFrame(rows).sort_values(['ticker', 'k', 'date'])
        if rows
        else pd.DataFrame(columns=['date', 'ticker', 'k', 'y_true', 'y_pred'])
    )
    return oos


def summarize_oos(oos: pd.DataFrame) -> pd.DataFrame:
    if oos.empty:
        return pd.DataFrame(columns=['ticker', 'k', 'n', 'oos_R2', 'hit_rate', 'mae', 'rmse'])

    def _summ(grp):
        y = grp['y_true'].values
        yhat = grp['y_pred'].values
        if len(grp) < 10:
            return pd.Series({'n': len(grp), 'oos_R2': np.nan, 'hit_rate': np.nan, 'mae': np.nan, 'rmse': np.nan})
        sse = float(np.sum((y - yhat) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (sse / sst) if sst > 1e-12 else np.nan
        hit = float(np.mean(np.sign(y) == np.sign(yhat)))
        mae = float(np.mean(np.abs(y - yhat)))
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        return pd.Series({'n': len(grp), 'oos_R2': r2, 'hit_rate': hit, 'mae': mae, 'rmse': rmse})

    summ = oos.groupby(['ticker', 'k'], as_index=False).apply(_summ).reset_index(drop=True)
    return summ


def plot_oos_r2_vs_k(oos_summary: pd.DataFrame):
    if oos_summary.empty:
        return
    plt.figure(figsize=(6, 4))
    for t in oos_summary['ticker'].unique():
        sub = oos_summary[oos_summary['ticker'] == t].sort_values('k')
        plt.plot(sub['k'], sub['oos_R2'], marker='o', label=t)
    plt.axhline(0, lw=1, color='gray')
    plt.xlabel('k (days)')
    plt.ylabel('Out-of-sample $R^2$')
    plt.title('OOS $R^2$ of drag forecast by horizon')
    plt.grid(alpha=0.3)
    plt.legend()
    _ensure_dir_for_fig('figures/oos_R2_vs_k.png')
    safesave('figures/oos_R2_vs_k.png', dpi=160)
    safesave('figures/oos_R2_vs_k.pdf', dpi=160)
    plt.close()


def plot_oos_hitrate_vs_k(oos_summary: pd.DataFrame):
    if oos_summary.empty:
        return
    plt.figure(figsize=(6, 4))
    for t in oos_summary['ticker'].unique():
        sub = oos_summary[oos_summary['ticker'] == t].sort_values('k')
        plt.plot(sub['k'], sub['hit_rate'], marker='o', label=t)
    plt.axhline(0.5, lw=1, color='gray', linestyle='--')  # coin flip
    plt.xlabel('k (days)')
    plt.ylabel('Sign hit-rate  P[ sign(pred)=sign(real) ]')
    plt.title('OOS sign hit-rate by horizon')
    plt.grid(alpha=0.3)
    plt.legend()
    _ensure_dir_for_fig('figures/oos_hitrate_vs_k.png')
    safesave('figures/oos_hitrate_vs_k.png', dpi=160)
    safesave('figures/oos_hitrate_vs_k.pdf', dpi=160)
    plt.close()


def plot_calibration_for_anchor(oos: pd.DataFrame, anchor: str = 'UPRO', k: int = 5, n_bins: int = 10):
    df = oos[(oos['ticker'] == anchor) & (oos['k'] == k)].copy()
    if df.empty or df['y_pred'].nunique() < n_bins:
        return
    df['bin'] = pd.qcut(df['y_pred'], q=n_bins, duplicates='drop')
    calib = (
        df.groupby('bin', observed=True)
        .agg(mean_pred=('y_pred', 'mean'), mean_real=('y_true', 'mean'), n=('y_true', 'size'))
        .reset_index(drop=True)
        .sort_values('mean_pred')
    )
    plt.figure(figsize=(4.5, 4))
    plt.plot(calib['mean_pred'], calib['mean_real'], marker='o')
    lo = float(np.nanmin([calib['mean_pred'].min(), calib['mean_real'].min()]))
    hi = float(np.nanmax([calib['mean_pred'].max(), calib['mean_real'].max()]))
    plt.plot([lo, hi], [lo, hi], color='gray', lw=1)  # ideal line
    plt.xlabel('Predicted drag (bin mean)')
    plt.ylabel('Realized drag (bin mean)')
    plt.title(f'Calibration — {anchor}, k={k}')
    plt.grid(alpha=0.3)
    out_png = f'figures/{anchor}/calibration_{anchor}_k{k}.png'
    out_pdf = f'figures/{anchor}/calibration_{anchor}_k{k}.pdf'
    _ensure_dir_for_fig(out_png)
    safesave(out_png, dpi=160)
    safesave(out_pdf, dpi=160)
    plt.close()


def plot_rolling_beta_vs_vix_slope(prices: pd.DataFrame, vix: pd.DataFrame, k: int = 21, window: int = 21):
    """
    Create scatterplot of rolling 21-day annualized volatility of the underlying vs the VIX slope,
    and save it to figures/{TICKER}/rolling_beta_{TICKER}.(png|pdf).
    """
    assert UNDERLYING in prices
    assert 'VIX' in vix
    assert prices.index.equals(vix.index)

    # Calculate features
    feats = build_xyv(prices[['S', 'L']], 'S', 'L', k)
    feats['beta'] = feats['y'] - feats['x'] * target_beta_from_ticker('L')
    feats['vix_slope'] = vix['VIX'].diff() / vix.index.to_series().to_period('D').diff().dt.days
    feats = feats.dropna(subset=['beta', 'vix_slope'])

    # Calculate rolling statistics
    rolling = feats.rolling(window).agg(
        {
            'beta': ['std', 'mean'],
            'vix_slope': ['std', 'mean'],
        }
    )
    rolling.columns = ['_'.join(col).strip('_') for col in rolling.columns.to_list()]
    rolling['beta_std_over_vix_slope_std'] = rolling['beta_std'] / rolling['vix_slope_std']

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(rolling['vix_slope_mean'], rolling['beta_mean'], s=5, alpha=0.7, label='rolling window')
    plt.xlabel('VIX slope')
    plt.ylabel('β')
    plt.title(f'Rolling beta vs VIX slope, {window}D window')
    plt.grid(alpha=0.3)
    plt.legend()
    _ensure_dir_for_fig(f'figures/{UNDERLYING}/rolling_beta_{UNDERLYING}.png')
    safesave(f'figures/{UNDERLYING}/rolling_beta_{UNDERLYING}.png', dpi=160)
    safesave(f'figures/{UNDERLYING}/rolling_beta_{UNDERLYING}.pdf', dpi=160)
    plt.close()


def plot_underlying_vs_slope(prices: pd.DataFrame, vix: pd.DataFrame):
    vix_slope = (vix['^VIX3M'] - vix['^VIX']).rename('vix_slope')
    rts = (np.log(prices[UNDERLYING]) / np.log(prices[UNDERLYING].shift(1)) - 1).rename('rts') * 100
    rts_and_slope = (
        rts.rolling(window=22)
        .std()
        .mul(np.sqrt(252))
        .rename('vol')
        .to_frame()
        .join(vix_slope)
        .join(rts.shift(-22).rolling(window=22).std().mul(np.sqrt(252)).rename('vol_fwd'))
        .dropna()
    )
    rts_and_slope.index.name = 'date'
    rts_and_slope.reset_index(inplace=True)
    ax = rts_and_slope.plot(
        kind='scatter', x='vix_slope', y='vol_fwd', color='black', title='VIX Slope vs SPY Volatility', figsize=(6, 4)
    )
    ax.set_xlabel('VIX Slope')
    ax.set_ylabel('SPY Volatility')

    covid_mask = (rts_and_slope['date'] >= '2020-02-27') & (rts_and_slope['date'] <= '2020-03-31')
    trump_tariffs_mask = (rts_and_slope['date'] >= '2025-04-07') & (rts_and_slope['date'] <= '2025-04-08')
    sovereign_credit_mask = (rts_and_slope['date'] >= '2011-08-08') & (rts_and_slope['date'] <= '2011-08-08')
    yuan_devaluation = (rts_and_slope['date'] >= '2015-08-24') & (rts_and_slope['date'] <= '2015-08-24')

    ax.scatter(
        rts_and_slope.loc[covid_mask, 'vix_slope'], rts_and_slope.loc[covid_mask, 'vol_fwd'], color='red', label='COVID'
    )
    ax.scatter(
        rts_and_slope.loc[trump_tariffs_mask, 'vix_slope'],
        rts_and_slope.loc[trump_tariffs_mask, 'vol_fwd'],
        color='purple',
        label='Trump Tariffs',
    )
    ax.scatter(
        rts_and_slope.loc[sovereign_credit_mask, 'vix_slope'],
        rts_and_slope.loc[sovereign_credit_mask, 'vol_fwd'],
        color='green',
        label='US Sovereign Credit Downgrade',
    )
    ax.scatter(
        rts_and_slope.loc[sovereign_credit_mask, 'vix_slope'],
        rts_and_slope.loc[yuan_devaluation, 'vol_fwd'],
        color='orange',
        label='Flash Crash - Yuan Devaluation',
    )

    ax.legend()

    _ensure_dir_for_fig(f'figures/{UNDERLYING}/scatter_slope_{UNDERLYING}.png')
    safesave(f'figures/{UNDERLYING}/scatter_slope_{UNDERLYING}.png', dpi=160)
    safesave(f'figures/{UNDERLYING}/scatter_slope_{UNDERLYING}.pdf', dpi=160)
    plt.close()


# --------------------------- Main ------------------------------


def main():
    tickers = [UNDERLYING] + LETF_LIST + VIX_TICKERS

    # Data
    dfs = ensure_prices(tickers, START, END)
    prices = align_adj_close({k: v for k, v in dfs.items() if not k.startswith('^')})
    vix = align_adj_close({k: v for k, v in dfs.items() if k.startswith('^')})

    all_dates = prices.index.intersection(vix.index)
    prices = prices.loc[all_dates]
    vix = vix.loc[all_dates]

    regimes = vix_regimes(vix)

    # Setup visuals
    plot_predictor_distributions(vix)
    spy_sum = multiscale_summary(prices, UNDERLYING)
    plot_short_energy_timeseries_annotated(spy_sum, regimes, top_n_spikes=5)
    plot_underlying_vs_slope(prices, vix)

    # Per-LETF analytics + plots
    hac_rows_all = []

    for letf in LETF_LIST:
        per_asset_rows = []
        for k in K_LIST:
            feats = build_xyv(prices[[UNDERLYING, letf]].rename(columns={UNDERLYING: 'S', letf: 'L'}), 'S', 'L', k)
            if len(feats) == 0:
                continue

            # Simple OLS (legacy) for table completeness
            y_arr = feats['y'].values
            X_arr = np.column_stack([feats['x'].values, feats['v'].values, np.ones(len(feats))])
            b, diag = ols_y_on_X(y_arr, X_arr)

            # HAC for k=5
            if k == 5:
                feats_k5 = feats.copy()
                hac_out = fit_ols_hac(feats_k5, lags=4)
                hac_out['asset'] = letf
                hac_rows_all.append(hac_out)

            # Structural beta via NLS
            r = R_ANNUAL
            f_fee = F_MAP.get(letf, 0.009)
            beta_nls, sse = nls_beta(y_arr, feats['x'].values, feats['v'].values, r, f_fee, k)

            # legacy OLS (gives realized β on x)
            beta_ols_x = float(b[0])  # slope on x

            per_asset_rows.append({'asset': letf, 'k': k, 'beta_nls': beta_nls, 'beta_ols_x': beta_ols_x})

            # Save detail tables for k=5
            if k == 5:
                pd.DataFrame(
                    {
                        'beta_hat_ols': [b[0]],
                        'theta_hat_ols': [b[1]],
                        'alpha_hat_ols': [b[2]],
                        'R2': [diag['R2']],
                        'SSE': [diag['SSE']],
                        'n': [diag['n']],
                    }
                ).to_csv(f'results/ols_k5_{letf}.csv', index=False, encoding='utf-8')

                with open(f'results/nls_k5_{letf}.txt', 'w') as fp:
                    fp.write(f'beta_hat_nls={beta_nls:.8f}\nSSE={sse:.10g}\n')

                if letf == ANCHOR:
                    pd.DataFrame(
                        {
                            'beta_hat_ols': [b[0]],
                            'theta_hat_ols': [b[1]],
                            'alpha_hat_ols': [b[2]],
                            'R2': [diag['R2']],
                            'SSE': [diag['SSE']],
                            'n': [diag['n']],
                        }
                    ).to_csv('results/ols_k5.csv', index=False, encoding='utf-8')
                    pd.DataFrame({'beta_hat_nls': [beta_nls], 'SSE': [sse]}).to_csv(
                        'results/nls_k5.csv', index=False, encoding='utf-8'
                    )

        # Beta vs k table/plot
        if per_asset_rows:
            dfb = pd.DataFrame(per_asset_rows).sort_values('k')
            dfb.to_csv(f'results/beta_vs_k_{letf}.csv', index=False, encoding='utf-8')

            # also keep a compact anchor CSV for the paper if you want
            if letf == ANCHOR:
                dfb[['k', 'beta_nls', 'beta_ols_x']].to_csv('results/beta_vs_k.csv', index=False, encoding='utf-8')

            plot_beta_vs_k_triple(dfb, letf, is_anchor=(letf == ANCHOR))

        # Key per-ticker visuals
        plot_rolling_beta(prices, regimes, letf, window=30)

        # Drag heatmap ingredients
        cats = ['backwardation', 'flat', 'contango']
        rows_drag = []
        beta_target = target_beta_from_ticker(letf)
        for k in K_LIST:
            feats_drag = build_xyv(prices[[UNDERLYING, letf]].rename(columns={UNDERLYING: 'S', letf: 'L'}), 'S', 'L', k)
            feats_drag = feats_drag.join(regimes[['vix_slope_state']], how='left')
            feats_drag['drag'] = feats_drag['y'] - beta_target * feats_drag['x']
            for st in cats:
                median_drag_state = float(np.nanmedian(feats_drag.loc[feats_drag['vix_slope_state'] == st, 'drag']))
                rows_drag.append({'k': k, 'state': st, 'median_drag': median_drag_state})

        plot_drag_heatmap(rows_drag, letf)
        plot_drag_timeseries(prices, regimes, letf, k=5)

        # One slope→drag scatter (UPRO only)
        if letf == 'UPRO':
            plot_drag_vs_vixslope_scatter(prices, regimes, letf, list(K_LIST))

    # HAC summary across tickers at k=5
    if hac_rows_all:
        pd.DataFrame(hac_rows_all).to_csv('results/ols_k5_hac.csv', index=False, encoding='utf-8')

    letf_summaries_for_asym = {letf: multiscale_summary(prices, letf) for letf in LETF_LIST}
    ai_rows_all = build_asymmetry_rows(prices, regimes, spy_sum, letf_summaries_for_asym)
    if ai_rows_all:
        pd.DataFrame(ai_rows_all).to_csv('results/asymmetry_table.csv', index=False, encoding='utf-8')
        plot_asymmetry_bars('results/asymmetry_table.csv')

    # Expected drag by today's regime
    expected_rows = []
    cats = ['backwardation', 'flat', 'contango']
    for letf in LETF_LIST:
        beta_target = target_beta_from_ticker(letf)
        for k in K_LIST:
            feats_exp = build_xyv(prices[[UNDERLYING, letf]].rename(columns={UNDERLYING: 'S', letf: 'L'}), 'S', 'L', k)
            feats_exp = feats_exp.join(regimes[['vix_slope_state']], how='left')
            feats_exp['drag'] = feats_exp['y'] - beta_target * feats_exp['x']
            med_back = float(np.nanmedian(feats_exp.loc[feats_exp['vix_slope_state'] == 'backwardation', 'drag']))
            med_flat = float(np.nanmedian(feats_exp.loc[feats_exp['vix_slope_state'] == 'flat', 'drag']))
            med_cont = float(np.nanmedian(feats_exp.loc[feats_exp['vix_slope_state'] == 'contango', 'drag']))
            today_state = str(regimes['vix_slope_state'].iloc[-1])
            nowcast = {'backwardation': med_back, 'flat': med_flat, 'contango': med_cont}.get(today_state, float('nan'))
            expected_rows.append(
                {
                    'asset': letf,
                    'k': k,
                    'regime_today': today_state,
                    'expected_drag_next_k': nowcast,
                    'median_drag_backwardation': med_back,
                    'median_drag_flat': med_flat,
                    'median_drag_contango': med_cont,
                }
            )
    pd.DataFrame(expected_rows).to_csv('results/expected_drag_by_regime.csv', index=False, encoding='utf-8')

    # Forecast models (drag ~ vix-level + vix-slope + short-energy)
    feat_daily = pd.DataFrame(index=prices.index)
    feat_daily['vix_level'] = vix['^VIX']
    feat_daily['vix_slope'] = vix['^VIX3M'] - vix['^VIX']
    feat_daily['short_energy_spy'] = spy_sum['E_short']

    forecast_rows = []
    for letf in LETF_LIST:
        beta_target = target_beta_from_ticker(letf)
        for k in K_LIST:
            feats_f = build_xyv(
                prices[[UNDERLYING, letf]].rename(columns={UNDERLYING: 'S', letf: 'L'}), 'S', 'L', k
            ).copy()
            tmp = feats_f.join(feat_daily, how='left')
            tmp['drag'] = tmp['y'] - beta_target * tmp['x']
            cols_needed = ['drag', 'vix_level', 'vix_slope', 'short_energy_spy']
            tmp2 = tmp.dropna(subset=cols_needed).copy()
            if len(tmp2) < 30:
                continue
            X = sm.add_constant(tmp2[['vix_level', 'vix_slope', 'short_energy_spy']], has_constant='add')
            yvec = tmp2['drag']
            model = sm.OLS(yvec, X).fit()

            params = model.params.to_dict()
            pvals = model.pvalues.to_dict()

            forecast_rows.append(
                {
                    'ticker': letf,
                    'k': k,
                    'const': params.get('const', np.nan),
                    'coef_vix_level': params.get('vix_level', np.nan),
                    'coef_vix_slope': params.get('vix_slope', np.nan),
                    'coef_short_energy': params.get('short_energy_spy', np.nan),
                    'pval_vix_level': pvals.get('vix_level', np.nan),
                    'pval_vix_slope': pvals.get('vix_slope', np.nan),
                    'pval_short_energy': pvals.get('short_energy_spy', np.nan),
                    'R2_in_sample': model.rsquared,
                    'n_obs': len(tmp2),
                }
            )

    forecast_table = pd.DataFrame(forecast_rows)
    if not forecast_table.empty:
        forecast_table.to_csv('results/drag_forecast_models.csv', index=False, encoding='utf-8')
        forecast_table.to_latex('results/drag_forecast_models.tex', index=False, float_format='%.8f')

        feat_daily.to_csv('results/panel_with_ms.csv', index=True, encoding='utf-8')

        # --- PLOT FORECAST RESULTS ---
        plot_forecast_sensitivity('results/drag_forecast_models.csv')
        plot_in_sample_R2('results/drag_forecast_models.csv')
        plot_predicted_vs_realized_drag(prices, spy_sum, vix, 'UPRO', k=5)
        plot_forecast_sensitivity_energy('results/drag_forecast_models.csv')

    # -------------------------------------------------
    # 7. OUT-OF-SAMPLE (walk-forward) VALIDATION
    # -------------------------------------------------
    oos = walkforward_oos(prices, vix, spy_sum, LETF_LIST, K_LIST, min_train_days=750)
    if not oos.empty:
        oos.to_csv('results/oos_drag_forecast.csv', index=False, encoding='utf-8')
        oos_summary = summarize_oos(oos)
        oos_summary.to_csv('results/oos_drag_summary.csv', index=False, encoding='utf-8')
        oos_summary.to_latex('results/oos_drag_summary.tex', index=False, float_format='%.6f')

        plot_oos_r2_vs_k(oos_summary)
        plot_oos_hitrate_vs_k(oos_summary)
        plot_calibration_for_anchor(oos, anchor=ANCHOR, k=5, n_bins=10)

    print('Done. See results/ and figures/.')


if __name__ == '__main__':
    main()
