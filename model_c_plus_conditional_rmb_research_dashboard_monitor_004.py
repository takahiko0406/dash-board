import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

print("Current working directory:", os.getcwd())


# ============================================================
# MODEL C+ UPGRADE:
# - Baseline universe: QQQM, XLE, XSOE, BIL with TQQQ overlay
# - Upgraded universe: QQQM, XLE, XSOE, XLI, XLB, BIL with TQQQ overlay
# - Adds industrial/materials regime logic:
#       XLB = copper/materials/early industrial cycle
#       XLI = industrial/reshoring/capex cycle
# - Includes old-vs-new comparison in one run
# - Adds a V2 overlay test:
#       V1 = original XLI/XLB overlay
#       V2 = stricter industrial regime classifier to reduce false signals
# - Adds divergence/crash detector risk layer:
#       detects tech-only rallies and scales down TQQQ before full risk-off
# ============================================================

# ============================================================
# 1. SETTINGS
# ============================================================
BASELINE_SECTOR_ETFS = ["QQQM", "XLE", "XSOE"]
UPGRADED_SECTOR_ETFS = ["QQQM", "XLE", "XSOE", "XLI", "XLB"]

cash_etf = "BIL"
spy_etf = "SPY"
feature_etfs = ["ITA", "SOXX", "HYG"]
execution_extra = ["TQQQ", "ERX", "UXI"]

# Download everything needed by both models.
all_assets = sorted(set(
    BASELINE_SECTOR_ETFS
    + UPGRADED_SECTOR_ETFS
    + [cash_etf, spy_etf]
    + feature_etfs
    + execution_extra
))

start_date = "2010-01-01"
end_date = None

forward_return_days = 10
rebalance_step = 10
train_window = 252 * 3
transaction_cost = 0.001
risk_free_rate_annual = 0.0

rf_params = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

overlay_scale = 0.002
risk_off_cash_threshold = 0.01
zscore_window = 252

# ============================================================
# CONDITIONAL DIVERGENCE + SOXX BREAKDOWN DEFENSE SETTINGS
# ============================================================
# Key upgrade vs pure divergence defense:
#   Divergence alone is NOT a sell signal. Your grid search proved that the
#   best profile was no defense because tech-only rallies can keep running.
#
#   This version only reduces TQQQ when BOTH are true:
#     1) Tech/real-economy divergence is high
#     2) SOXX/QQQM short-term momentum starts breaking
#
# That makes it an early crash detector, not an anti-momentum rule.
use_conditional_breakdown_defense = False

conditional_breakdown_rules = {
    "watch": {
        "divergence_min": 2.75,
        "breakdown_score_min": 1.0,
        "risk_off_min": -0.25,
        "tqqq_multiplier": 0.90,
        "cash_buffer": 0.00,
    },
    "warning": {
        "divergence_min": 3.00,
        "breakdown_score_min": 2.0,
        "risk_off_min": 0.00,
        "tqqq_multiplier": 0.65,
        "cash_buffer": 0.10,
    },
    "danger": {
        "divergence_min": 3.25,
        "breakdown_score_min": 3.0,
        "risk_off_min": 0.25,
        "tqqq_multiplier": 0.35,
        "cash_buffer": 0.25,
    },
}

# Component thresholds used to calculate breakdown_score.
# Example: SOXX 5-day return below -3% counts as one breakdown point.
breakdown_component_thresholds = {
    "soxx_5d_max": -0.030,
    "soxx_10d_max": -0.045,
    "soxx_dd_21_max": -0.060,
    "qqqm_5d_max": -0.020,
    "qqqm_10d_max": -0.035,
    "qqqm_dd_21_max": -0.045,
}

# TQQQ overlay only applies when QQQM is the top signal asset.
tiered_tqqq_rule = {
    "moderate": {
        "gap_min": 0.003,
        "top_score_min": 0.010,
        "growth_min": 0.30,
        "soxx_min": 0.30,
        "risk_off_max": 1.00,
        "vix_max": 25.0,
        "replace_fraction": 0.50,
    },
    "strong": {
        "gap_min": 0.004,
        "top_score_min": 0.010,
        "growth_min": 0.20,
        "soxx_min": 0.20,
        "risk_off_max": 0.75,
        "vix_max": 22.0,
        "replace_fraction": 1.00,
    },
}

# ============================================================
# 2. PERFORMANCE FUNCTIONS
# ============================================================
def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    cumulative = (1 + returns).prod()
    years = len(returns) / 252
    if years <= 0:
        return np.nan
    return cumulative ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - rf_annual) / ann_vol


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return drawdown.min()


def compute_turnover(old_weights: dict, new_weights: dict, universe: list) -> float:
    old_vec = np.array([old_weights.get(a, 0.0) for a in universe], dtype=float)
    new_vec = np.array([new_weights.get(a, 0.0) for a in universe], dtype=float)
    return np.abs(new_vec - old_vec).sum()


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std.replace(0, np.nan)
    z = z.clip(-3, 3)
    return z.fillna(0.0)


def performance_summary(name: str, returns: pd.Series, avg_turnover: float) -> dict:
    return {
        "model": name,
        "annual_return": annualized_return(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_rate_annual),
        "max_drawdown": max_drawdown(returns),
        "avg_turnover": avg_turnover,
        "start": returns.dropna().index.min(),
        "end": returns.dropna().index.max(),
        "days": len(returns.dropna()),
    }

# ============================================================
# 3. DATA DOWNLOAD
# ============================================================
def download_close_data(tickers, start, end=None):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.sort_index()


print("Downloading price data...")
prices = download_close_data(all_assets, start_date, end_date).dropna(how="all")
if prices.empty:
    raise ValueError("No price data downloaded.")

print("Downloading macro proxies...")
macro_tickers = {
    "short_rate": "^IRX",
    "long_rate": "^TNX",
    "oil": "CL=F",
    "usd": "DX-Y.NYB",
    "vix_level": "^VIX",
    "copper": "HG=F",
    # RMB feature candidate:
    # CNH=X is Yahoo Finance USD/CNH. If USD/CNH falls, RMB strengthens.
    "usd_cnh": "CNH=X",
    # Fallback / comparison: onshore USD/CNY. Usually more managed than CNH.
    "usd_cny": "CNY=X",
}
macro_raw = download_close_data(list(macro_tickers.values()), start_date, end_date)
macro_raw = macro_raw.rename(columns={v: k for k, v in macro_tickers.items()})
macro_raw = macro_raw.reindex(prices.index).ffill()

required_cols = all_assets
prices = prices.dropna(subset=required_cols)
macro_raw = macro_raw.reindex(prices.index).ffill()

print(f"Latest available data date: {prices.index[-1].date()}")

# ============================================================
# 4. COMMON SERIES / MACRO FEATURES
# ============================================================
asset_returns = prices[UPGRADED_SECTOR_ETFS + [cash_etf, spy_etf, "TQQQ", "ERX", "UXI"]].pct_change()

spy = prices[spy_etf]
spy_ret_1m = spy.pct_change(21)
spy_ret_3m = spy.pct_change(63)
spy_ret_6m = spy.pct_change(126)
spy_vol_1m = spy.pct_change().rolling(21).std() * np.sqrt(252)
spy_vol_3m = spy.pct_change().rolling(63).std() * np.sqrt(252)

short_rate = macro_raw["short_rate"] / 100.0
long_rate = macro_raw["long_rate"] / 100.0
yield_curve = long_rate - short_rate

oil_1m = macro_raw["oil"].pct_change(21)
oil_3m = macro_raw["oil"].pct_change(63)

usd_1m = macro_raw["usd"].pct_change(21)
usd_3m = macro_raw["usd"].pct_change(63)
usd_6m = macro_raw["usd"].pct_change(126)
usd_level_strength = rolling_zscore(macro_raw["usd"], zscore_window)
usd_1m_strength = rolling_zscore(usd_1m, zscore_window)
usd_3m_strength = rolling_zscore(usd_3m, zscore_window)

# ============================================================
# RMB / CHINA LIQUIDITY FEATURE CANDIDATE
# ============================================================
# Yahoo CNH=X is USD/CNH.
# If USD/CNH rises, RMB weakens.
# If USD/CNH falls, RMB strengthens.
# Therefore RMB strength = negative USD/CNH momentum.
if "usd_cnh" in macro_raw.columns and macro_raw["usd_cnh"].notna().sum() > 100:
    usd_cnh = macro_raw["usd_cnh"].copy()
elif "usd_cny" in macro_raw.columns and macro_raw["usd_cny"].notna().sum() > 100:
    print("WARNING: CNH=X not available. Falling back to CNY=X.")
    usd_cnh = macro_raw["usd_cny"].copy()
else:
    print("WARNING: CNH/CNY data unavailable. RMB features will be zero.")
    usd_cnh = pd.Series(index=macro_raw.index, data=np.nan)

usd_cnh = usd_cnh.reindex(prices.index).ffill()
usd_cnh_1m = usd_cnh.pct_change(21)
usd_cnh_3m = usd_cnh.pct_change(63)
usd_cnh_6m = usd_cnh.pct_change(126)

# Positive values mean RMB strength, not USD strength.
rmb_1m = -usd_cnh_1m
rmb_3m = -usd_cnh_3m
rmb_6m = -usd_cnh_6m
rmb_strength = rolling_zscore(rmb_3m, zscore_window)
rmb_1m_strength = rolling_zscore(rmb_1m, zscore_window)
rmb_3m_strength = rolling_zscore(rmb_3m, zscore_window)

# China/EM liquidity: RMB strength helps, DXY strength hurts.
china_em_liquidity_strength = (
    0.60 * rmb_3m_strength
    - 0.40 * usd_3m_strength
).fillna(0.0).clip(-3, 3)

vix_level = macro_raw["vix_level"]
vix_1m = macro_raw["vix_level"].pct_change(21)

copper_1m = macro_raw["copper"].pct_change(21)
copper_3m = macro_raw["copper"].pct_change(63)
copper_rel_spy_1m = copper_1m - spy_ret_1m
copper_rel_spy_3m = copper_3m - spy_ret_3m

ita_1m = prices["ITA"].pct_change(21)
ita_rel_spy_1m = ita_1m - spy_ret_1m

soxx_1m = prices["SOXX"].pct_change(21)
soxx_3m = prices["SOXX"].pct_change(63)
soxx_rel_spy_1m = soxx_1m - spy_ret_1m

# Short-term breakdown indicators for the conditional divergence trigger.
# These are NOT used to reduce TQQQ unless divergence is already high.
soxx_5d = prices["SOXX"].pct_change(5)
soxx_10d = prices["SOXX"].pct_change(10)
soxx_dd_21 = prices["SOXX"] / prices["SOXX"].rolling(21).max() - 1.0
qqqm_5d = prices["QQQM"].pct_change(5)
qqqm_10d = prices["QQQM"].pct_change(10)
qqqm_dd_21 = prices["QQQM"] / prices["QQQM"].rolling(21).max() - 1.0

hyg_1m = prices["HYG"].pct_change(21)
hyg_3m = prices["HYG"].pct_change(63)
hyg_6m = prices["HYG"].pct_change(126)
hyg_rel_spy_1m = hyg_1m - spy_ret_1m
hyg_rel_spy_3m = hyg_3m - spy_ret_3m

qqqm_rel_spy_1m = prices["QQQM"].pct_change(21) - spy_ret_1m
xli_rel_spy_1m = prices["XLI"].pct_change(21) - spy_ret_1m
xlb_rel_spy_1m = prices["XLB"].pct_change(21) - spy_ret_1m

# Continuous regime strengths
ita_strength = rolling_zscore(ita_rel_spy_1m, zscore_window)
soxx_strength = rolling_zscore(soxx_rel_spy_1m, zscore_window)
qqqm_strength = rolling_zscore(qqqm_rel_spy_1m, zscore_window)
xli_strength = rolling_zscore(xli_rel_spy_1m, zscore_window)
xlb_strength = rolling_zscore(xlb_rel_spy_1m, zscore_window)
oil_strength = rolling_zscore(oil_1m, zscore_window)
vix_strength = rolling_zscore(vix_1m, zscore_window)
copper_strength = rolling_zscore(copper_rel_spy_1m, zscore_window)
copper_3m_strength = rolling_zscore(copper_rel_spy_3m, zscore_window)
hyg_strength = rolling_zscore(hyg_rel_spy_1m, zscore_window)

war_strength = ((ita_strength + oil_strength) / 2.0).fillna(0.0)
growth_strength = ((qqqm_strength + soxx_strength) / 2.0).fillna(0.0)
risk_off_strength = ((-qqqm_strength + vix_strength) / 2.0).fillna(0.0)
credit_strength = ((hyg_strength - vix_strength) / 2.0).fillna(0.0)

# New: industrial/materials score.
# Idea:
#   copper = raw materials demand
#   XLB relative strength = materials confirmation
#   XLI relative strength = industrial/capex confirmation
#   credit strength = risk-on confirmation
industrial_strength = (
    0.40 * copper_strength
    + 0.25 * xlb_strength
    + 0.20 * xli_strength
    + 0.15 * credit_strength
).fillna(0.0).clip(-3, 3)

materials_strength = (
    0.60 * copper_strength
    + 0.25 * copper_3m_strength
    + 0.15 * xlb_strength
).fillna(0.0).clip(-3, 3)

# Divergence detector:
#   Positive = tech/semis are strong, but industrial/copper/materials confirmation is weak.
#   It does not force an exit by itself. It becomes dangerous when risk-off also rises.
tech_real_economy_divergence = (
    0.50 * soxx_strength
    + 0.50 * qqqm_strength
    - 0.35 * industrial_strength
    - 0.35 * materials_strength
    - 0.30 * copper_strength
).fillna(0.0).clip(-5, 5)

crash_pressure = (
    0.60 * tech_real_economy_divergence
    + 0.40 * risk_off_strength
).fillna(0.0).clip(-5, 5)

# ============================================================
# CONDITIONAL RMB ACTIVATION LOGIC
# ============================================================
# Purpose:
#   The previous unconditional RMB test showed RMB/China-EM information was real,
#   but it reduced performance during AI-dominant regimes.
#
#   This version only lets RMB matter when the market is showing evidence of
#   reflation / industrial broadening. This tests whether RMB helps in the
#   regimes where professionals would actually expect it to matter.
#
# Activation idea:
#   RMB signal ON when:
#     1) industrial/materials/copper confirmation is improving
#     2) risk-off is not dominant
#     3) SOXX is not so dominant that AI overwhelms the macro signal
#
# Values are continuous, not only 0/1, to avoid brittle threshold behavior.
industrial_broadening_gate = (
    0.40 * (industrial_strength > 0.25).astype(float)
    + 0.25 * (materials_strength > 0.25).astype(float)
    + 0.25 * (copper_strength > 0.25).astype(float)
    + 0.10 * (credit_strength > -0.50).astype(float)
).fillna(0.0)

# Penalize if AI/SOXX dominance is extremely strong or risk-off is rising.
# In that environment the previous test showed RMB can dilute the TQQQ engine.
rmb_activation_penalty = (
    0.50 * (soxx_strength > 2.00).astype(float)
    + 0.50 * (risk_off_strength > 0.75).astype(float)
).fillna(0.0)

rmb_activation_gate = (industrial_broadening_gate - rmb_activation_penalty).clip(0.0, 1.0)

# This is the actual conditional RMB signal used by the challenger.
conditional_china_em_liquidity_strength = (
    china_em_liquidity_strength * rmb_activation_gate
).fillna(0.0).clip(-3, 3)

conditional_rmb_strength = (
    rmb_strength * rmb_activation_gate
).fillna(0.0).clip(-3, 3)

conditional_rmb_3m_strength = (
    rmb_3m_strength * rmb_activation_gate
).fillna(0.0).clip(-3, 3)

# ============================================================
# 5. FEATURE BUILDER
# ============================================================
def build_features_by_asset(sector_etfs: list, include_rmb_features: bool = False):
    features_by_asset = {}

    for asset in sector_etfs:
        px = prices[asset]

        ret_1m = px.pct_change(21)
        ret_3m = px.pct_change(63)
        ret_6m = px.pct_change(126)
        ret_12m = px.pct_change(252)
        rel_6m_vs_spy = ret_6m - spy_ret_6m
        vol_1m = px.pct_change().rolling(21).std() * np.sqrt(252)
        vol_3m = px.pct_change().rolling(63).std() * np.sqrt(252)

        is_QQQM = 1 if asset == "QQQM" else 0
        is_XLE = 1 if asset == "XLE" else 0
        is_XSOE = 1 if asset == "XSOE" else 0
        is_XLI = 1 if asset == "XLI" else 0
        is_XLB = 1 if asset == "XLB" else 0

        df = pd.DataFrame({
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "rel_6m_vs_spy": rel_6m_vs_spy,

            "spy_ret_1m": spy_ret_1m,
            "spy_ret_3m": spy_ret_3m,
            "spy_ret_6m": spy_ret_6m,
            "spy_vol_1m": spy_vol_1m,
            "spy_vol_3m": spy_vol_3m,

            "vol_1m": vol_1m,
            "vol_3m": vol_3m,

            "short_rate": short_rate,
            "long_rate": long_rate,
            "yield_curve": yield_curve,

            "oil_1m": oil_1m,
            "oil_3m": oil_3m,

            "usd_1m": usd_1m,
            "usd_3m": usd_3m,
            "usd_6m": usd_6m,
            "usd_level_strength": usd_level_strength,
            "usd_1m_strength": usd_1m_strength,
            "usd_3m_strength": usd_3m_strength,

            "vix_level": vix_level,
            "vix_1m": vix_1m,

            "copper_1m": copper_1m,
            "copper_3m": copper_3m,
            "copper_rel_spy_1m": copper_rel_spy_1m,
            "copper_rel_spy_3m": copper_rel_spy_3m,
            "copper_strength": copper_strength,
            "copper_3m_strength": copper_3m_strength,

            "ita_1m": ita_1m,
            "ita_rel_spy_1m": ita_rel_spy_1m,

            "soxx_1m": soxx_1m,
            "soxx_3m": soxx_3m,
            "soxx_rel_spy_1m": soxx_rel_spy_1m,
            "soxx_5d": soxx_5d,
            "soxx_10d": soxx_10d,
            "soxx_dd_21": soxx_dd_21,
            "qqqm_5d": qqqm_5d,
            "qqqm_10d": qqqm_10d,
            "qqqm_dd_21": qqqm_dd_21,

            "hyg_1m": hyg_1m,
            "hyg_3m": hyg_3m,
            "hyg_6m": hyg_6m,
            "hyg_rel_spy_1m": hyg_rel_spy_1m,
            "hyg_rel_spy_3m": hyg_rel_spy_3m,
            "hyg_strength": hyg_strength,
            "credit_strength": credit_strength,

            "qqqm_rel_spy_1m": qqqm_rel_spy_1m,
            "xli_rel_spy_1m": xli_rel_spy_1m,
            "xlb_rel_spy_1m": xlb_rel_spy_1m,

            "war_strength": war_strength,
            "growth_strength": growth_strength,
            "risk_off_strength": risk_off_strength,
            "industrial_strength": industrial_strength,
            "materials_strength": materials_strength,
            "tech_real_economy_divergence": tech_real_economy_divergence,
            "crash_pressure": crash_pressure,

            "is_QQQM": is_QQQM,
            "is_XLE": is_XLE,
            "is_XSOE": is_XSOE,
            "is_XLI": is_XLI,
            "is_XLB": is_XLB,

            # Asset-specific interactions.
            "yield_curve_QQQM": yield_curve * is_QQQM,

            "usd_XSOE": usd_1m * is_XSOE,
            "usd_3m_XSOE": usd_3m * is_XSOE,
            "usd_6m_XSOE": usd_6m * is_XSOE,
            "usd_level_XSOE": usd_level_strength * is_XSOE,
            "usd_1m_strength_XSOE": usd_1m_strength * is_XSOE,
            "usd_3m_strength_XSOE": usd_3m_strength * is_XSOE,

            "copper_XSOE": copper_strength * is_XSOE,
            "copper_XLB": copper_strength * is_XLB,
            "copper_3m_XLB": copper_3m_strength * is_XLB,
            "copper_XLI": copper_strength * is_XLI,

            "industrial_XLI": industrial_strength * is_XLI,
            "materials_XLB": materials_strength * is_XLB,
            "growth_XLI": growth_strength * is_XLI,

            "hyg_QQQM": hyg_strength * is_QQQM,
            "hyg_XSOE": hyg_strength * is_XSOE,
            "hyg_XLI": hyg_strength * is_XLI,
            "credit_QQQM": credit_strength * is_QQQM,
            "credit_XSOE": credit_strength * is_XSOE,
            "credit_XLI": credit_strength * is_XLI,

            "war_XLE": war_strength * is_XLE,
            "growth_QQQM": growth_strength * is_QQQM,
            "risk_off_QQQM": risk_off_strength * is_QQQM,
            "divergence_QQQM": tech_real_economy_divergence * is_QQQM,
            "crash_pressure_QQQM": crash_pressure * is_QQQM,
            "risk_off_XSOE": risk_off_strength * is_XSOE,
            "risk_off_XLI": risk_off_strength * is_XLI,
            "risk_off_XLB": risk_off_strength * is_XLB,
        })

        if include_rmb_features:
            # CONDITIONAL RMB / China-EM liquidity features.
            # Baseline remains unchanged. The challenger receives RMB information only
            # after it is gated by industrial/copper/materials broadening.
            df["rmb_activation_gate"] = rmb_activation_gate
            df["industrial_broadening_gate"] = industrial_broadening_gate
            df["conditional_rmb_strength"] = conditional_rmb_strength
            df["conditional_rmb_3m_strength"] = conditional_rmb_3m_strength
            df["conditional_china_em_liquidity_strength"] = conditional_china_em_liquidity_strength

            # Asset-specific interaction terms.
            # XSOE should be most sensitive; XLB/XLI may benefit in reflation regimes.
            df["conditional_rmb_XSOE"] = conditional_rmb_strength * is_XSOE
            df["conditional_rmb_3m_XSOE"] = conditional_rmb_3m_strength * is_XSOE
            df["conditional_china_em_liquidity_XSOE"] = conditional_china_em_liquidity_strength * is_XSOE
            df["conditional_rmb_XLB"] = conditional_rmb_strength * is_XLB
            df["conditional_rmb_XLI"] = conditional_rmb_strength * is_XLI

        df["target"] = px.shift(-forward_return_days) / px - 1.0
        features_by_asset[asset] = df

    return features_by_asset

# ============================================================
# 6. TRAINING HELPERS
# ============================================================
def build_train_data(features_by_asset, asset_list, end_loc, train_window):
    start_loc = end_loc - train_window
    if start_loc < 0:
        return None, None

    x_parts = []
    y_parts = []

    for asset in asset_list:
        df = features_by_asset[asset].iloc[start_loc:end_loc].copy().dropna()
        if df.empty:
            continue
        x_parts.append(df.drop(columns=["target"]))
        y_parts.append(df["target"])

    if len(x_parts) == 0:
        return None, None

    x_train = pd.concat(x_parts, axis=0)
    y_train = pd.concat(y_parts, axis=0)

    common_idx = x_train.index.intersection(y_train.index)
    x_train = x_train.loc[common_idx]
    y_train = y_train.loc[common_idx]

    if len(x_train) == 0:
        return None, None
    return x_train, y_train


def get_today_features(features_by_asset, asset: str, date: pd.Timestamp):
    row = features_by_asset[asset].loc[[date]].drop(columns=["target"], errors="ignore")
    if row.empty:
        return None
    if row.isna().any(axis=1).iloc[0]:
        return None
    return row

# ============================================================
# 7. OVERLAY / ALLOCATION LOGIC
# ============================================================
def apply_regime_overlay(raw_preds: dict, date: pd.Timestamp, sector_etfs: list, overlay_style: str = "v1"):
    def val(series, default=0.0):
        if date in series.index and pd.notna(series.loc[date]):
            return float(series.loc[date])
        return default

    war = val(war_strength)
    growth = val(growth_strength)
    risk_off = val(risk_off_strength)
    soxx = val(soxx_strength)
    copper = val(copper_strength)
    copper3 = val(copper_3m_strength)
    industrial = val(industrial_strength)
    materials = val(materials_strength)
    usd_regime = val(usd_3m_strength)
    hyg_regime = val(hyg_strength)
    credit_regime = val(credit_strength)
    divergence = val(tech_real_economy_divergence)
    crash = val(crash_pressure)

    # Conditional SOXX/QQQM breakdown data.
    soxx_5d_now = val(soxx_5d)
    soxx_10d_now = val(soxx_10d)
    soxx_dd_21_now = val(soxx_dd_21)
    qqqm_5d_now = val(qqqm_5d)
    qqqm_10d_now = val(qqqm_10d)
    qqqm_dd_21_now = val(qqqm_dd_21)

    breakdown_score = 0.0
    breakdown_score += 1.0 if soxx_5d_now <= breakdown_component_thresholds["soxx_5d_max"] else 0.0
    breakdown_score += 1.0 if soxx_10d_now <= breakdown_component_thresholds["soxx_10d_max"] else 0.0
    breakdown_score += 1.0 if soxx_dd_21_now <= breakdown_component_thresholds["soxx_dd_21_max"] else 0.0
    breakdown_score += 1.0 if qqqm_5d_now <= breakdown_component_thresholds["qqqm_5d_max"] else 0.0
    breakdown_score += 1.0 if qqqm_10d_now <= breakdown_component_thresholds["qqqm_10d_max"] else 0.0
    breakdown_score += 1.0 if qqqm_dd_21_now <= breakdown_component_thresholds["qqqm_dd_21_max"] else 0.0

    adjusted = raw_preds.copy()

    war_pos = max(0.0, war)
    growth_pos = max(0.0, growth)
    risk_off_pos = max(0.0, risk_off)
    soxx_pos = max(0.0, soxx)
    copper_pos = max(0.0, copper)
    copper3_pos = max(0.0, copper3)
    industrial_pos = max(0.0, industrial)
    materials_pos = max(0.0, materials)
    usd_regime_pos = max(0.0, usd_regime)
    hyg_pos = max(0.0, hyg_regime)
    credit_pos = max(0.0, credit_regime)
    scale = overlay_scale

    def add(asset, amount):
        if asset in adjusted:
            adjusted[asset] += amount

    # Existing logic preserved.
    if war_pos > 0:
        add("XLE", scale * war_pos)
        add("QQQM", -scale * 0.5 * war_pos)
        add("XSOE", -scale * 0.3 * war_pos)

    if growth_pos > 0:
        add("QQQM", scale * growth_pos)
        add("XLE", -scale * 0.4 * growth_pos)

    if soxx_pos > 0:
        add("QQQM", scale * 0.8 * soxx_pos)

    if copper_pos > 0:
        add("XSOE", scale * 0.8 * copper_pos)

    if usd_regime_pos > 0:
        add("XSOE", -scale * 0.6 * usd_regime_pos)

    if hyg_pos > 0:
        add("QQQM", scale * 0.35 * hyg_pos)
        add("XSOE", scale * 0.45 * hyg_pos)

    if credit_pos > 0:
        add("QQQM", scale * 0.25 * credit_pos)
        add("XSOE", scale * 0.35 * credit_pos)

    if risk_off_pos > 0:
        add("QQQM", -scale * 1.2 * risk_off_pos)
        add("XSOE", -scale * 1.0 * risk_off_pos)

    if war_pos > 0 and risk_off_pos > 1.0:
        add("XLE", -scale * 0.4 * risk_off_pos)

    # New XLI/XLB overlay logic.
    # V1 = original broad boost; V2 = stricter regime classifier.
    # Keep overlays modest because ML already sees the features.
    if overlay_style == "v1":
        # XLB: early industrial/materials/copper cycle.
        if "XLB" in sector_etfs:
            if copper_pos > 0:
                add("XLB", scale * 0.90 * copper_pos)
            if copper3_pos > 0:
                add("XLB", scale * 0.35 * copper3_pos)
            if materials_pos > 0:
                add("XLB", scale * 0.50 * materials_pos)
            if risk_off_pos > 0:
                add("XLB", -scale * 0.70 * risk_off_pos)

        # XLI: industrial/reshoring/capex cycle; likes industrial acceleration and credit support.
        if "XLI" in sector_etfs:
            if industrial_pos > 0:
                add("XLI", scale * 0.80 * industrial_pos)
            if growth_pos > 0:
                add("XLI", scale * 0.35 * growth_pos)
            if credit_pos > 0:
                add("XLI", scale * 0.25 * credit_pos)
            if risk_off_pos > 0:
                add("XLI", -scale * 0.80 * risk_off_pos)

    elif overlay_style in ("v2", "hybrid"):
        # V2 philosophy:
        # - XLB should be boosted only when materials/copper strength is confirmed.
        # - XLI should be boosted only when industrial strength is positive AND risk-off is not dominant.
        # - Strong USD/risk-off gets a small penalty because it often hurts global cyclicals/materials.
        industrial_regime_on = (industrial > 0.25) and (risk_off < 0.75) and (usd_regime < 1.50)
        materials_regime_on = (materials > 0.25) and (risk_off < 1.00)
        early_cycle_on = (copper3 > 0.50) and (credit_regime > -0.50) and (risk_off < 1.00)

        if "XLB" in sector_etfs:
            if materials_regime_on:
                add("XLB", scale * 0.70 * materials_pos)
            if early_cycle_on:
                add("XLB", scale * 0.30 * copper3_pos)
            if risk_off_pos > 0:
                add("XLB", -scale * 0.85 * risk_off_pos)
            if usd_regime > 1.0:
                add("XLB", -scale * 0.20 * usd_regime_pos)

        if "XLI" in sector_etfs:
            if industrial_regime_on:
                add("XLI", scale * 0.75 * industrial_pos)
                if growth_pos > 0:
                    add("XLI", scale * 0.20 * growth_pos)
                if credit_pos > 0:
                    add("XLI", scale * 0.20 * credit_pos)
            if risk_off_pos > 0:
                add("XLI", -scale * 0.90 * risk_off_pos)
            if usd_regime > 1.5:
                add("XLI", -scale * 0.15 * usd_regime_pos)

        # HYBRID extra: preserve fast-growth tech/TQQQ engine when semis + QQQM leadership are very strong.
        # This prevents XLI/XLB from diluting the original tech engine unless industrial/materials signals are truly active.
        if overlay_style == "hybrid":
            tech_regime_on = (growth > 1.00) and (soxx > 1.00) and (risk_off < 0.50)
            industrial_regime_on_h = (industrial > 0.50) and (copper3 > 0.25) and (risk_off < 0.75) and (usd_regime < 1.25)
            materials_regime_on_h = (materials > 0.50) and (copper3 > 0.50) and (risk_off < 0.90)
            if tech_regime_on:
                add("QQQM", scale * 0.45 * min(growth_pos + soxx_pos, 6.0))
                if "XLI" in sector_etfs and not industrial_regime_on_h:
                    add("XLI", -scale * 0.25 * max(0.0, -industrial))
                if "XLB" in sector_etfs and not materials_regime_on_h:
                    add("XLB", -scale * 0.25 * max(0.0, -materials))

            # NEW: Non-tech conviction boost.
            # Purpose: reduce tech bias when oil/industrial/materials regimes are truly strong.
            oil_regime = war
            if oil_regime > 0.5:
                add("XLE", scale * 1.5 * oil_regime)
            if industrial > 0.5:
                add("XLI", scale * 1.2 * industrial)
            if copper > 0.7:
                add("XLB", scale * 1.0 * copper)
            if oil_regime > 0.5 or industrial > 0.5:
                add("QQQM", -scale * 0.8 * max(oil_regime, industrial))

    else:
        raise ValueError(f"Unknown overlay_style: {overlay_style}")

    overlay_info = {
        "war_strength": war,
        "growth_strength": growth,
        "risk_off_strength": risk_off,
        "soxx_strength": soxx,
        "copper_strength": copper,
        "copper_3m_strength": copper3,
        "industrial_strength": industrial,
        "materials_strength": materials,
        "usd_3m_strength": usd_regime,
        "hyg_strength": hyg_regime,
        "credit_strength": credit_regime,
        "tech_real_economy_divergence": divergence,
        "crash_pressure": crash,
        "soxx_5d": soxx_5d_now,
        "soxx_10d": soxx_10d_now,
        "soxx_dd_21": soxx_dd_21_now,
        "qqqm_5d": qqqm_5d_now,
        "qqqm_10d": qqqm_10d_now,
        "qqqm_dd_21": qqqm_dd_21_now,
        "breakdown_score": breakdown_score,
        "overlay_style": overlay_style,
    }
    return adjusted, overlay_info


def should_go_cash(top_score: float, second_score: float, risk_off_strength_val: float):
    threshold = 0.0
    if risk_off_strength_val > 1.0:
        threshold += risk_off_cash_threshold
    return (top_score < threshold) and (second_score < threshold)


def get_conviction_weights(top_score: float, second_score: float):
    gap = top_score - second_score
    if gap < 0.005:
        w_top = 0.60
    elif gap < 0.015:
        w_top = 0.70
    elif gap < 0.030:
        w_top = 0.80
    elif gap < 0.050:
        w_top = 0.90
    else:
        w_top = 1.00
    return w_top, 1.0 - w_top, gap


def tqqq_replace_fraction(top_asset: str, top_score: float, second_score: float, overlay_info: dict, date: pd.Timestamp):
    if top_asset != "QQQM":
        return 0.0

    gap = top_score - second_score
    vix_now = float(vix_level.loc[date]) if date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan
    growth = overlay_info["growth_strength"]
    soxx = overlay_info["soxx_strength"]
    risk_off = overlay_info["risk_off_strength"]

    strong = (
        gap >= tiered_tqqq_rule["strong"]["gap_min"]
        and top_score >= tiered_tqqq_rule["strong"]["top_score_min"]
        and growth >= tiered_tqqq_rule["strong"]["growth_min"]
        and soxx >= tiered_tqqq_rule["strong"]["soxx_min"]
        and risk_off <= tiered_tqqq_rule["strong"]["risk_off_max"]
        and (pd.isna(vix_now) or vix_now <= tiered_tqqq_rule["strong"]["vix_max"])
    )
    if strong:
        return tiered_tqqq_rule["strong"]["replace_fraction"]

    moderate = (
        gap >= tiered_tqqq_rule["moderate"]["gap_min"]
        and top_score >= tiered_tqqq_rule["moderate"]["top_score_min"]
        and growth >= tiered_tqqq_rule["moderate"]["growth_min"]
        and soxx >= tiered_tqqq_rule["moderate"]["soxx_min"]
        and risk_off <= tiered_tqqq_rule["moderate"]["risk_off_max"]
        and (pd.isna(vix_now) or vix_now <= tiered_tqqq_rule["moderate"]["vix_max"])
    )
    if moderate:
        return tiered_tqqq_rule["moderate"]["replace_fraction"]

    return 0.0



def tqqq_dynamic_replace_fraction(top_asset: str, top_score: float, second_score: float, overlay_info: dict, date: pd.Timestamp):
    """Conviction + volatility adjusted TQQQ replacement fraction."""
    if top_asset != "QQQM":
        return 0.0

    gap = top_score - second_score
    growth = float(overlay_info.get("growth_strength", 0.0))
    soxx = float(overlay_info.get("soxx_strength", 0.0))
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    vix_now = float(vix_level.loc[date]) if date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan

    permission = (
        top_score >= 0.008
        and gap >= 0.002
        and growth >= 0.20
        and soxx >= 0.20
        and risk_off <= 1.00
        and (pd.isna(vix_now) or vix_now <= 25.0)
    )
    if not permission:
        return 0.0

    conviction = (gap - 0.002) / (0.020 - 0.002)
    conviction = float(np.clip(conviction, 0.0, 1.0))
    replace_fraction = 0.40 + 0.60 * conviction

    if pd.isna(vix_now):
        vol_adj = 1.0
    elif vix_now < 15:
        vol_adj = 1.1
    elif vix_now < 25:
        vol_adj = 1.0
    elif vix_now < 35:
        vol_adj = 0.8
    else:
        vol_adj = 0.6
    replace_fraction *= vol_adj

    if growth > 1.0 and soxx > 1.0 and risk_off < 0.50:
        replace_fraction += 0.10

    if risk_off > 0.50:
        replace_fraction *= 0.75

    return float(np.clip(replace_fraction, 0.0, 1.0))


def multi_asset_leverage_fraction(asset: str, top_asset: str, score_gap: float, overlay_info: dict, date: pd.Timestamp):
    """Safe 2x leverage gate for XLE->ERX and XLI->UXI."""
    if asset != top_asset:
        return 0.0

    vix_now = float(vix_level.loc[date]) if date is not None and date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))

    if score_gap > 0.015:
        frac = 0.50
    elif score_gap > 0.006:
        frac = 0.25
    else:
        frac = 0.0

    if not pd.isna(vix_now):
        if vix_now > 30:
            frac *= 0.3
        elif vix_now > 25:
            frac *= 0.6

    if risk_off > 0.5:
        frac *= 0.5

    return float(np.clip(frac, 0.0, 0.60))


def conditional_breakdown_defense_level(overlay_info: dict) -> str:
    """
    Conditional trigger:
    - Divergence alone does nothing.
    - Defense activates only when SOXX/QQQM actually starts breaking.
    """
    if not use_conditional_breakdown_defense:
        return "off"

    divergence = float(overlay_info.get("tech_real_economy_divergence", 0.0))
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    breakdown_score = float(overlay_info.get("breakdown_score", 0.0))

    for level in ["danger", "warning", "watch"]:
        rule = conditional_breakdown_rules[level]
        if (
            divergence >= rule["divergence_min"]
            and breakdown_score >= rule["breakdown_score_min"]
            and risk_off >= rule["risk_off_min"]
        ):
            return level
    return "normal"


def apply_conditional_breakdown_defense(exec_weights: dict, overlay_info: dict) -> dict:
    """
    Reduce TQQQ only after tech leadership starts failing.
    Freed TQQQ exposure is first moved to QQQM; optional cash buffer then scales down risk.
    """
    level = conditional_breakdown_defense_level(overlay_info)
    if level in ("off", "normal"):
        return exec_weights

    rule = conditional_breakdown_rules[level]
    tqqq_multiplier = rule["tqqq_multiplier"]
    cash_buffer = rule["cash_buffer"]

    old_tqqq = exec_weights.get("TQQQ", 0.0)
    new_tqqq = old_tqqq * tqqq_multiplier
    freed_from_tqqq = old_tqqq - new_tqqq

    exec_weights["TQQQ"] = new_tqqq
    exec_weights["QQQM"] = exec_weights.get("QQQM", 0.0) + freed_from_tqqq

    if cash_buffer > 0:
        for asset in list(exec_weights.keys()):
            if asset != cash_etf:
                exec_weights[asset] *= (1.0 - cash_buffer)
        exec_weights[cash_etf] = exec_weights.get(cash_etf, 0.0) + cash_buffer

    total = sum(exec_weights.values())
    if total > 0 and abs(total - 1.0) > 1e-8:
        for asset in exec_weights:
            exec_weights[asset] /= total

    return exec_weights


def build_execution_weights(
    signal_weights: dict,
    overlay_fraction: float,
    sector_etfs: list,
    top_asset=None,
    score_gap=0.0,
    overlay_info=None,
    date=None,
):
    exec_universe = ["TQQQ", "ERX", "UXI"] + sector_etfs + [cash_etf]
    exec_weights = {a: 0.0 for a in exec_universe}
    overlay_info = overlay_info or {}

    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    vix_now = float(vix_level.loc[date]) if date is not None and date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan

    if risk_off > 1.5 or (not pd.isna(vix_now) and vix_now > 32):
        exec_weights[cash_etf] = 1.0
        return exec_weights

    if risk_off > 1.0 or (not pd.isna(vix_now) and vix_now > 28):
        defensive_cash = 0.50
    else:
        defensive_cash = 0.0

    qqqm_signal = signal_weights.get("QQQM", 0.0)
    exec_weights["TQQQ"] = qqqm_signal * overlay_fraction
    exec_weights["QQQM"] = qqqm_signal * (1.0 - overlay_fraction)

    xle_signal = signal_weights.get("XLE", 0.0)
    if xle_signal > 0:
        frac = multi_asset_leverage_fraction("XLE", top_asset, score_gap, overlay_info, date)
        exec_weights["ERX"] = xle_signal * frac
        exec_weights["XLE"] = xle_signal * (1.0 - frac)

    xli_signal = signal_weights.get("XLI", 0.0)
    if xli_signal > 0:
        frac = multi_asset_leverage_fraction("XLI", top_asset, score_gap, overlay_info, date)
        exec_weights["UXI"] = xli_signal * frac
        exec_weights["XLI"] = xli_signal * (1.0 - frac)

    for asset in sector_etfs:
        if asset not in ["QQQM", "XLE", "XLI"]:
            exec_weights[asset] = signal_weights.get(asset, 0.0)

    exec_weights[cash_etf] = signal_weights.get(cash_etf, 0.0)

    if defensive_cash > 0:
        for asset in exec_weights:
            if asset != cash_etf:
                exec_weights[asset] *= (1.0 - defensive_cash)
        exec_weights[cash_etf] = defensive_cash

    exec_weights = apply_conditional_breakdown_defense(exec_weights, overlay_info)

    return exec_weights

# ============================================================
# 8. STRATEGY RUNNER
# ============================================================
def run_strategy(model_name: str, sector_etfs: list, features_by_asset: dict, overlay_style: str = "v1", tqqq_style: str = "tiered"):
    signal_universe = sector_etfs + [cash_etf]
    exec_universe = ["TQQQ", "ERX", "UXI"] + sector_etfs + [cash_etf]
    dates = prices.index

    min_needed = max(train_window, 252) + 1
    max_loc = len(dates) - forward_return_days
    rebalance_locs = list(range(min_needed, max_loc, rebalance_step))

    portfolio_daily_returns = pd.Series(index=dates, dtype=float)
    current_exec_weights = {a: 0.0 for a in exec_universe}
    current_exec_weights[cash_etf] = 1.0

    rebalance_records = []
    turnover_list = []

    for i, loc in enumerate(rebalance_locs):
        rebalance_date = dates[loc]

        x_train, y_train = build_train_data(features_by_asset, sector_etfs, loc, train_window)
        if x_train is None or len(x_train) < 50:
            continue

        model = RandomForestRegressor(**rf_params)
        model.fit(x_train, y_train)

        raw_preds = {}
        for asset in sector_etfs:
            x_today = get_today_features(features_by_asset, asset, rebalance_date)
            if x_today is None:
                continue
            raw_preds[asset] = float(model.predict(x_today)[0])

        if len(raw_preds) < 2:
            continue

        adjusted_preds, overlay_info = apply_regime_overlay(raw_preds, rebalance_date, sector_etfs, overlay_style=overlay_style)
        ranked = sorted(adjusted_preds.items(), key=lambda x: x[1], reverse=True)
        top_asset, top_score = ranked[0]
        second_asset, second_score = ranked[1]

        w_top, w_second, score_gap = get_conviction_weights(top_score, second_score)

        signal_weights = {a: 0.0 for a in signal_universe}
        signal_weights[top_asset] = w_top
        signal_weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            signal_weights = {a: 0.0 for a in signal_universe}
            signal_weights[cash_etf] = 1.0

        if tqqq_style == "dynamic":
            overlay_fraction = tqqq_dynamic_replace_fraction(
                top_asset, top_score, second_score, overlay_info, rebalance_date
            )
        else:
            overlay_fraction = tqqq_replace_fraction(
                top_asset, top_score, second_score, overlay_info, rebalance_date
            )

        exec_weights = build_execution_weights(
            signal_weights,
            overlay_fraction,
            sector_etfs,
            top_asset=top_asset,
            score_gap=score_gap,
            overlay_info=overlay_info,
            date=rebalance_date,
        )

        overlay_info["conditional_breakdown_defense_level"] = conditional_breakdown_defense_level(overlay_info)

        latest_like = {
            "exec_weights": exec_weights,
            "risk_off_strength": overlay_info.get("risk_off_strength", 0.0),
            "growth_strength": overlay_info.get("growth_strength", 0.0),
            "soxx_strength": overlay_info.get("soxx_strength", 0.0),
            "score_gap": score_gap,
            "top_score": top_score,
        }
        latest_like = apply_v2_continuous_tqqq_alert(latest_like)
        exec_weights = latest_like["exec_weights"]

        overlay_info["v2_tqqq_scale"] = latest_like.get("v2_tqqq_scale", 1.0)
        overlay_info["v2_alert_action"] = latest_like.get("v2_alert_action", "NONE")
        overlay_info["conditional_breakdown_defense_level"] = conditional_breakdown_defense_level(overlay_info)

        turnover = compute_turnover(current_exec_weights, exec_weights, exec_universe)
        turnover_list.append(turnover)

        next_loc = rebalance_locs[i + 1] if i + 1 < len(rebalance_locs) else max_loc
        hold_dates = dates[loc + 1: next_loc + 1]
        if len(hold_dates) == 0:
            continue

        hold_rets = pd.Series(index=hold_dates, data=0.0)
        for asset, w in exec_weights.items():
            if w != 0:
                hold_rets = hold_rets.add(
                    w * asset_returns[asset].reindex(hold_dates).fillna(0.0),
                    fill_value=0.0,
                )

        cost = turnover * transaction_cost
        hold_rets.iloc[0] -= cost
        portfolio_daily_returns.loc[hold_dates] = hold_rets.values
        current_exec_weights = exec_weights.copy()

        row = {
            "model": model_name,
            "date": rebalance_date,
            "top_asset": top_asset,
            "second_asset": second_asset,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "turnover": turnover,
            "tx_cost_applied": cost,
            "overlay_fraction": overlay_fraction,
            "v2_tqqq_scale": overlay_info.get("v2_tqqq_scale", 1.0),
            "v2_alert_action": overlay_info.get("v2_alert_action", "NONE"),
            **overlay_info,
        }

        for a in sector_etfs:
            row[f"raw_pred_{a}"] = raw_preds.get(a, np.nan)
            row[f"adj_pred_{a}"] = adjusted_preds.get(a, np.nan)
            row[f"signal_w_{a}"] = signal_weights.get(a, 0.0)
            row[f"exec_w_{a}"] = exec_weights.get(a, 0.0)

        row[f"signal_w_{cash_etf}"] = signal_weights.get(cash_etf, 0.0)
        row[f"exec_w_{cash_etf}"] = exec_weights.get(cash_etf, 0.0)
        row["exec_w_TQQQ"] = exec_weights.get("TQQQ", 0.0)
        row["exec_w_ERX"] = exec_weights.get("ERX", 0.0)
        row["exec_w_UXI"] = exec_weights.get("UXI", 0.0)

        rebalance_records.append(row)

    portfolio_daily_returns = portfolio_daily_returns.dropna()
    rebalance_df = pd.DataFrame(rebalance_records)
    avg_turnover = float(np.mean(turnover_list)) if turnover_list else np.nan

    return portfolio_daily_returns, rebalance_df, avg_turnover
# ============================================================
# 9. LATEST RECOMMENDATION
# ============================================================
def get_latest_recommendation(model_name: str, sector_etfs: list, features_by_asset: dict, overlay_style: str = "v1", tqqq_style: str = "tiered"):
    signal_universe = sector_etfs + [cash_etf]
    dates = prices.index

    for latest_loc in range(len(dates) - 1, train_window, -1):
        latest_date = dates[latest_loc]

        x_train, y_train = build_train_data(features_by_asset, sector_etfs, latest_loc, train_window)
        if x_train is None or len(x_train) < 50:
            continue

        model = RandomForestRegressor(**rf_params)
        model.fit(x_train, y_train)

        raw_preds = {}
        for asset in sector_etfs:
            x_today = get_today_features(features_by_asset, asset, latest_date)
            if x_today is None:
                continue
            raw_preds[asset] = float(model.predict(x_today)[0])

        if len(raw_preds) < 2:
            continue

        adjusted_preds, overlay_info = apply_regime_overlay(raw_preds, latest_date, sector_etfs, overlay_style=overlay_style)
        ranked = sorted(adjusted_preds.items(), key=lambda x: x[1], reverse=True)
        top_asset, top_score = ranked[0]
        second_asset, second_score = ranked[1]
        w_top, w_second, score_gap = get_conviction_weights(top_score, second_score)

        signal_weights = {a: 0.0 for a in signal_universe}
        signal_weights[top_asset] = w_top
        signal_weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            signal_weights = {a: 0.0 for a in signal_universe}
            signal_weights[cash_etf] = 1.0

        if tqqq_style == "dynamic":
            overlay_fraction = tqqq_dynamic_replace_fraction(top_asset, top_score, second_score, overlay_info, latest_date)
        else:
            overlay_fraction = tqqq_replace_fraction(top_asset, top_score, second_score, overlay_info, latest_date)
        exec_weights = build_execution_weights(
            signal_weights,
            overlay_fraction,
            sector_etfs,
            top_asset=top_asset,
            score_gap=score_gap,
            overlay_info=overlay_info,
            date=latest_date,
        )

        overlay_info["conditional_breakdown_defense_level"] = conditional_breakdown_defense_level(overlay_info)

        feature_importance_df = pd.DataFrame({
            "feature": x_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return {
            "model": model_name,
            "date": latest_date,
            "raw_predictions": raw_preds,
            "adjusted_predictions": adjusted_preds,
            "signal_weights": signal_weights,
            "exec_weights": exec_weights,
            "feature_importance": feature_importance_df,
            "top_asset": top_asset,
            "second_asset": second_asset,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "overlay_fraction": overlay_fraction,
            "conditional_breakdown_defense_level": conditional_breakdown_defense_level(overlay_info),
            **overlay_info,
        }
    return None

# ============================================================
# 10. PRINT / SAVE HELPERS
# ============================================================
def print_weights(title: str, weights: dict, order: list):
    print(f"\n=== {title} ===")
    for asset in order:
        print(f"{asset}: {weights.get(asset, 0.0):.1%}")


def print_latest(latest: dict, sector_etfs: list):
    if latest is None:
        print("No latest recommendation available.")
        return
    print(f"\n=== Latest Recommendation: {latest['model']} ===")
    print("Signal date:", latest["date"].date())

    print("\nRaw predicted next-period returns:")
    for k, v in sorted(latest["raw_predictions"].items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print("\nAdjusted predicted next-period returns:")
    for k, v in sorted(latest["adjusted_predictions"].items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print(f"\nOverlay fraction on QQQM sleeve: {latest['overlay_fraction']:.1%}")
    print(f"Top asset: {latest['top_asset']}")
    print(f"Second asset: {latest['second_asset']}")
    print(f"Top score: {latest['top_score']:.4f}")
    print(f"Second score: {latest['second_score']:.4f}")
    print(f"Score gap: {latest['score_gap']:.4f}")
    print(f"Growth strength: {latest['growth_strength']:.3f}")
    print(f"SOXX strength: {latest['soxx_strength']:.3f}")
    print(f"Risk-off strength: {latest['risk_off_strength']:.3f}")
    print(f"Overlay style: {latest.get('overlay_style', 'v1')}")
    print(f"Copper strength: {latest['copper_strength']:.3f}")
    print(f"Industrial strength: {latest['industrial_strength']:.3f}")
    print(f"Materials strength: {latest['materials_strength']:.3f}")
    print(f"USD 3M strength: {latest['usd_3m_strength']:.3f}")
    print(f"Credit strength: {latest['credit_strength']:.3f}")
    print(f"Tech/real-economy divergence: {latest['tech_real_economy_divergence']:.3f}")
    print(f"Crash pressure: {latest['crash_pressure']:.3f}")
    print(f"Divergence defense level: {latest.get('conditional_breakdown_defense_level', 'normal')}")

    print_weights("Suggested SIGNAL Weights", latest["signal_weights"], sector_etfs + [cash_etf])
    print_weights("Suggested EXECUTED Weights", latest["exec_weights"], ["TQQQ", "ERX", "UXI"] + sector_etfs + [cash_etf])

    print("\n=== Latest Feature Importance Summary ===")
    print(latest["feature_importance"].head(25).to_string(index=False))


def save_latest(prefix: str, latest: dict):
    if latest is None:
        return
    latest_df = pd.DataFrame([{
        "signal_date": latest["date"],
        "latest_data_date": prices.index[-1],
        "top_asset": latest["top_asset"],
        "second_asset": latest["second_asset"],
        "top_score": latest["top_score"],
        "second_score": latest["second_score"],
        "score_gap": latest["score_gap"],
        "overlay_fraction": latest["overlay_fraction"],
        "overlay_style": latest.get("overlay_style", "v1"),
        "v2_tqqq_scale": latest.get("v2_tqqq_scale", 1.0),
        "v2_alert_action": latest.get("v2_alert_action", "NONE"),
        "war_strength": latest["war_strength"],
        "growth_strength": latest["growth_strength"],
        "risk_off_strength": latest["risk_off_strength"],
        "soxx_strength": latest["soxx_strength"],
        "copper_strength": latest["copper_strength"],
        "copper_3m_strength": latest["copper_3m_strength"],
        "industrial_strength": latest["industrial_strength"],
        "materials_strength": latest["materials_strength"],
        "usd_3m_strength": latest["usd_3m_strength"],
        "hyg_strength": latest["hyg_strength"],
        "credit_strength": latest["credit_strength"],
        "tech_real_economy_divergence": latest["tech_real_economy_divergence"],
        "crash_pressure": latest["crash_pressure"],
        "conditional_breakdown_defense_level": latest.get("conditional_breakdown_defense_level", "normal"),
        **{f"signal_w_{k}": v for k, v in latest["signal_weights"].items()},
        **{f"exec_w_{k}": v for k, v in latest["exec_weights"].items()},
        **{f"raw_pred_{k}": v for k, v in latest["raw_predictions"].items()},
        **{f"adj_pred_{k}": v for k, v in latest["adjusted_predictions"].items()},
    }])
    latest_df.to_csv(f"{prefix}_latest_recommendation.csv", index=False)
    latest["feature_importance"].to_csv(f"{prefix}_feature_importance.csv", index=False)

                                        


def apply_v2_continuous_tqqq_alert(latest: dict) -> dict:
    """
    V2 execution overlay.
    Keeps model prediction unchanged.
    Only adjusts final TQQQ exposure into QQQM.
    """

    if latest is None:
        return latest

    exec_weights = latest["exec_weights"].copy()

    risk_off = float(latest.get("risk_off_strength", 0.0))
    growth = float(latest.get("growth_strength", 0.0))
    soxx = float(latest.get("soxx_strength", 0.0))
    score_gap = float(latest.get("score_gap", 0.0))
    top_score = float(latest.get("top_score", 0.0))

    if top_score <= 0 or risk_off >= 1.50 or exec_weights.get("BIL", 0.0) >= 0.99:
        for a in exec_weights:
            exec_weights[a] = 0.0
        exec_weights["BIL"] = 1.0

        latest["exec_weights"] = exec_weights
        latest["v2_tqqq_scale"] = 0.0
        latest["v2_alert_action"] = "HARD_EXIT_TO_BIL"
        return latest

    tqqq_scale = 1.0

    if risk_off > 0.30:
        tqqq_scale *= 0.85
    if risk_off > 0.50:
        tqqq_scale *= 0.70
    if risk_off > 0.75:
        tqqq_scale *= 0.50
    if risk_off > 1.00:
        tqqq_scale *= 0.35

    if soxx < 0.50:
        tqqq_scale *= 0.80
    if soxx < 0.00:
        tqqq_scale *= 0.50

    if growth < 0.50:
        tqqq_scale *= 0.80
    if growth < 0.00:
        tqqq_scale *= 0.50

    if score_gap < 0.010:
        tqqq_scale *= 0.85
    if score_gap < 0.003:
        tqqq_scale *= 0.70

    old_tqqq = exec_weights.get("TQQQ", 0.0)
    new_tqqq = old_tqqq * tqqq_scale
    moved_to_qqqm = old_tqqq - new_tqqq

    exec_weights["TQQQ"] = new_tqqq
    exec_weights["QQQM"] = exec_weights.get("QQQM", 0.0) + moved_to_qqqm

    if risk_off > 1.25:
        cash_add = 0.20
        for a in exec_weights:
            if a != "BIL":
                exec_weights[a] *= (1.0 - cash_add)
        exec_weights["BIL"] = exec_weights.get("BIL", 0.0) + cash_add

    total = sum(exec_weights.values())
    if total > 0:
        for a in exec_weights:
            exec_weights[a] /= total

    latest["exec_weights"] = exec_weights
    latest["v2_tqqq_scale"] = tqqq_scale
    latest["v2_alert_action"] = "V2_TQQQ_TO_QQQM_OVERLAY"

    return latest
# ============================================================
# 11. PRODUCTION RUN: CURRENT BEST MODEL + DIVERGENCE ALERTS
# ============================================================
# This script keeps your trading model unchanged.
#
# Important:
#   Divergence / SOXX breakdown is ALERT ONLY.
#   It does NOT automatically reduce TQQQ.
#
# Why:
#   Previous tests showed no-defense was best in the tested 2023-2026 period.
#   So divergence is useful as monitoring information, not yet proven as a trading rule.


PRODUCTION_PREFIX = "model_c_plus_current_best_with_divergence_alerts"

# Make sure conditional defense is OFF.
# We still calculate divergence and breakdown variables for monitoring.
use_conditional_breakdown_defense = False


def classify_divergence_alert(latest: dict) -> str:
    """
    Alert-only classification.
    It does not change weights.
    """
    divergence = float(latest.get("tech_real_economy_divergence", 0.0))
    breakdown_score = float(latest.get("breakdown_score", 0.0))
    risk_off = float(latest.get("risk_off_strength", 0.0))
    soxx = float(latest.get("soxx_strength", 0.0))
    growth = float(latest.get("growth_strength", 0.0))

    # Highest concern: divergence + actual tech breakdown + risk-off rising.
    if divergence >= 3.0 and breakdown_score >= 2 and risk_off >= 0.0:
        return "DANGER: high divergence + SOXX/QQQM breakdown + risk-off rising"

    if divergence >= 2.5 and breakdown_score >= 1 and risk_off >= -0.25:
        return "WARNING: high divergence + early SOXX/QQQM weakness"

    if divergence >= 2.5 and soxx > 1.0 and growth > 1.0:
        return "WATCH: narrow tech-led rally; no action unless SOXX breaks"

    if breakdown_score >= 2 and risk_off >= 0.0:
        return "WARNING: tech breakdown pressure, but divergence not extreme"

    return "NORMAL"


def build_alert_row(latest: dict) -> pd.DataFrame:
    """
    Save a compact alert dashboard CSV.
    """
    row = {
        "signal_date": latest["date"],
        "latest_data_date": prices.index[-1],
        "top_asset": latest["top_asset"],
        "second_asset": latest["second_asset"],
        "top_score": latest["top_score"],
        "second_score": latest["second_score"],
        "score_gap": latest["score_gap"],
        "overlay_fraction": latest["overlay_fraction"],

        "alert_level": classify_divergence_alert(latest),

        "growth_strength": latest.get("growth_strength", np.nan),
        "soxx_strength": latest.get("soxx_strength", np.nan),
        "risk_off_strength": latest.get("risk_off_strength", np.nan),
        "industrial_strength": latest.get("industrial_strength", np.nan),
        "materials_strength": latest.get("materials_strength", np.nan),
        "copper_strength": latest.get("copper_strength", np.nan),
        "credit_strength": latest.get("credit_strength", np.nan),

        "tech_real_economy_divergence": latest.get("tech_real_economy_divergence", np.nan),
        "crash_pressure": latest.get("crash_pressure", np.nan),
        "breakdown_score": latest.get("breakdown_score", np.nan),

        "soxx_5d": latest.get("soxx_5d", np.nan),
        "soxx_10d": latest.get("soxx_10d", np.nan),
        "soxx_dd_21": latest.get("soxx_dd_21", np.nan),
        "qqqm_5d": latest.get("qqqm_5d", np.nan),
        "qqqm_10d": latest.get("qqqm_10d", np.nan),
        "qqqm_dd_21": latest.get("qqqm_dd_21", np.nan),

        **{f"signal_w_{k}": v for k, v in latest["signal_weights"].items()},
        **{f"exec_w_{k}": v for k, v in latest["exec_weights"].items()},
        **{f"raw_pred_{k}": v for k, v in latest["raw_predictions"].items()},
        **{f"adj_pred_{k}": v for k, v in latest["adjusted_predictions"].items()},
    }
    return pd.DataFrame([row])


def print_alert_dashboard(latest: dict):
    print("\n========================")
    print("DIVERGENCE / BREAKDOWN ALERT DASHBOARD")
    print("========================")
    print("Alert:", classify_divergence_alert(latest))
    print(f"Tech/real-economy divergence: {latest.get('tech_real_economy_divergence', np.nan):.3f}")
    print(f"Crash pressure:               {latest.get('crash_pressure', np.nan):.3f}")
    print(f"Breakdown score:              {latest.get('breakdown_score', np.nan):.1f}")
    print(f"SOXX 5d:                      {latest.get('soxx_5d', np.nan):.3f}")
    print(f"SOXX 10d:                     {latest.get('soxx_10d', np.nan):.3f}")
    print(f"SOXX 21d drawdown:            {latest.get('soxx_dd_21', np.nan):.3f}")
    print(f"QQQM 5d:                      {latest.get('qqqm_5d', np.nan):.3f}")
    print(f"QQQM 10d:                     {latest.get('qqqm_10d', np.nan):.3f}")
    print(f"QQQM 21d drawdown:            {latest.get('qqqm_dd_21', np.nan):.3f}")

    print("\nInterpretation:")
    print("- NORMAL: no special warning.")
    print("- WATCH: divergence is high, but tech still leads. Monitor only.")
    print("- WARNING: divergence plus early SOXX/QQQM weakness. Be careful with new TQQQ buys.")
    print("- DANGER: divergence plus breakdown plus risk-off. Consider manual risk reduction.")





# ============================================================
# 11B. MACRO / RMB DASHBOARD MONITORING LAYER
# ============================================================
# IMPORTANT:
# This dashboard is for interpretation and learning.
# It does NOT change weights, predictions, or execution.
# It helps monitor whether the market is shifting from AI leadership
# toward EM / reflation / industrial broadening.

def _safe_series_value(series: pd.Series, date: pd.Timestamp, default=np.nan) -> float:
    try:
        if date in series.index and pd.notna(series.loc[date]):
            return float(series.loc[date])
    except Exception:
        pass
    return default


def classify_macro_regime_for_dashboard(latest: dict) -> str:
    growth = float(latest.get("growth_strength", 0.0))
    soxx = float(latest.get("soxx_strength", 0.0))
    risk_off = float(latest.get("risk_off_strength", 0.0))
    industrial = float(latest.get("industrial_strength", 0.0))
    materials = float(latest.get("materials_strength", 0.0))
    copper = float(latest.get("copper_strength", 0.0))
    credit = float(latest.get("credit_strength", 0.0))

    date = latest.get("date")
    rmb_gate = _safe_series_value(rmb_activation_gate, date, 0.0)
    china_liq = _safe_series_value(china_em_liquidity_strength, date, 0.0)

    if risk_off > 1.25:
        return "RISK_OFF_DEFENSIVE"

    if growth > 1.5 and soxx > 1.5 and industrial < 0.5 and materials < 0.5:
        return "NARROW_AI_LEADERSHIP"

    if rmb_gate > 0 and industrial > 0.25 and copper > 0.25 and credit > -0.5:
        return "POSSIBLE_EM_REFLATION_BROADENING"

    if china_liq > 0.5 and copper > 0.0 and industrial > 0.0:
        return "CHINA_EM_LIQUIDITY_IMPROVING"

    if industrial > 0.5 and materials > 0.5:
        return "INDUSTRIAL_MATERIALS_BROADENING"

    return "MIXED_OR_TRANSITIONAL"


def build_macro_regime_dashboard_row(latest: dict, dashboard_model_name: str) -> pd.DataFrame:
    date = latest.get("date")

    row = {
        "dashboard_model": dashboard_model_name,
        "signal_date": date,
        "latest_data_date": prices.index[-1],
        "macro_regime_label": classify_macro_regime_for_dashboard(latest),

        # Actual model outcome
        "top_asset": latest.get("top_asset"),
        "second_asset": latest.get("second_asset"),
        "top_score": latest.get("top_score", np.nan),
        "second_score": latest.get("second_score", np.nan),
        "score_gap": latest.get("score_gap", np.nan),
        "overlay_fraction": latest.get("overlay_fraction", np.nan),
        "exec_w_TQQQ": latest.get("exec_weights", {}).get("TQQQ", np.nan),
        "exec_w_QQQM": latest.get("exec_weights", {}).get("QQQM", np.nan),
        "exec_w_XSOE": latest.get("exec_weights", {}).get("XSOE", np.nan),
        "exec_w_XLI": latest.get("exec_weights", {}).get("XLI", np.nan),
        "exec_w_XLB": latest.get("exec_weights", {}).get("XLB", np.nan),
        "exec_w_XLE": latest.get("exec_weights", {}).get("XLE", np.nan),
        "exec_w_BIL": latest.get("exec_weights", {}).get("BIL", np.nan),

        # Tech / AI
        "growth_strength": latest.get("growth_strength", np.nan),
        "soxx_strength": latest.get("soxx_strength", np.nan),
        "qqqm_5d": latest.get("qqqm_5d", np.nan),
        "qqqm_10d": latest.get("qqqm_10d", np.nan),
        "soxx_5d": latest.get("soxx_5d", np.nan),
        "soxx_10d": latest.get("soxx_10d", np.nan),
        "tech_real_economy_divergence": latest.get("tech_real_economy_divergence", np.nan),
        "breakdown_score": latest.get("breakdown_score", np.nan),

        # Industrial / reflation
        "industrial_strength": latest.get("industrial_strength", np.nan),
        "materials_strength": latest.get("materials_strength", np.nan),
        "copper_strength": latest.get("copper_strength", np.nan),
        "copper_3m_strength": latest.get("copper_3m_strength", np.nan),
        "credit_strength": latest.get("credit_strength", np.nan),
        "hyg_strength": latest.get("hyg_strength", np.nan),

        # Oil / war / energy
        "war_strength": latest.get("war_strength", np.nan),

        # Risk / USD
        "risk_off_strength": latest.get("risk_off_strength", np.nan),
        "crash_pressure": latest.get("crash_pressure", np.nan),
        "usd_3m_strength": latest.get("usd_3m_strength", np.nan),

        # RMB / China / EM dashboard-only diagnostics
        "usd_cnh_latest": _safe_series_value(usd_cnh, date),
        "usd_cnh_1m": _safe_series_value(usd_cnh_1m, date),
        "usd_cnh_3m": _safe_series_value(usd_cnh_3m, date),
        "rmb_1m": _safe_series_value(rmb_1m, date),
        "rmb_3m": _safe_series_value(rmb_3m, date),
        "rmb_strength": _safe_series_value(rmb_strength, date),
        "rmb_3m_strength": _safe_series_value(rmb_3m_strength, date),
        "china_em_liquidity_strength": _safe_series_value(china_em_liquidity_strength, date),
        "industrial_broadening_gate": _safe_series_value(industrial_broadening_gate, date),
        "rmb_activation_gate": _safe_series_value(rmb_activation_gate, date),
        "conditional_china_em_liquidity_strength": _safe_series_value(conditional_china_em_liquidity_strength, date),
    }
    return pd.DataFrame([row])


def macro_dashboard_interpretation_lines(latest: dict, dashboard_model_name: str) -> list:
    date = latest.get("date")
    regime = classify_macro_regime_for_dashboard(latest)
    growth = float(latest.get("growth_strength", 0.0))
    soxx = float(latest.get("soxx_strength", 0.0))
    industrial = float(latest.get("industrial_strength", 0.0))
    materials = float(latest.get("materials_strength", 0.0))
    copper = float(latest.get("copper_strength", 0.0))
    risk_off = float(latest.get("risk_off_strength", 0.0))
    credit = float(latest.get("credit_strength", 0.0))
    divergence = float(latest.get("tech_real_economy_divergence", 0.0))
    breakdown = float(latest.get("breakdown_score", 0.0))

    usd_cnh_now = _safe_series_value(usd_cnh, date)
    usd_cnh_1m_now = _safe_series_value(usd_cnh_1m, date)
    usd_cnh_3m_now = _safe_series_value(usd_cnh_3m, date)
    rmb_strength_now = _safe_series_value(rmb_strength, date)
    china_liq = _safe_series_value(china_em_liquidity_strength, date)
    broad_gate = _safe_series_value(industrial_broadening_gate, date, 0.0)
    rmb_gate = _safe_series_value(rmb_activation_gate, date, 0.0)
    conditional_liq = _safe_series_value(conditional_china_em_liquidity_strength, date, 0.0)

    lines = []
    lines.append("========================")
    lines.append(f"MACRO / RMB DASHBOARD MONITOR: {dashboard_model_name}")
    lines.append("========================")
    lines.append(f"Signal date: {date.date() if hasattr(date, 'date') else date}")
    lines.append(f"Macro regime label: {regime}")
    lines.append("")

    lines.append("--- ACTUAL MODEL OUTCOME ---")
    lines.append(f"Top asset: {latest.get('top_asset')} | Second: {latest.get('second_asset')}")
    lines.append(f"Top score: {latest.get('top_score', np.nan):.4f} | Gap: {latest.get('score_gap', np.nan):.4f}")
    lines.append(f"Overlay fraction: {latest.get('overlay_fraction', np.nan):.1%}")
    lines.append(f"Executed weights: TQQQ={latest.get('exec_weights', {}).get('TQQQ', 0):.1%}, QQQM={latest.get('exec_weights', {}).get('QQQM', 0):.1%}, XSOE={latest.get('exec_weights', {}).get('XSOE', 0):.1%}, XLI={latest.get('exec_weights', {}).get('XLI', 0):.1%}, XLB={latest.get('exec_weights', {}).get('XLB', 0):.1%}, XLE={latest.get('exec_weights', {}).get('XLE', 0):.1%}, BIL={latest.get('exec_weights', {}).get('BIL', 0):.1%}")
    lines.append("")

    lines.append("--- TECH / AI LEADERSHIP ---")
    lines.append(f"Growth strength: {growth:.3f}")
    lines.append(f"SOXX strength:   {soxx:.3f}")
    lines.append(f"Tech/real-economy divergence: {divergence:.3f}")
    lines.append(f"Breakdown score: {breakdown:.1f}")
    if growth > 1.5 and soxx > 1.5:
        lines.append("Interpretation: AI / semiconductor leadership remains dominant.")
    elif growth > 0 and soxx > 0:
        lines.append("Interpretation: tech is still positive, but not extreme.")
    else:
        lines.append("Interpretation: tech leadership is weak or fading.")
    lines.append("")

    lines.append("--- INDUSTRIAL / MATERIALS / COPPER ---")
    lines.append(f"Industrial strength: {industrial:.3f}")
    lines.append(f"Materials strength:  {materials:.3f}")
    lines.append(f"Copper strength:     {copper:.3f}")
    lines.append(f"Credit strength:     {credit:.3f}")
    lines.append(f"Industrial broadening gate: {broad_gate:.3f}")
    if industrial > 0.5 and materials > 0.5 and copper > 0.25:
        lines.append("Interpretation: industrial/materials broadening is confirmed.")
    elif broad_gate > 0:
        lines.append("Interpretation: early signs of industrial/EM broadening exist, but not full confirmation.")
    else:
        lines.append("Interpretation: industrial/materials broadening is not confirmed yet.")
    lines.append("")

    lines.append("--- RMB / CHINA / EM LIQUIDITY ---")
    lines.append(f"USD/CNH or USD/CNY latest: {usd_cnh_now:.4f}")
    lines.append(f"USD/CNH 1M change: {usd_cnh_1m_now:.3%}")
    lines.append(f"USD/CNH 3M change: {usd_cnh_3m_now:.3%}")
    lines.append(f"RMB strength z-score: {rmb_strength_now:.3f}")
    lines.append(f"China/EM liquidity strength: {china_liq:.3f}")
    lines.append(f"RMB activation gate: {rmb_gate:.3f}")
    lines.append(f"Conditional China/EM liquidity: {conditional_liq:.3f}")
    if rmb_gate <= 0:
        lines.append("Interpretation: RMB is monitored only. It is NOT activated as a tradable EM/reflation signal now.")
    else:
        lines.append("Interpretation: RMB activation is ON because RMB/China liquidity aligns with broader EM/reflation conditions.")
    lines.append("")

    lines.append("--- RISK / USD / CRASH PRESSURE ---")
    lines.append(f"Risk-off strength: {risk_off:.3f}")
    lines.append(f"Crash pressure:     {latest.get('crash_pressure', np.nan):.3f}")
    lines.append(f"USD 3M strength:    {latest.get('usd_3m_strength', np.nan):.3f}")
    if risk_off > 1.0:
        lines.append("Interpretation: risk-off is elevated; leverage should be treated carefully.")
    else:
        lines.append("Interpretation: risk-off pressure is not dominant.")
    lines.append("")

    lines.append("--- DASHBOARD DECISION ---")
    lines.append("This dashboard does not override the model. Use it to monitor whether future regime shifts make RMB/EM logic more relevant.")
    if regime == "POSSIBLE_EM_REFLATION_BROADENING":
        lines.append("Research note: if this label persists, rerun the conditional RMB robustness test before using RMB version for production rebalancing.")
    elif regime == "NARROW_AI_LEADERSHIP":
        lines.append("Research note: current environment still favors the original AI/SOXX/TQQQ engine; RMB should remain dashboard-only.")
    return lines


def print_macro_regime_dashboard(latest: dict, dashboard_model_name: str):
    for line in macro_dashboard_interpretation_lines(latest, dashboard_model_name):
        print(line)


def save_macro_dashboard_text(latest_list: list, filename: str):
    all_lines = []
    for latest, name in latest_list:
        all_lines.extend(macro_dashboard_interpretation_lines(latest, name))
        all_lines.append("")
        all_lines.append("")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))

# ============================================================
# 12. RIGOROUS RMB FEATURE TEST HELPERS
# ============================================================
def compare_two_models(base_returns: pd.Series, challenger_returns: pd.Series) -> pd.DataFrame:
    common_idx = base_returns.dropna().index.intersection(challenger_returns.dropna().index)
    base = base_returns.loc[common_idx]
    challenger = challenger_returns.loc[common_idx]
    diff = challenger - base

    rows = []
    rows.append({
        "test": "FULL_PERIOD",
        "start": common_idx.min(),
        "end": common_idx.max(),
        "days": len(common_idx),
        "base_annual_return": annualized_return(base),
        "challenger_annual_return": annualized_return(challenger),
        "delta_annual_return": annualized_return(challenger) - annualized_return(base),
        "base_sharpe": sharpe_ratio(base, risk_free_rate_annual),
        "challenger_sharpe": sharpe_ratio(challenger, risk_free_rate_annual),
        "delta_sharpe": sharpe_ratio(challenger, risk_free_rate_annual) - sharpe_ratio(base, risk_free_rate_annual),
        "base_max_drawdown": max_drawdown(base),
        "challenger_max_drawdown": max_drawdown(challenger),
        "delta_max_drawdown": max_drawdown(challenger) - max_drawdown(base),
        "mean_daily_delta": diff.mean(),
        "hit_rate_challenger_beats_base": (diff > 0).mean(),
    })
    return pd.DataFrame(rows)


def subperiod_comparison(base_returns: pd.Series, challenger_returns: pd.Series, freq: str = "YE") -> pd.DataFrame:
    common_idx = base_returns.dropna().index.intersection(challenger_returns.dropna().index)
    df = pd.DataFrame({
        "base": base_returns.loc[common_idx],
        "challenger": challenger_returns.loc[common_idx],
    }).dropna()
    df["period"] = df.index.to_period(freq)

    rows = []
    for period, g in df.groupby("period"):
        if len(g) < 40:
            continue
        rows.append({
            "period": str(period),
            "days": len(g),
            "base_annual_return": annualized_return(g["base"]),
            "challenger_annual_return": annualized_return(g["challenger"]),
            "delta_annual_return": annualized_return(g["challenger"]) - annualized_return(g["base"]),
            "base_sharpe": sharpe_ratio(g["base"], risk_free_rate_annual),
            "challenger_sharpe": sharpe_ratio(g["challenger"], risk_free_rate_annual),
            "delta_sharpe": sharpe_ratio(g["challenger"], risk_free_rate_annual) - sharpe_ratio(g["base"], risk_free_rate_annual),
            "base_max_drawdown": max_drawdown(g["base"]),
            "challenger_max_drawdown": max_drawdown(g["challenger"]),
            "delta_max_drawdown": max_drawdown(g["challenger"]) - max_drawdown(g["base"]),
            "hit_rate_challenger_beats_base": (g["challenger"] > g["base"]).mean(),
        })
    return pd.DataFrame(rows)


def summarize_feature_family_importance(feature_importance_df: pd.DataFrame, family_keywords: list) -> pd.DataFrame:
    rows = []
    for key in family_keywords:
        mask = feature_importance_df["feature"].str.contains(key, case=False, regex=False)
        rows.append({
            "family_keyword": key,
            "feature_count": int(mask.sum()),
            "total_importance": float(feature_importance_df.loc[mask, "importance"].sum()),
            "max_single_feature_importance": float(feature_importance_df.loc[mask, "importance"].max()) if mask.any() else 0.0,
        })
    return pd.DataFrame(rows).sort_values("total_importance", ascending=False)

# ============================================================
# 12. RUN CONTROLLED BASELINE VS CONDITIONAL RMB CHALLENGER TEST
# ============================================================
BASE_PREFIX = "model_c_plus_current_best_baseline_no_rmb"
RMB_PREFIX = "model_c_plus_conditional_rmb_activation_test_001"
COMPARE_PREFIX = "conditional_rmb_activation_test_001"

print("\nBuilding BASELINE feature set: current best, no RMB features...")
baseline_features = build_features_by_asset(UPGRADED_SECTOR_ETFS, include_rmb_features=False)

print("\nBuilding CONDITIONAL RMB CHALLENGER feature set: current best + CONDITIONAL RMB activation features...")
rmb_features = build_features_by_asset(UPGRADED_SECTOR_ETFS, include_rmb_features=True)

print("\nRunning BASELINE current best model...")
base_returns, base_rebalance, base_turnover = run_strategy(
    "BASELINE_NO_RMB_CURRENT_BEST",
    UPGRADED_SECTOR_ETFS,
    baseline_features,
    overlay_style="hybrid",
    tqqq_style="dynamic",
)

print("\nRunning CONDITIONAL RMB CHALLENGER model...")
rmb_returns, rmb_rebalance, rmb_turnover = run_strategy(
    "CONDITIONAL_RMB_ACTIVATOR_001",
    UPGRADED_SECTOR_ETFS,
    rmb_features,
    overlay_style="hybrid",
    tqqq_style="dynamic",
)

if len(base_returns) == 0:
    raise ValueError("Baseline model produced no returns.")
if len(rmb_returns) == 0:
    raise ValueError("RMB challenger model produced no returns.")

summary_df = pd.DataFrame([
    performance_summary("BASELINE_NO_RMB_CURRENT_BEST", base_returns, base_turnover),
    performance_summary("CONDITIONAL_RMB_ACTIVATOR_001", rmb_returns, rmb_turnover),
])

full_compare_df = compare_two_models(base_returns, rmb_returns)
yearly_compare_df = subperiod_comparison(base_returns, rmb_returns, freq="Y")

print("\n=== PERFORMANCE SUMMARY: BASELINE VS CONDITIONAL RMB CHALLENGER ===")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n=== FULL PERIOD DELTA: CONDITIONAL RMB CHALLENGER - BASELINE ===")
print(full_compare_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\n=== YEARLY ROBUSTNESS CHECK ===")
print(yearly_compare_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Latest recommendations for both models.
base_latest = get_latest_recommendation(
    "BASELINE_NO_RMB_CURRENT_BEST",
    UPGRADED_SECTOR_ETFS,
    baseline_features,
    overlay_style="hybrid",
    tqqq_style="dynamic",
)
base_latest = apply_v2_continuous_tqqq_alert(base_latest)

rmb_latest = get_latest_recommendation(
    "CONDITIONAL_RMB_ACTIVATOR_001",
    UPGRADED_SECTOR_ETFS,
    rmb_features,
    overlay_style="hybrid",
    tqqq_style="dynamic",
)
rmb_latest = apply_v2_continuous_tqqq_alert(rmb_latest)

print_latest(base_latest, UPGRADED_SECTOR_ETFS)
print_alert_dashboard(base_latest)
print_macro_regime_dashboard(base_latest, "BASELINE_NO_RMB_CURRENT_BEST_DASHBOARD_ONLY")

print_latest(rmb_latest, UPGRADED_SECTOR_ETFS)
print_alert_dashboard(rmb_latest)
print_macro_regime_dashboard(rmb_latest, "CONDITIONAL_RMB_ACTIVATOR_RESEARCH_MODEL")

base_macro_dashboard_df = build_macro_regime_dashboard_row(
    base_latest,
    "BASELINE_NO_RMB_CURRENT_BEST_DASHBOARD_ONLY",
)
rmb_macro_dashboard_df = build_macro_regime_dashboard_row(
    rmb_latest,
    "CONDITIONAL_RMB_ACTIVATOR_RESEARCH_MODEL",
)
macro_dashboard_df = pd.concat(
    [base_macro_dashboard_df, rmb_macro_dashboard_df],
    ignore_index=True,
)

# RMB feature importance diagnostics.
rmb_family_importance = summarize_feature_family_importance(
    rmb_latest["feature_importance"],
    ["conditional", "rmb_activation", "china_em", "rmb", "usd", "copper", "soxx", "hyg", "credit"],
)
print("\n=== CONDITIONAL RMB CHALLENGER FEATURE FAMILY IMPORTANCE ===")
print(rmb_family_importance.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Latest RMB diagnostics.
rmb_diag = pd.DataFrame([{
    "latest_data_date": prices.index[-1],
    "usd_cnh_latest": float(usd_cnh.iloc[-1]) if pd.notna(usd_cnh.iloc[-1]) else np.nan,
    "usd_cnh_1m": float(usd_cnh_1m.iloc[-1]) if pd.notna(usd_cnh_1m.iloc[-1]) else np.nan,
    "usd_cnh_3m": float(usd_cnh_3m.iloc[-1]) if pd.notna(usd_cnh_3m.iloc[-1]) else np.nan,
    "rmb_1m": float(rmb_1m.iloc[-1]) if pd.notna(rmb_1m.iloc[-1]) else np.nan,
    "rmb_3m": float(rmb_3m.iloc[-1]) if pd.notna(rmb_3m.iloc[-1]) else np.nan,
    "rmb_strength": float(rmb_strength.iloc[-1]) if pd.notna(rmb_strength.iloc[-1]) else np.nan,
    "rmb_3m_strength": float(rmb_3m_strength.iloc[-1]) if pd.notna(rmb_3m_strength.iloc[-1]) else np.nan,
    "china_em_liquidity_strength": float(china_em_liquidity_strength.iloc[-1]) if pd.notna(china_em_liquidity_strength.iloc[-1]) else np.nan,
    "rmb_activation_gate": float(rmb_activation_gate.iloc[-1]) if pd.notna(rmb_activation_gate.iloc[-1]) else np.nan,
    "industrial_broadening_gate": float(industrial_broadening_gate.iloc[-1]) if pd.notna(industrial_broadening_gate.iloc[-1]) else np.nan,
    "conditional_china_em_liquidity_strength": float(conditional_china_em_liquidity_strength.iloc[-1]) if pd.notna(conditional_china_em_liquidity_strength.iloc[-1]) else np.nan,
}])
print("\n=== LATEST RMB DIAGNOSTICS ===")
print(rmb_diag.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Activation diagnostics over the actual backtest period.
activation_stats = pd.DataFrame([{
    "activation_days_pct": float((rmb_activation_gate.reindex(base_returns.index).fillna(0.0) > 0).mean()),
    "mean_activation_gate": float(rmb_activation_gate.reindex(base_returns.index).fillna(0.0).mean()),
    "mean_conditional_china_em_liquidity": float(conditional_china_em_liquidity_strength.reindex(base_returns.index).fillna(0.0).mean()),
    "latest_activation_gate": float(rmb_activation_gate.iloc[-1]) if pd.notna(rmb_activation_gate.iloc[-1]) else np.nan,
}])
print("\n=== CONDITIONAL RMB ACTIVATION STATS ===")
print(activation_stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Decision rule printout: do not adopt unless robust.
try:
    delta_sharpe = float(full_compare_df["delta_sharpe"].iloc[0])
    delta_return = float(full_compare_df["delta_annual_return"].iloc[0])
    delta_dd = float(full_compare_df["delta_max_drawdown"].iloc[0])
    yearly_delta_sharpe_mean = float(yearly_compare_df["delta_sharpe"].mean()) if len(yearly_compare_df) else np.nan
    yearly_positive_rate = float((yearly_compare_df["delta_sharpe"] > 0).mean()) if len(yearly_compare_df) else np.nan

    print("\n=== ADOPTION DECISION RULE ===")
    print("Adopt Conditional RMB activator only if most of these are true:")
    print("1) delta_sharpe > 0")
    print("2) delta_annual_return > 0")
    print("3) max drawdown is not materially worse")
    print("4) yearly/subperiod delta_sharpe is positive in a majority of periods")
    print("5) RMB-family feature importance is not effectively zero")
    print("\nObserved:")
    print(f"delta_sharpe: {delta_sharpe:.6f}")
    print(f"delta_annual_return: {delta_return:.6f}")
    print(f"delta_max_drawdown: {delta_dd:.6f}")
    print(f"mean_yearly_delta_sharpe: {yearly_delta_sharpe_mean:.6f}")
    print(f"positive_yearly_delta_sharpe_rate: {yearly_positive_rate:.2%}")
except Exception as e:
    print("Could not calculate adoption decision summary:", e)

# ============================================================
# 13. SAVE OUTPUTS
# ============================================================
base_returns.to_csv(f"{BASE_PREFIX}_portfolio_daily_returns.csv", header=["portfolio_return"])
base_rebalance.to_csv(f"{BASE_PREFIX}_rebalance_log.csv", index=False)
save_latest(BASE_PREFIX, base_latest)
build_alert_row(base_latest).to_csv(f"{BASE_PREFIX}_alert_dashboard.csv", index=False)

rmb_returns.to_csv(f"{RMB_PREFIX}_portfolio_daily_returns.csv", header=["portfolio_return"])
rmb_rebalance.to_csv(f"{RMB_PREFIX}_rebalance_log.csv", index=False)
save_latest(RMB_PREFIX, rmb_latest)
build_alert_row(rmb_latest).to_csv(f"{RMB_PREFIX}_alert_dashboard.csv", index=False)

summary_df.to_csv(f"{COMPARE_PREFIX}_performance_summary.csv", index=False)
full_compare_df.to_csv(f"{COMPARE_PREFIX}_full_period_delta.csv", index=False)
yearly_compare_df.to_csv(f"{COMPARE_PREFIX}_yearly_robustness.csv", index=False)
rmb_family_importance.to_csv(f"{COMPARE_PREFIX}_rmb_feature_family_importance.csv", index=False)
rmb_diag.to_csv(f"{COMPARE_PREFIX}_latest_rmb_diagnostics.csv", index=False)
activation_stats.to_csv(f"{COMPARE_PREFIX}_activation_stats.csv", index=False)
macro_dashboard_df.to_csv(f"{COMPARE_PREFIX}_macro_rmb_dashboard_monitor.csv", index=False)
save_macro_dashboard_text(
    [
        (base_latest, "BASELINE_NO_RMB_CURRENT_BEST_DASHBOARD_ONLY"),
        (rmb_latest, "CONDITIONAL_RMB_ACTIVATOR_RESEARCH_MODEL"),
    ],
    f"{COMPARE_PREFIX}_macro_rmb_dashboard_monitor.txt",
)

print("\nSaved:")
print(f"- {BASE_PREFIX}_portfolio_daily_returns.csv")
print(f"- {BASE_PREFIX}_rebalance_log.csv")
print(f"- {BASE_PREFIX}_latest_recommendation.csv")
print(f"- {BASE_PREFIX}_feature_importance.csv")
print(f"- {BASE_PREFIX}_alert_dashboard.csv")
print(f"- {RMB_PREFIX}_portfolio_daily_returns.csv")
print(f"- {RMB_PREFIX}_rebalance_log.csv")
print(f"- {RMB_PREFIX}_latest_recommendation.csv")
print(f"- {RMB_PREFIX}_feature_importance.csv")
print(f"- {RMB_PREFIX}_alert_dashboard.csv")
print(f"- {COMPARE_PREFIX}_performance_summary.csv")
print(f"- {COMPARE_PREFIX}_full_period_delta.csv")
print(f"- {COMPARE_PREFIX}_yearly_robustness.csv")
print(f"- {COMPARE_PREFIX}_rmb_feature_family_importance.csv")
print(f"- {COMPARE_PREFIX}_latest_rmb_diagnostics.csv")
print(f"- {COMPARE_PREFIX}_activation_stats.csv")
print(f"- {COMPARE_PREFIX}_macro_rmb_dashboard_monitor.csv")
print(f"- {COMPARE_PREFIX}_macro_rmb_dashboard_monitor.txt")
