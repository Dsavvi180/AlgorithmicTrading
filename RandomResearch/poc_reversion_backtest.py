# Grid-search POC reversion with TWAP-direction filter (ATR-free, macOS/Apple Silicon friendly)
# - Signal when price is outside the central 30% volume band (15–85th by volume) OR beyond an absolute distance from POC
# - Only take trades in the direction of the TWAP slope; require POC to be close to TWAP
# - Parallel grid search; prints progress, best results, and re-runs the best configuration
# - Uses only first 4 years; RTH filtering with open/close padding; robust guards

import os, math, warnings
from datetime import timedelta, time
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================
# Global config (I/O)
# =====================
CSV_PATH = "/Users/damensavvasavvi/Desktop/Desktop - Damen’s MacBook Air (2)/AlgoTrading/marketDataNasdaqFutures/NQ_continuous_backadjusted_1m_cleaned.csv"
USE_FIRST_YEARS = 4
RESULTS_CSV = "grid_results_poc_twap_NO_ATR.csv"
TOP_K = 10            # how many top configs to print
SCORE_HORIZON_GLOBAL = 5   # default scoring horizon (minutes)
RANDOM_SEED = 42
MAX_WORKERS = None     # None = use all cores; set an int (e.g., 6–8) if you want fewer PIDs
GRID_PROGRESS_EVERY = 20   # print every N completed grid runs

# =====================
# Fixed session config
# =====================
USE_RTH = True
RTH_START_ET = time(9, 30)
RTH_END_ET   = time(16, 0)
EXCLUDE_OPEN_MIN = 5   # exclude first 5 minutes after open
EXCLUDE_CLOSE_MIN = 5  # exclude last 5 minutes before close

# =====================
# Helpers
# =====================
def weighted_percentile(values, weights, percent):
    values = np.asarray(values); weights = np.asarray(weights)
    if values.size == 0: return np.nan
    idx = np.argsort(values); v = values[idx]; w = weights[idx]
    c = np.cumsum(w)
    if c.size == 0 or c[-1] == 0: return np.percentile(v, percent)
    cutoff = percent/100.0 * c[-1]
    return np.interp(cutoff, c, v)

def build_volume_profile(prices, vols, bin_size):
    prices = np.asarray(prices); vols = np.asarray(vols)
    if prices.size < 2 or np.all(np.isnan(prices)) or np.nansum(vols) == 0: return None
    mask = ~np.isnan(prices); prices = prices[mask]; vols = vols[mask]
    if prices.size < 2: return None
    pmin, pmax = prices.min(), prices.max()
    if not np.isfinite(pmin) or not np.isfinite(pmax): return None
    if pmin == pmax: pmin -= bin_size; pmax += bin_size
    nbins = max(2, int(np.ceil((pmax - pmin) / bin_size)))
    edges = np.linspace(pmin, pmin + nbins*bin_size, nbins + 1)
    hist, edges = np.histogram(prices, bins=edges, weights=vols)
    centers = 0.5*(edges[:-1] + edges[1:])
    if hist.sum() <= 0: return None
    poc = centers[np.argmax(hist)]
    # central 30%: 15th–85th percentiles by **volume**
    p_lo = weighted_percentile(centers, hist, 15)
    p_hi = weighted_percentile(centers, hist, 85)
    return centers, hist, poc, p_lo, p_hi

def in_rth(ts_utc, open_pad=0, close_pad=0):
    local_dt = ts_utc.tz_convert("US/Eastern")
    start = local_dt.replace(hour=RTH_START_ET.hour, minute=RTH_START_ET.minute, second=0, microsecond=0) + timedelta(minutes=open_pad)
    end   = local_dt.replace(hour=RTH_END_ET.hour,   minute=RTH_END_ET.minute,   second=0, microsecond=0) - timedelta(minutes=close_pad)
    if end <= start: return False
    return (local_dt >= start) and (local_dt <= end)

def ret_towards_poc(curr_close, fut_close, poc):
    direction = np.sign(poc - curr_close)
    return direction * (fut_close - curr_close) / curr_close

def t_like(x):
    x = pd.Series(x).dropna()
    if len(x) < 2: return np.nan
    m = x.mean(); sd = x.std(ddof=1)
    return m / (sd / math.sqrt(len(x))) if sd > 0 else np.nan

# =====================
# Data loader
# =====================
def load_base_df():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}.")
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    # First N years
    start = df.index.min()
    end = start + relativedelta(years=USE_FIRST_YEARS)
    df = df.loc[(df.index >= start) & (df.index < end)].copy()
    # Numeric
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close","volume"])
    # RTH mask
    if USE_RTH:
        df = df[df.index.map(lambda ts: in_rth(ts, EXCLUDE_OPEN_MIN, EXCLUDE_CLOSE_MIN))]
    return df

BASE_DF = load_base_df()

# =====================
# Backtest core (ATR-free)
# =====================
def run_backtest(params):
    """
    Returns dict with metrics + params + skip/cancel counters.
    """
    # Unpack params
    USE_COUNT_WINDOW = params.get("USE_COUNT_WINDOW", True)
    WINDOW_OBS = params.get("WINDOW_OBS", 240)
    WINDOW_MINUTES = params.get("WINDOW_MINUTES", 240)
    MIN_WINDOW_ROWS = params.get("MIN_WINDOW_ROWS", 80)
    BIN_SIZE = params.get("BIN_SIZE", 1.0)

    MIN_DIST_POINTS = params.get("MIN_DIST_POINTS", 2.0)   # minimum raw distance from POC
    ABS_THRESHOLD_POINTS = params.get("ABS_THRESHOLD_POINTS", 30.0)

    POC_STABILITY_POINTS = params.get("POC_STABILITY_POINTS", 5.0)
    MIN_ROLLING_VOL = params.get("MIN_ROLLING_VOL", 50)
    VOL_LOOKBACK = params.get("VOL_LOOKBACK", 20)

    SIG_RETURN_THRESHOLD = params.get("SIG_RETURN_THRESHOLD", 0.0005)
    RET_HORIZONS = params.get("RET_HORIZONS", [1,3,5,10])
    score_h = params.get("SCORE_HORIZON", SCORE_HORIZON_GLOBAL)

    # TWAP constraints
    TWAP_LOOKBACK = params.get("TWAP_LOOKBACK", 120)           # bars for TWAP
    TWAP_SLOPE_LOOKBACK = params.get("TWAP_SLOPE_LOOKBACK", 5) # slope window
    TWAP_PROX_POINTS = params.get("TWAP_PROX_POINTS", 10.0)    # |POC - TWAP| <= this

    # Copy & precompute rolling features
    df = BASE_DF.copy()
    df["vol_roll"] = df["volume"].rolling(VOL_LOOKBACK, min_periods=1).mean()
    df["twap"] = df["close"].rolling(TWAP_LOOKBACK, min_periods=1).mean()
    df["twap_slope"] = df["twap"] - df["twap"].shift(TWAP_SLOPE_LOOKBACK)

    times = df.index
    if len(times) < (max(RET_HORIZONS) + max(WINDOW_OBS, WINDOW_MINUTES) + 2):
        return {"ok": False, "reason": "insufficient_data", "params": params}

    window_len = max(WINDOW_OBS, WINDOW_MINUTES) if USE_COUNT_WINDOW else WINDOW_MINUTES

    def get_window_slice(i, use_next=False):
        if USE_COUNT_WINDOW:
            end_ix = i + (1 if use_next else 0)
            start_ix = max(0, end_ix - WINDOW_OBS)
            return df.iloc[start_ix:end_ix]
        else:
            end_time = times[i + (1 if use_next else 0)]
            start_time = end_time - timedelta(minutes=WINDOW_MINUTES)
            return df.loc[(df.index >= start_time) & (df.index < end_time)]

    upper_i = len(times) - max(RET_HORIZONS) - 1
    results = []
    cancelled = 0

    # Skip-counters for diagnostics
    reasons = {
        "win_too_short": 0,
        "liq_vol_filter": 0,
        "profile_none": 0,
        "twap_missing": 0,
        "twap_poc_far": 0,
        "twap_dir_mismatch": 0,
        "not_outside_band": 0,
        "min_dist_fail": 0,
        "t1_win_too_short": 0,
        "t1_profile_none": 0,
        "cancelled_inside_or_unstable": 0,
    }

    PRINT_EVERY = params.get("PRINT_EVERY", 50000)

    for i in range(window_len, upper_i):
        if PRINT_EVERY and (i % PRINT_EVERY == 0):
            pct = 100.0 * i / upper_i
            worst = max(reasons.items(), key=lambda kv: kv[1])[0]
            print(f"[pid {os.getpid()}] {pct:.1f}% | sig={len(results)} canc={cancelled} | worst='{worst}' -> {reasons[worst]}")

        # ----- build profile on t-window -----
        wdf_t = get_window_slice(i, use_next=False)
        if len(wdf_t) < MIN_WINDOW_ROWS:
            reasons["win_too_short"] += 1; continue

        # Liquidity filter
        if (wdf_t["vol_roll"].iloc[-1] < MIN_ROLLING_VOL):
            reasons["liq_vol_filter"] += 1; continue

        prof = build_volume_profile(wdf_t["close"].values, wdf_t["volume"].values, BIN_SIZE)
        if prof is None:
            reasons["profile_none"] += 1; continue
        _, _, poc, p_lo, p_hi = prof

        t = times[i]
        price_t = df.iloc[i]["close"]
        twap_t = df.iloc[i]["twap"]
        twap_slope = df.iloc[i]["twap_slope"]

        # ----- TWAP constraints -----
        if not np.isfinite(twap_t) or not np.isfinite(twap_slope):
            reasons["twap_missing"] += 1; continue
        if abs(poc - twap_t) > TWAP_PROX_POINTS:
            reasons["twap_poc_far"] += 1; continue

        # direction must match TWAP slope
        exp_dir = np.sign(poc - price_t)  # +1 long (POC above), -1 short (POC below)
        twap_dir = np.sign(twap_slope)
        if exp_dir == 0 or twap_dir == 0 or exp_dir != twap_dir:
            reasons["twap_dir_mismatch"] += 1; continue

        # ----- Outside central 30% band (or absolute fallback) -----
        outside = False
        if np.isfinite(p_lo) and np.isfinite(p_hi):
            outside = (price_t < p_lo) or (price_t > p_hi)
        if not outside and abs(price_t - poc) > ABS_THRESHOLD_POINTS:
            outside = True
        if not outside:
            reasons["not_outside_band"] += 1; continue

        # ----- Minimum raw distance from POC -----
        if abs(price_t - poc) < MIN_DIST_POINTS:
            reasons["min_dist_fail"] += 1; continue

        # ----- Cancel rule at t+1 + POC stability -----
        wdf_t1 = get_window_slice(i, use_next=True)
        if len(wdf_t1) < MIN_WINDOW_ROWS:
            reasons["t1_win_too_short"] += 1; continue

        prof1 = build_volume_profile(wdf_t1["close"].values, wdf_t1["volume"].values, BIN_SIZE)
        if prof1 is None:
            reasons["t1_profile_none"] += 1; continue
        _, _, poc1, p_lo1, p_hi1 = prof1

        price_t1 = df.iloc[i+1]["close"]
        # back inside band next bar?
        inside_next = False
        if np.isfinite(p_lo1) and np.isfinite(p_hi1):
            inside_next = (p_lo1 <= price_t1 <= p_hi1)
        else:
            inside_next = (abs(price_t1 - poc1) <= ABS_THRESHOLD_POINTS)

        poc_shift = abs(poc1 - poc) if (np.isfinite(poc1) and np.isfinite(poc)) else np.inf
        one_min_ret_mag = abs((price_t1 - price_t) / price_t)
        if (inside_next and (one_min_ret_mag < SIG_RETURN_THRESHOLD)) or (poc_shift > POC_STABILITY_POINTS):
            cancelled += 1
            reasons["cancelled_inside_or_unstable"] += 1
            continue

        rec = {"time": t, "price": price_t, "poc": poc, "twap": twap_t, "twap_slope": twap_slope}
        for h in RET_HORIZONS:
            fut_price = df.iloc[i+h]["close"]
            rec[f"ret_{h}m"] = ret_towards_poc(price_t, fut_price, poc)
        results.append(rec)

    if not results:
        return {"ok": True, "n": 0, "t_like": np.nan, "mean": np.nan, "params": params, "reasons": reasons}

    res = pd.DataFrame(results)
    col = f"ret_{score_h}m"
    strat = res[col].dropna()
    score_t = t_like(strat)
    score_mean = strat.mean()
    return {
        "ok": True,
        "n": len(res),
        "t_like": float(score_t) if np.isfinite(score_t) else np.nan,
        "mean": float(score_mean) if np.isfinite(score_mean) else np.nan,
        "params": params,
        "reasons": reasons,
    }

# =====================
# Parameter grid (ATR-free & feasible)
#   ~3*3*3*3*3*2*3*3*3*3 ≈ a few hundred runs — reasonable for an M4
# =====================
def build_param_grid():
    grid = []
    for WINDOW_OBS in [180, 240, 360]:
        for BIN_SIZE in [0.5, 1.0, 2.0]:
            for ABS_THRESHOLD_POINTS in [20.0, 30.0, 40.0]:
                for POC_STABILITY_POINTS in [3.0, 5.0, 8.0]:
                    for MIN_DIST_POINTS in [1.0, 2.0, 3.0]:
                        for MIN_ROLLING_VOL in [30, 50]:
                            for VOL_LOOKBACK in [10, 20, 30]:
                                for TWAP_LOOKBACK in [60, 120, 240]:
                                    for TWAP_SLOPE_LOOKBACK in [3, 5, 10]:
                                        for TWAP_PROX_POINTS in [5.0, 10.0, 15.0]:
                                            grid.append({
                                                "USE_COUNT_WINDOW": True,
                                                "WINDOW_OBS": WINDOW_OBS,
                                                "WINDOW_MINUTES": WINDOW_OBS,   # unused in count-window mode
                                                "MIN_WINDOW_ROWS": max(40, WINDOW_OBS//3),
                                                "BIN_SIZE": BIN_SIZE,
                                                "MIN_DIST_POINTS": MIN_DIST_POINTS,
                                                "ABS_THRESHOLD_POINTS": ABS_THRESHOLD_POINTS,
                                                "POC_STABILITY_POINTS": POC_STABILITY_POINTS,
                                                "MIN_ROLLING_VOL": MIN_ROLLING_VOL,
                                                "VOL_LOOKBACK": VOL_LOOKBACK,
                                                "SIG_RETURN_THRESHOLD": 0.0005,
                                                "RET_HORIZONS": [1,3,5,10],
                                                "SCORE_HORIZON": SCORE_HORIZON_GLOBAL,
                                                "TWAP_LOOKBACK": TWAP_LOOKBACK,
                                                "TWAP_SLOPE_LOOKBACK": TWAP_SLOPE_LOOKBACK,
                                                "TWAP_PROX_POINTS": TWAP_PROX_POINTS,
                                                "PRINT_EVERY": 50000,
                                            })
    return grid

# =====================
# Parallel grid search
# =====================
def run_grid_search(grid, max_workers=None):
    out = []; err = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_backtest, g) for g in grid]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:
                res = {"ok": False, "reason": f"exception: {type(e).__name__}: {e}"}
            out.append(res)
            if not res.get("ok"):
                err += 1
                if err <= 3:
                    print(f"[grid][error sample] {res.get('reason')}")
            if i % GRID_PROGRESS_EVERY == 0:
                print(f"[grid] {i}/{len(grid)} completed... (errors so far: {err})")
    return out

# =====================
# Summaries & selection
# =====================
def select_and_print_best(results):
    rows = []
    agg_reasons = {}
    for r in results:
        if not r.get("ok"): continue
        p = r["params"]
        rows.append({
            "n": r.get("n", 0),
            "t_like": r.get("t_like", np.nan),
            "mean": r.get("mean", np.nan),
            "WINDOW_OBS": p["WINDOW_OBS"],
            "BIN_SIZE": p["BIN_SIZE"],
            "ABS_THRESHOLD_POINTS": p["ABS_THRESHOLD_POINTS"],
            "POC_STABILITY_POINTS": p["POC_STABILITY_POINTS"],
            "MIN_DIST_POINTS": p["MIN_DIST_POINTS"],
            "MIN_ROLLING_VOL": p["MIN_ROLLING_VOL"],
            "VOL_LOOKBACK": p["VOL_LOOKBACK"],
            "TWAP_LOOKBACK": p["TWAP_LOOKBACK"],
            "TWAP_SLOPE_LOOKBACK": p["TWAP_SLOPE_LOOKBACK"],
            "TWAP_PROX_POINTS": p["TWAP_PROX_POINTS"],
        })
        # aggregate skip reasons
        rs = r.get("reasons", {})
        for k, v in rs.items():
            agg_reasons[k] = agg_reasons.get(k, 0) + v

    if not rows:
        print("No successful runs.")
        return None, None
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved all grid results -> {RESULTS_CSV}")

    print("\nTop by t-like (higher is better):")
    top_t = df.sort_values(["t_like","n"], ascending=[False, False]).head(TOP_K)
    print(top_t.to_string(index=False))

    print("\nTop by mean return (higher is better):")
    top_m = df.sort_values(["mean","n"], ascending=[False, False]).head(TOP_K)
    print(top_m.to_string(index=False))

    if agg_reasons:
        worst = sorted(agg_reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("\nMost common skip/cancel reasons across grid (top 5):")
        for k, v in worst:
            print(f"  {k}: {v}")

    # Build best params (by t-like)
    best_row = top_t.iloc[0]
    best_params = {
        "USE_COUNT_WINDOW": True,
        "WINDOW_OBS": int(best_row["WINDOW_OBS"]),
        "WINDOW_MINUTES": int(best_row["WINDOW_OBS"]),
        "MIN_WINDOW_ROWS": max(40, int(best_row["WINDOW_OBS"])//3),
        "BIN_SIZE": float(best_row["BIN_SIZE"]),
        "MIN_DIST_POINTS": float(best_row["MIN_DIST_POINTS"]),
        "ABS_THRESHOLD_POINTS": float(best_row["ABS_THRESHOLD_POINTS"]),
        "POC_STABILITY_POINTS": float(best_row["POC_STABILITY_POINTS"]),
        "MIN_ROLLING_VOL": int(best_row["MIN_ROLLING_VOL"]),
        "VOL_LOOKBACK": int(best_row["VOL_LOOKBACK"]),
        "SIG_RETURN_THRESHOLD": 0.0005,
        "RET_HORIZONS": [1,3,5,10],
        "SCORE_HORIZON": SCORE_HORIZON_GLOBAL,
        "TWAP_LOOKBACK": int(best_row["TWAP_LOOKBACK"]),
        "TWAP_SLOPE_LOOKBACK": int(best_row["TWAP_SLOPE_LOOKBACK"]),
        "TWAP_PROX_POINTS": float(best_row["TWAP_PROX_POINTS"]),
        "PRINT_EVERY": 50000,
    }
    return best_params, df

# =====================
# Re-run best config (prints only)
# =====================
def rerun_full(best_params):
    res = run_backtest(best_params)
    print("\n=== Best configuration (re-run metrics) ===")
    print({k: best_params[k] for k in sorted(best_params)})
    print(f"signals={res.get('n')}  t_like={res.get('t_like')}  mean={res.get('mean')}")
    if "reasons" in res:
        worst = sorted(res["reasons"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("Top skip/cancel reasons in best run:")
        for k, v in worst:
            print(f"  {k}: {v}")

# =====================
# Main
# =====================
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    print(f"Dataset bars after filters: {len(BASE_DF)}")
    grid = build_param_grid()
    print(f"Total grid combinations: {len(grid)}")

    results = run_grid_search(grid, max_workers=MAX_WORKERS)
    best_params, df_all = select_and_print_best(results)
    if best_params is not None:
        rerun_full(best_params)
