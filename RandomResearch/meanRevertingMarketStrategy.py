import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

# ------------------- Config -------------------
@dataclass
class MomConfig:
    cooldown: int = 5
    random_seed: int = 42
    draws_per_event: int = 1
    alpha: float = 0.05

# ------------------- Utils -------------------
def benjamini_hochberg(pvals: List[float]) -> List[float]:
    if not pvals:
        return []
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals, dtype=float)[order]
    adj = np.empty(m, dtype=float)
    cummin = 1.0
    for i in range(m, 0, -1):
        val = min(ranked[i-1] * m / i, 1.0)
        cummin = min(cummin, val)
        adj[i-1] = cummin
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out.tolist()

def two_way_cost_log(bps: float) -> float:
    if bps <= 0:
        return 0.0
    c = bps / 1e4
    return float(-2.0 * np.log1p(-c))

def score_from_stats(ev_mean: float, rnd_mean: float, p: float) -> float:
    if not (np.isfinite(ev_mean) and np.isfinite(rnd_mean) and np.isfinite(p)):
        return -np.inf
    lift = ev_mean - rnd_mean
    return float(lift * (-np.log10(max(p, 1e-300))))

# ------------------- Indicators -------------------
def precompute_emas(close: pd.Series, spans: List[int]) -> Dict[int, np.ndarray]:
    out = {}
    s = close.astype(float)
    for sp in spans:
        out[sp] = s.ewm(span=sp, adjust=False).mean().to_numpy(dtype=float)
    return out

def compute_rsi(close: pd.Series, window: int = 14) -> np.ndarray:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_numpy(dtype=float)

def precompute_rsis(close: pd.Series, windows: List[int]) -> Dict[int, np.ndarray]:
    out = {}
    for w in windows:
        out[w] = compute_rsi(close, window=w)
    return out

def base_bull_cross_indices(ema_s: np.ndarray, ema_l: np.ndarray, rsi: np.ndarray, rsi_threshold: float = 65.0) -> np.ndarray:
    cond = (
        (ema_s[:-1] <= ema_l[:-1]) &
        (ema_s[1:] > ema_l[1:]) &
        (rsi[1:] > rsi_threshold)
    )
    return (np.nonzero(cond)[0] + 1).astype(np.int64)

def base_bear_cross_indices(ema_s: np.ndarray, ema_l: np.ndarray, rsi: np.ndarray, rsi_threshold: float = 35.0) -> np.ndarray:
    cond = (
        (ema_s[:-1] >= ema_l[:-1]) &
        (ema_s[1:] < ema_l[1:]) &
        (rsi[1:] < rsi_threshold)
    )
    return (np.nonzero(cond)[0] + 1).astype(np.int64)

# ------------------- Core -------------------
def filter_nonoverlap_and_horizon(price_idx: np.ndarray, horizon: int, n_close: int) -> np.ndarray:
    if price_idx.size == 0:
        return price_idx
    max_i = n_close - 1 - horizon
    valid = price_idx[price_idx <= max_i]
    if valid.size == 0:
        return valid
    keep = []
    last = -10**12
    for i in valid:
        if i - last >= horizon:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.int64)

def forward_sums_from_cumsum(ret_csum: np.ndarray, ret_idx: np.ndarray, horizon: int) -> np.ndarray:
    if ret_idx.size == 0:
        return np.empty(0, dtype=float)
    a = ret_idx + 1
    b = ret_idx + 1 + horizon
    return ret_csum[b] - ret_csum[a]

def sample_random_forward_sums(ret_len: int, horizon: int, n: int, seed: int, ret_csum: np.ndarray) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=float)
    lo = horizon
    hi = ret_len - horizon - 1
    if hi < lo:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    size_space = (hi - lo + 1)
    replace = n > size_space
    js = rng.choice(np.arange(lo, hi + 1, dtype=np.int64), size=n, replace=replace)
    return forward_sums_from_cumsum(ret_csum, js, horizon)

def compute_stats(event_fw: np.ndarray, random_fw: np.ndarray) -> Dict[str, float]:
    out = {
        "event_n": int(event_fw.size),
        "random_n": int(random_fw.size),
        "event_mean": float(np.mean(event_fw)) if event_fw.size else np.nan,
        "random_mean": float(np.mean(random_fw)) if random_fw.size else np.nan,
        "event_median": float(np.median(event_fw)) if event_fw.size else np.nan,
        "random_median": float(np.median(random_fw)) if random_fw.size else np.nan,
        "event_std": float(np.std(event_fw, ddof=1)) if event_fw.size > 1 else np.nan,
        "random_std": float(np.std(random_fw, ddof=1)) if random_fw.size > 1 else np.nan,
    }
    if event_fw.size > 1 and random_fw.size > 1:
        t_stat, t_p = stats.ttest_ind(event_fw, random_fw, equal_var=False, nan_policy="omit")
        u_stat, u_p = stats.mannwhitneyu(event_fw, random_fw, alternative="two-sided")
        ks_stat, ks_p = stats.ks_2samp(event_fw, random_fw, alternative="two-sided", mode="auto")
        out.update({
            "ttest_t": float(t_stat), "ttest_p": float(t_p),
            "mwu_U": float(u_stat), "mwu_p": float(u_p),
            "ks_D": float(ks_stat), "ks_p": float(ks_p),
        })
    return out

# ------------------- Plots -------------------
def plot_histogram(event_fw: np.ndarray, random_fw: np.ndarray, horizon: int, regime: str):
    plt.figure()
    plt.hist(event_fw, bins=50, alpha=0.7, density=True, label="Event")
    plt.hist(random_fw, bins=50, alpha=0.5, density=True, label="Random")
    plt.title(f"{regime} | Forward Log-Return ({horizon} bars): Histogram")
    plt.xlabel("Cumulative log-return")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_ecdf(event_fw: np.ndarray, random_fw: np.ndarray, horizon: int, regime: str):
    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x)+1) / len(x)
        return x, y
    xe, ye = ecdf(event_fw)
    xr, yr = ecdf(random_fw)
    plt.figure()
    plt.plot(xe, ye, label="Event")
    plt.plot(xr, yr, label="Random")
    plt.title(f"{regime} | Forward Log-Return ({horizon} bars): ECDF")
    plt.xlabel("Cumulative log-return")
    plt.ylabel("ECDF")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

def plot_box(event_fw: np.ndarray, random_fw: np.ndarray, horizon: int, regime: str):
    plt.figure()
    plt.boxplot([event_fw, random_fw], labels=["Event", "Random"], showmeans=True)
    plt.title(f"{regime} | Forward Log-Return ({horizon} bars): Boxplot")
    plt.ylabel("Cumulative log-return")
    plt.show()

# ------------------- (TRAIN + TEST functions stay same as I gave before) -------------------


# ------------------- Train Grid -------------------
def run_train_grid_parallel(
    df_train: pd.DataFrame,
    short_range: range,
    long_range: range,
    horizon_range: range,
    rsi_range: range,
    base_cfg: MomConfig,
    direction: str,
    min_events: int,
    two_way_cost_bps: float,
    log: logging.Logger,
    max_workers: int = None,
):
    close = df_train['close'].astype(float)
    n_close = close.size
    ret = np.log(close).diff().dropna().to_numpy(dtype=float)
    ret_len = ret.size
    ret_csum = np.concatenate(([0.0], np.cumsum(ret))).astype(float)

    all_spans = sorted(set(list(short_range) + list(long_range)))
    ema_map = precompute_emas(close, all_spans)
    rsi_map = precompute_rsis(close, list(rsi_range))
    cost_log = two_way_cost_log(two_way_cost_bps)

    pair_list = [(s, l) for s in short_range for l in long_range if s < l]
    total_pairs = len(pair_list)
    total_triples = total_pairs * len(horizon_range) * len(rsi_range)
    log.info(f"TRAIN grid combos: {total_triples:,}")

    results_train: List[Dict[str, float]] = []
    pvals_train: List[float] = []

    def worker(pair):
        s, l = pair
        ema_s = ema_map[s]
        ema_l = ema_map[l]
        local_results = []
        for rsi_w in rsi_range:
            rsi = rsi_map[rsi_w]
            if direction == "bull":
                base_idx = base_bull_cross_indices(ema_s, ema_l, rsi)
            else:
                base_idx = base_bear_cross_indices(ema_s, ema_l, rsi)
            for H in horizon_range:
                filt_i = filter_nonoverlap_and_horizon(base_idx, H, n_close)
                if filt_i.size == 0:
                    continue
                j = filt_i - 1
                j = j[(j >= 0) & (j <= ret_len - H - 1)]
                if j.size < min_events:
                    continue
                ev_fw = forward_sums_from_cumsum(ret_csum, j, H)
                if cost_log != 0.0 and ev_fw.size:
                    ev_fw = ev_fw - cost_log
                n_random = max(ev_fw.size * base_cfg.draws_per_event, 1)
                seed = (base_cfg.random_seed ^ (s * 1_000_003) ^ (l * 10_000_019) ^ (H * 97_000_031) ^ rsi_w) & 0x7fffffff
                rnd_fw = sample_random_forward_sums(ret_len, H, n_random, seed, ret_csum)
                if cost_log != 0.0 and rnd_fw.size:
                    rnd_fw = rnd_fw - cost_log
                stats = compute_stats(ev_fw, rnd_fw)
                if stats["event_n"] < min_events or not np.isfinite(stats.get("event_mean", np.nan)):
                    continue
                ev_mean = stats.get("event_mean", np.nan)
                rnd_mean = stats.get("random_mean", np.nan)
                p = stats.get("ttest_p", np.nan)
                sc = score_from_stats(ev_mean, rnd_mean, p)
                ev_std = stats.get("event_std", np.nan)
                sharpe = (ev_mean / ev_std) if (np.isfinite(ev_std) and ev_std > 0) else np.nan
                local_results.append({
                    "short": s, "long": l, "horizon": H, "rsi_window": rsi_w,
                    "score": sc,
                    "lift": float((ev_mean or 0.0) - (rnd_mean or 0.0)),
                    "p": float(p) if np.isfinite(p) else np.nan,
                    "event_n": int(stats["event_n"]),
                    "random_n": int(stats["random_n"]),
                    "effective_n": int(stats["event_n"]),
                    "sharpe_event": float(sharpe) if np.isfinite(sharpe) else np.nan,
                })
        return local_results

    if max_workers is None:
        max_workers = os.cpu_count() or 8
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, pair): pair for pair in pair_list}
        for fut in as_completed(futures):
            res_list = fut.result()
            if res_list:
                results_train.extend(res_list)
                pvals_train.extend([d["p"] for d in res_list if np.isfinite(d["p"])])
            completed += 1
            if completed % 20 == 0 or completed == total_pairs:
                pct = 100.0 * completed / total_pairs
                log.info(f"[{completed:,}/{total_pairs:,}] {pct:5.1f}% pairs done; "
                         f"viable combos so far: {len(results_train):,}")
    return results_train, pvals_train

# ------------------- Test Evaluation -------------------
def compute_event_vs_random_ema_cross_test(
    df_test: pd.DataFrame,
    short_span: int,
    long_span: int,
    horizon: int,
    rsi_window: int,
    cfg: MomConfig,
    direction: str = "bull",
    two_way_cost_bps: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    close = df_test['close'].astype(float)
    ret = np.log(close).diff().dropna()
    ema_s = close.ewm(span=short_span, adjust=False).mean()
    ema_l = close.ewm(span=long_span, adjust=False).mean()
    rsi = compute_rsi(close, window=rsi_window)
    if direction == "bull":
        cross = (ema_s.shift(1) <= ema_l.shift(1)) & (ema_s > ema_l) & (rsi > 65)
    else:
        cross = (ema_s.shift(1) >= ema_l.shift(1)) & (ema_s < ema_l) & (rsi < 35)
    idx_all = np.where(cross.to_numpy(dtype=bool))[0].tolist()
    cool = max(cfg.cooldown, horizon)
    last = -10**9
    filtered = []
    max_i = len(close) - 1 - horizon
    for i in idx_all:
        if i > max_i:
            continue
        if i - last >= cool:
            filtered.append(i)
            last = i
    events = [i-1 for i in filtered if i-1 >= 0 and i-1 <= (len(ret) - horizon - 1)]
    ret_np = ret.to_numpy(dtype=float)
    ret_csum = np.concatenate(([0.0], np.cumsum(ret_np)))
    ev_fw = forward_sums_from_cumsum(ret_csum, np.asarray(events, dtype=np.int64), horizon)
    rnd_n = max(ev_fw.size * cfg.draws_per_event, 1)
    seed = (cfg.random_seed ^ (short_span * 1_000_003) ^ (long_span * 10_000_019) ^ (horizon * 97_000_031) ^ rsi_window) & 0x7fffffff
    rnd_fw = sample_random_forward_sums(len(ret_np), horizon, rnd_n, seed, ret_csum)
    cost_log = two_way_cost_log(two_way_cost_bps)
    if cost_log != 0.0:
        ev_fw = ev_fw - cost_log
        rnd_fw = rnd_fw - cost_log
    stats = compute_stats(ev_fw, rnd_fw)
    stats["effective_n"] = int(stats["event_n"])
    stats["horizon"] = int(horizon)
    stats["cost_log"] = float(cost_log)
    return ev_fw, rnd_fw, stats

# ------------------- Main -------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("ema-cross-optim")

    marketDataNQ = '/Users/damensavvasavvi/Desktop/AlgoTrading/marketDataNasdaqFutures/NQ_continuous_backadjusted_1m_cleaned.csv'
    prices = pd.read_csv(marketDataNQ, parse_dates=['timestamp']).set_index('timestamp')
    prices = prices[prices.index < '2024-01-01']
    df = prices[['close']].dropna()
    timeline = df.index
    newyork_session = ((timeline.dayofweek < 5) & (timeline.hour >= 10) & (timeline.hour < 17))
    df = df[newyork_session]
    log_returns = np.log(df / df.shift(1))
    log_returns.fillna(0, inplace=True)
    rolling_vol = log_returns.rolling(20).std()
    median_vol = rolling_vol.median()

    regimes = {
        "nonvolatile": df[rolling_vol < 1.5 * median_vol],
        "volatile": df[rolling_vol >= 1.5 * median_vol],
        "full": df.copy()
    }

    base_cfg = MomConfig(cooldown=5, random_seed=123, draws_per_event=1, alpha=0.05)
    direction = "bull"
    two_way_cost_bps = 0.0
    min_events = 30

    short_range = range(4, 101)
    long_range  = range(20, 201)
    horizon_range = range(1, 21)
    rsi_range = range(5, 201)

    for regime_name, df_regime in regimes.items():
        log.info(f"\n=== Running regime: {regime_name} ===")
        train_frac = 0.70
        split_idx = int(len(df_regime) * train_frac)
        df_train = df_regime.iloc[:split_idx].copy()
        df_test  = df_regime.iloc[split_idx:].copy()
        log.info(f"Train size: {len(df_train):,} rows | Test size: {len(df_test):,} rows")
        results_train, pvals_train = run_train_grid_parallel(
            df_train=df_train,
            short_range=short_range,
            long_range=long_range,
            horizon_range=horizon_range,
            rsi_range=rsi_range,
            base_cfg=base_cfg,
            direction=direction,
            min_events=min_events,
            two_way_cost_bps=two_way_cost_bps,
            log=log,
            max_workers=None,
        )
        if not results_train:
            log.warning("No viable combos on TRAIN.")
            continue
        results_train_sorted = sorted(results_train, key=lambda d: d["score"], reverse=True)
        print("\nTop 15 TRAIN configs:")
        print(pd.DataFrame(results_train_sorted[:15]).to_string(index=False))
        M = len([p for p in pvals_train if np.isfinite(p)])
        best_train = results_train_sorted[0]
        best_p = best_train["p"]
        bonf_p = min(best_p * max(M, 1), 1.0)
        bh_all = benjamini_hochberg([p for p in pvals_train if np.isfinite(p)])
        bh_p = min([bh for p, bh in zip([p for p in pvals_train if np.isfinite(p)], bh_all) if p == best_p] or [min(bh_all) if bh_all else 1.0])
        print(f"\nBest TRAIN: short={best_train['short']}, long={best_train['long']}, horizon={best_train['horizon']}, RSI={best_train['rsi_window']}")
        print(f"TRAIN raw p={best_p:.4g}, Bonferroni(M={M})={bonf_p:.4g}, BH-FDR≈{bh_p:.4g}")
        ev_fw_t, rnd_fw_t, stats_test = compute_event_vs_random_ema_cross_test(
            df_test,
            best_train["short"],
            best_train["long"],
            best_train["horizon"],
            best_train["rsi_window"],
            base_cfg,
            direction=direction,
            two_way_cost_bps=two_way_cost_bps
        )
        print("\n=== TEST Summary (best TRAIN triple) ===")
        for k, v in stats_test.items():
            print(f"{k}: {v}")
        
        if ev_fw_t.size >= 2 and rnd_fw_t.size >= 2:
            plot_histogram(ev_fw_t, rnd_fw_t, horizon=best_train["horizon"], regime=regime_name)
            plot_ecdf(ev_fw_t, rnd_fw_t, horizon=best_train["horizon"], regime=regime_name)
            plot_box(ev_fw_t, rnd_fw_t, horizon=best_train["horizon"], regime=regime_name)

