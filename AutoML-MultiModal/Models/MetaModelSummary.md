# 🧩 Meta-Model (Logistic Regression) Summary

## 1. Purpose
The logistic regression model acts as a **meta-model** — learning **when to trust each base model** under different market conditions, as well as **when to trust each horizon 5-60**.

It doesn’t replace the base models; it **calibrates and weights** them dynamically.

---

## 2. Inputs

**Meta-model features:**
- Outputs (probabilities or scores) from all base models:
  - Price Directional Model  
  - Volatility Model  
  - Liquidity Model  
  - Drift / OFI Models  
- Current market context indicators:
  - `ret_1_z60` → return z-score (directional bias)
  - `vol_z_60` → volatility z-score (market activity)
  - `liq_z_60` → volume z-score (liquidity regime)
- Optional one-hot regime flags:
  - e.g. `high_vol`, `low_vol`, `high_drift`, `ofi_buy`, etc.

---

## 3. Targets
Same as the main directional task:
- **Binary:** next 5–60 min cumulative return direction (`up` / `down`)
- **Regression (optional):** magnitude of move over future horizons

---

## 4. Role Separation

| Layer | Learns | Inputs | Outputs |
|--------|--------|--------|----------|
| **Base Models** | Market-specific signals (price, vol, OFI, etc.) | Raw causal features (past 120 bars) | Forecasts or probabilities |
| **Meta-Model (LogReg)** | Reliability and conditional weighting | Base model outputs + current context | Final unified forecast |

---

## 5. Why Context Features Are Needed
Including real-time context (`ret_z`, `vol_z`, `liq_z`, etc.) **does not violate** the modular structure —  
it helps the meta-model learn **when** each base model performs best.

- High vol? → Trust volatility or OFI model more  
- Low vol? → Price model is more reliable  
- Low liquidity? → Downweight all short-horizon signals

---

## ✅ Summary

> The logistic regression model is the **portfolio manager** of your ensemble —  
> it looks at current market conditions and decides **which expert to trust most** at each moment.

