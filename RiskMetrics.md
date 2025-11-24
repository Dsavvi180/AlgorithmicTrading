# 📊 Trading Performance Ratios Explained

Trading isn’t about the biggest profits — it’s about **stability and efficiency**.  
These ratios measure how much reward you earn for every unit of risk or variance you take.

---

## ⚙️ 1. Sharpe Ratio — *Return per unit of total risk*

### Formula
$$
Sharpe = \frac{E[R_p - R_f]}{\sigma_p}
$$

### Definitions
- **Rₚ** — portfolio or strategy return  
- **R_f** — risk-free rate (≈ 0 for intraday traders)  
- **σₚ** — standard deviation of returns (total volatility)

### Interpretation
The Sharpe ratio measures how much *excess return* you earn per unit of volatility.

| Sharpe | Meaning |
|---------|----------|
| < 1.0 | Too volatile / inconsistent |
| 1.0 – 2.0 | Solid professional level |
| 2.0 – 3.0 | Hedge-fund quality |
| > 3.0 | Exceptional consistency |

### Example
If your strategy makes +1 % per day with 0.5 % standard deviation:

$$
Sharpe = \frac{1 - 0}{0.5} = 2.0
$$

---

## ⚖️ 2. Sortino Ratio — *Return per unit of downside risk*

### Formula
$$
Sortino = \frac{E[R_p - R_f]}{\sigma_{\text{down}}}
$$

### Definitions
- **σ₍down₎** — standard deviation of *negative* returns only  

### Why It Matters
- Sharpe penalises all volatility, good or bad.  
- Sortino ignores upside variance and focuses only on *bad* volatility.  
- Ideal for asymmetric strategies with large winning outliers.

---

## 📈 3. Volatility of the Equity Curve — *Consistency indicator*

### Formula
$$
\sigma_{\text{equity}} = 
\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(r_i - \bar{r})^2}
$$

### Interpretation
- **Low σₑq:** smooth, stable equity growth  
- **High σₑq:** choppy, stressful, and statistically risky  

**Visual intuition**  
- “Stair-step” equity curve → low volatility  
- “Saw-tooth” curve → high volatility  

---

## 🔍 Comparison Table

| Metric | Measures | Good When | Watch Out For |
|---------|-----------|------------|----------------|
| **Sharpe** | Reward per total volatility | ≥ 1.5 | Penalises upside variance |
| **Sortino** | Reward per downside volatility | ≥ 2.0 | Needs many data points |
| **Equity Volatility** | Smoothness of growth | Low | Doesn’t include return scale |

---

## 💡 Example — Why Variance Matters

| Trader | Avg Daily Return | Std Dev | Sharpe |
|---------|------------------|----------|---------|
| **A** | +0.5 % | 0.25 % | **2.0** |
| **B** | +1.0 % | 1.0 % | **1.0** |

Trader A earns *half* the profit but with *one-quarter* the risk →  
more efficient, more scalable, and psychologically sustainable.

---

## 🧮 Quick Python Implementation

```python
import numpy as np

# Example: per-trade or daily returns
returns = np.array([...])
rf = 0.0  # risk-free rate (≈0 for intraday)

sharpe = (np.mean(returns) - rf) / np.std(returns)
downside = np.std([r for r in returns if r < 0])
sortino = (np.mean(returns) - rf) / downside
equity_vol = np.std(returns)

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Equity Curve Volatility: {equity_vol:.4f}")
