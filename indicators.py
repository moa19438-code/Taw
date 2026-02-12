from __future__ import annotations

from typing import List, Optional, Tuple


def sma(values: List[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: List[float], period: int) -> Optional[float]:
    """Exponential moving average over the full series; returns last EMA."""
    if period <= 0 or len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if period <= 0 or len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if period <= 0 or len(closes) < period + 1 or len(highs) != len(lows) or len(lows) != len(closes):
        return None
    trs: List[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def bollinger_bands(closes: List[float], period: int = 20, stdev_mult: float = 2.0) -> Optional[Tuple[float, float, float, float]]:
    """Returns (mid, upper, lower, pct_b) where pct_b in [0..1] (can exceed)."""
    if period <= 1 or len(closes) < period:
        return None
    window = closes[-period:]
    mid = sum(window) / period
    var = sum((x - mid) ** 2 for x in window) / period
    sd = var ** 0.5
    upper = mid + stdev_mult * sd
    lower = mid - stdev_mult * sd
    last = closes[-1]
    pct_b = (last - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
    return mid, upper, lower, pct_b


def macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Tuple[float, float, float]]:
    """Returns (macd_line, signal_line, histogram)."""
    if min(fast, slow, signal) <= 0 or len(closes) < slow + signal:
        return None
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    if fast_ema is None or slow_ema is None:
        return None
    macd_line = fast_ema - slow_ema

    # Build MACD series to compute signal EMA properly
    # (approx by recomputing EMA on rolling MACD values)
    macd_series: List[float] = []
    for i in range(len(closes)):
        sub = closes[: i + 1]
        fe = ema(sub, fast)
        se = ema(sub, slow)
        if fe is None or se is None:
            continue
        macd_series.append(fe - se)
    if len(macd_series) < signal:
        return None
    signal_line = ema(macd_series, signal)
    if signal_line is None:
        return None
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Optional[Tuple[float, float]]:
    """Returns (%K, %D)."""
    if min(k_period, d_period) <= 0 or len(closes) < k_period or len(highs) != len(lows) or len(lows) != len(closes):
        return None
    hh = max(highs[-k_period:])
    ll = min(lows[-k_period:])
    if hh - ll == 0:
        k = 50.0
    else:
        k = 100.0 * (closes[-1] - ll) / (hh - ll)

    # %D: SMA of last d_period %K values
    k_vals: List[float] = []
    for i in range(len(closes) - k_period, len(closes)):
        sub_h = highs[: i + 1]
        sub_l = lows[: i + 1]
        sub_c = closes[: i + 1]
        if len(sub_c) < k_period:
            continue
        hh_i = max(sub_h[-k_period:])
        ll_i = min(sub_l[-k_period:])
        if hh_i - ll_i == 0:
            k_vals.append(50.0)
        else:
            k_vals.append(100.0 * (sub_c[-1] - ll_i) / (hh_i - ll_i))
    if len(k_vals) < d_period:
        d = k
    else:
        d = sum(k_vals[-d_period:]) / d_period
    return k, d


def _wilder_smooth(values: List[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[Tuple[float, float, float]]:
    """Returns (ADX, +DI, -DI)."""
    if period <= 0 or len(closes) < period + 1 or len(highs) != len(lows) or len(lows) != len(closes):
        return None

    plus_dm: List[float] = []
    minus_dm: List[float] = []
    tr_list: List[float] = []

    for i in range(1, len(closes)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
        mdm = down_move if (down_move > up_move and down_move > 0) else 0.0
        plus_dm.append(pdm)
        minus_dm.append(mdm)
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)

    if len(tr_list) < period:
        return None

    # Wilder smoothing (simple approximation for last value)
    tr14 = _wilder_smooth(tr_list, period)
    pdm14 = _wilder_smooth(plus_dm, period)
    mdm14 = _wilder_smooth(minus_dm, period)
    if tr14 is None or tr14 == 0 or pdm14 is None or mdm14 is None:
        return None

    pdi = 100.0 * (pdm14 / tr14)
    mdi = 100.0 * (mdm14 / tr14)
    dx = 100.0 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) != 0 else 0.0

    # ADX: SMA of last period DX values (approx)
    dx_series: List[float] = []
    for j in range(period, len(tr_list) + 1):
        trn = sum(tr_list[j - period : j])
        pdmn = sum(plus_dm[j - period : j])
        mdmn = sum(minus_dm[j - period : j])
        if trn == 0:
            dx_series.append(0.0)
            continue
        pdi_j = 100.0 * (pdmn / trn)
        mdi_j = 100.0 * (mdmn / trn)
        dx_j = 100.0 * abs(pdi_j - mdi_j) / (pdi_j + mdi_j) if (pdi_j + mdi_j) != 0 else 0.0
        dx_series.append(dx_j)

    if len(dx_series) < period:
        adx_val = dx
    else:
        adx_val = sum(dx_series[-period:]) / period

    return adx_val, pdi, mdi


def obv(closes: List[float], volumes: List[float]) -> Optional[float]:
    """On-Balance Volume; returns last OBV value (relative)."""
    if len(closes) < 2 or len(closes) != len(volumes):
        return None
    v = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            v += volumes[i]
        elif closes[i] < closes[i - 1]:
            v -= volumes[i]
    return v


def vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], period: int = 20) -> Optional[float]:
    """Approx VWAP using typical price for daily bars over `period`."""
    if period <= 0 or len(closes) < period or len(highs) != len(lows) or len(lows) != len(closes) or len(closes) != len(volumes):
        return None
    tp = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(len(closes) - period, len(closes))]
    vol = volumes[-period:]
    denom = sum(vol)
    if denom == 0:
        return None
    num = sum(tp[i] * vol[i] for i in range(period))
    return num / denom
