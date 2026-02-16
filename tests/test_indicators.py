import math
from core.indicators import sma, ema, rsi, bollinger_bands, macd, stochastic, adx, vwap, obv


def test_sma_basic():
    assert sma([1,2,3,4,5], 5) == 3.0
    assert sma([1,2,3,4,5], 3) == 4.0


def test_ema_runs():
    v = [1]*30
    assert abs(ema(v, 10) - 1.0) < 1e-9


def test_rsi_bounds():
    up = list(range(1, 40))
    r = rsi(up, 14)
    assert r is not None and 80 <= r <= 100


def test_bollinger_returns():
    closes = [100]*30
    bb = bollinger_bands(closes, 20, 2.0)
    assert bb is not None
    mid, up, lo, pctb = bb
    assert abs(mid-100) < 1e-6
    assert abs(up-100) < 1e-6
    assert abs(lo-100) < 1e-6
    assert 0 <= pctb <= 1


def test_macd_constant_zeroish():
    closes = [50]*100
    m = macd(closes, 12, 26, 9)
    assert m is not None
    macd_line, sig, hist = m
    assert abs(macd_line) < 1e-6
    assert abs(sig) < 1e-6
    assert abs(hist) < 1e-6


def test_stochastic_range():
    highs = [10+i for i in range(40)]
    lows = [5+i for i in range(40)]
    closes = [7+i for i in range(40)]
    k, d = stochastic(highs, lows, closes, 14, 3)
    assert 0 <= k <= 100
    assert 0 <= d <= 100


def test_adx_runs():
    highs = [10+i*0.5 for i in range(60)]
    lows = [9+i*0.5 for i in range(60)]
    closes = [9.5+i*0.5 for i in range(60)]
    out = adx(highs, lows, closes, 14)
    assert out is not None
    adx_v, pdi, mdi = out
    assert 0 <= adx_v <= 100
    assert 0 <= pdi <= 100
    assert 0 <= mdi <= 100


def test_vwap_and_obv():
    highs = [10]*30
    lows = [8]*30
    closes = [9]*30
    vols = [100]*30
    assert abs(vwap(highs, lows, closes, vols, 20) - 9.0) < 1e-6
    assert obv(closes, vols) == 0.0
