# US Stocks Scanner + Executor Bot (Render-ready)

This service scans the US stock universe (via Alpaca Assets API), scores candidates, sends results to Telegram, and can optionally place Alpaca paper/live trades using bracket orders (entry + take-profit + stop-loss).

## What it does
- Builds a **dynamic universe** from Alpaca: active + tradable US equities.
- Filters & scores symbols using lightweight technical heuristics (trend, momentum, volatility, volume).
- Telegram command **/analyze** triggers a scan and returns the top picks.
- HTTP endpoint **/scan?key=...** triggers a scan (for cron jobs).
- Optional execution: if `AUTO_TRADE=true` and safety latches are enabled, it places **bracket orders** on the top picks.

## Safety (important)
Trading is risky. This project provides automation infrastructure, not a profit guarantee.

Two safety latches must be ON to place any order:
- `EXECUTE_TRADES=true`
- `ALLOW_LIVE_TRADING=true`

Start with Alpaca **paper trading**.

## Deploy on Render
- Create a new **Web Service** for this repo.
- Start command:
  `gunicorn main:app --bind 0.0.0.0:$PORT`
- Add environment variables (see below).

## Environment variables
Required:
- `RUN_KEY` : secret key for `/scan`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID` : numeric chat id to send messages to
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ALPACA_BASE_URL` : paper: `https://paper-api.alpaca.markets` or live: `https://api.alpaca.markets`

Recommended (scanner):
- `UNIVERSE_MAX=1500` (limit number of symbols scanned)
- `MIN_PRICE=2`
- `MAX_PRICE=250`
- `MIN_AVG_DOLLAR_VOL=2000000` (2M)
- `LOOKBACK_DAYS=60`
- `TOP_N=12`

Risk & execution:
- `BROKER=alpaca`
- `AUTO_TRADE=false` (default)
- `RISK_PER_TRADE_PCT=0.25`  (risk per trade as % of equity; conservative)
- `MAX_DAILY_TRADES=2`
- `TP_R_MULT=1.8`  (take profit = R * this multiplier)
- `SL_ATR_MULT=1.5` (stop = ATR * this multiplier)
- `EXECUTE_TRADES=false`
- `ALLOW_LIVE_TRADING=false`

## Endpoints
- `/` health
- `/scan?key=RUN_KEY` run scan (and optional trade)
- `/orders?key=RUN_KEY` last stored orders
- `/status?key=RUN_KEY` quick status snapshot

## Telegram commands
- `/analyze` run scan now
- `/status` show status
- `/orders` show last orders
- `/help`
