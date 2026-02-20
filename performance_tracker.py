
class PerformanceTracker:
    def __init__(self):
        self.trades = []

    def record_trade(self, symbol, mode, result_pct):
        self.trades.append({
            "symbol": symbol,
            "mode": mode,
            "result_pct": result_pct
        })

    def weekly_stats(self):
        if not self.trades:
            return {"trades": 0}

        wins = [t for t in self.trades if t["result_pct"] > 0]
        losses = [t for t in self.trades if t["result_pct"] <= 0]

        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(self.trades) * 100, 2)
        }

    def compare_modes(self):
        modes = {}
        for t in self.trades:
            modes.setdefault(t["mode"], []).append(t["result_pct"])

        summary = {}
        for mode, vals in modes.items():
            summary[mode] = {
                "count": len(vals),
                "avg_result": round(sum(vals) / len(vals), 2)
            }
        return summary
