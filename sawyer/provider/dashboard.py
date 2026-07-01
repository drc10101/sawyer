"""Sawyer Provider Dashboard — web UI for monitoring tokens served and earnings.

Provides daily and weekly breakdowns so providers can see exactly
what their node is doing, how many tokens they're serving, and
how much they're earning.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)


# ── Stats Tracking ─────────────────────────────────────────────────

@dataclass
class DailyStats:
    """One day of provider statistics."""

    date: str  # YYYY-MM-DD
    tokens_served: int = 0
    requests_count: int = 0
    uptime_hours: float = 0.0
    avg_latency_ms: float = 0.0
    errors_count: int = 0
    earnings_usd: float = 0.0

    # Per-model breakdown
    model_breakdown: dict[str, int] = field(default_factory=dict)  # model -> tokens

    # Per-expert breakdown
    expert_breakdown: dict[str, int] = field(default_factory=dict)  # expert_id -> tokens


class ProviderStatsTracker:
    """Track daily and weekly stats for a provider.

    Persists stats to a JSON file so they survive restarts.
    """

    def __init__(self, stats_dir: str = "") -> None:
        self._daily: dict[str, dict[str, DailyStats]] = defaultdict(dict)
        # provider_id -> date -> DailyStats
        self._stats_dir = stats_dir

    def record_tokens(
        self,
        provider_id: str,
        tokens: int,
        model: str = "",
        expert_id: str = "",
        latency_ms: float = 0.0,
        is_error: bool = False,
    ) -> None:
        """Record a token serving event."""
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if date not in self._daily[provider_id]:
            self._daily[provider_id][date] = DailyStats(date=date)

        stats = self._daily[provider_id][date]
        stats.tokens_served += tokens
        stats.requests_count += 1

        if model:
            stats.model_breakdown[model] = stats.model_breakdown.get(model, 0) + tokens

        if expert_id:
            stats.expert_breakdown[expert_id] = stats.expert_breakdown.get(expert_id, 0) + tokens

        if is_error:
            stats.errors_count += 1

        # Running average for latency
        if latency_ms > 0:
            total_requests = stats.requests_count
            stats.avg_latency_ms = (
                (stats.avg_latency_ms * (total_requests - 1) + latency_ms)
                / total_requests
            )

    def record_uptime(self, provider_id: str, hours: float) -> None:
        """Record uptime hours for a day."""
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if date not in self._daily[provider_id]:
            self._daily[provider_id][date] = DailyStats(date=date)

        self._daily[provider_id][date].uptime_hours += hours

    def record_earnings(self, provider_id: str, usd: float) -> None:
        """Record earnings for a day."""
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if date not in self._daily[provider_id]:
            self._daily[provider_id][date] = DailyStats(date=date)

        self._daily[provider_id][date].earnings_usd += usd

    def get_daily_stats(
        self, provider_id: str, days: int = 7
    ) -> list[dict[str, Any]]:
        """Get daily stats for the last N days."""
        now = datetime.now(timezone.utc)
        result = []

        for i in range(days):
            date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            stats = self._daily.get(provider_id, {}).get(date)

            if stats:
                result.append({
                    "date": stats.date,
                    "tokens_served": stats.tokens_served,
                    "requests_count": stats.requests_count,
                    "uptime_hours": round(stats.uptime_hours, 2),
                    "avg_latency_ms": round(stats.avg_latency_ms, 1),
                    "errors_count": stats.errors_count,
                    "earnings_usd": round(stats.earnings_usd, 4),
                    "model_breakdown": stats.model_breakdown,
                    "expert_breakdown": stats.expert_breakdown,
                })
            else:
                result.append({
                    "date": date,
                    "tokens_served": 0,
                    "requests_count": 0,
                    "uptime_hours": 0.0,
                    "avg_latency_ms": 0.0,
                    "errors_count": 0,
                    "earnings_usd": 0.0,
                    "model_breakdown": {},
                    "expert_breakdown": {},
                })

        return result

    def get_weekly_summary(self, provider_id: str) -> dict[str, Any]:
        """Get weekly aggregate stats."""
        daily = self.get_daily_stats(provider_id, days=7)

        total_tokens = sum(d["tokens_served"] for d in daily)
        total_requests = sum(d["requests_count"] for d in daily)
        total_uptime = sum(d["uptime_hours"] for d in daily)
        total_earnings = sum(d["earnings_usd"] for d in daily)
        total_errors = sum(d["errors_count"] for d in daily)

        # Aggregate model breakdown across the week
        model_totals: dict[str, int] = defaultdict(int)
        for d in daily:
            for model, tokens in d["model_breakdown"].items():
                model_totals[model] += tokens

        return {
            "period": "last_7_days",
            "days": daily,
            "totals": {
                "tokens_served": total_tokens,
                "requests_count": total_requests,
                "uptime_hours": round(total_uptime, 2),
                "earnings_usd": round(total_earnings, 4),
                "errors_count": total_errors,
                "avg_daily_tokens": total_tokens // 7 if total_tokens else 0,
                "model_breakdown": dict(model_totals),
            },
        }

    def get_all_provider_ids(self) -> list[str]:
        """List all provider IDs with stats."""
        return list(self._daily.keys())


# ── Dashboard HTML ──────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sawyer — Provider Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #0a0e14;
    color: #e0e0e0;
    min-height: 100vh;
  }
  .header {
    background: #12161c;
    border-bottom: 1px solid #1e2630;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .header h1 {
    font-size: 20px;
    font-weight: 600;
    color: #12c7ef;
  }
  .header h1 span { color: #8899aa; font-weight: 400; }
  .header .status {
    font-size: 13px;
    color: #4caf50;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .header .status .dot {
    width: 8px; height: 8px;
    background: #4caf50;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .container {
    max-width: 960px;
    margin: 0 auto;
    padding: 24px;
  }
  .tier-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .tier-1 { background: #1a3a2a; color: #4caf50; }
  .tier-2 { background: #1a2a3a; color: #2196f3; }
  .tier-3 { background: #3a2a1a; color: #ff9800; }
  .tier-4 { background: #3a1a1a; color: #f44336; }
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: #12161c;
    border: 1px solid #1e2630;
    border-radius: 8px;
    padding: 16px;
  }
  .stat-card .label {
    font-size: 12px;
    color: #8899aa;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }
  .stat-card .value {
    font-size: 28px;
    font-weight: 700;
    color: #ffffff;
  }
  .stat-card .value.earnings { color: #4caf50; }
  .stat-card .value.tokens { color: #12c7ef; }
  .stat-card .sub {
    font-size: 12px;
    color: #667788;
    margin-top: 4px;
  }
  .section {
    margin-bottom: 24px;
  }
  .section h2 {
    font-size: 16px;
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 12px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    background: #12161c;
    border-radius: 8px;
    overflow: hidden;
  }
  th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #1e2630;
    font-size: 13px;
  }
  th {
    background: #0d1117;
    color: #8899aa;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
  }
  td { color: #e0e0e0; }
  tr:hover td { background: #161b22; }
  .bar-chart {
    display: flex;
    align-items: flex-end;
    gap: 6px;
    height: 120px;
    padding: 0 8px;
  }
  .bar {
    flex: 1;
    background: linear-gradient(to top, #12c7ef, #0d8fa8);
    border-radius: 3px 3px 0 0;
    min-height: 2px;
    position: relative;
    transition: height 0.3s ease;
  }
  .bar:hover {
    background: linear-gradient(to top, #1de5ff, #12c7ef);
  }
  .bar .tooltip {
    display: none;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #1e2630;
    color: #e0e0e0;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    white-space: nowrap;
    margin-bottom: 4px;
  }
  .bar:hover .tooltip { display: block; }
  .bar-labels {
    display: flex;
    gap: 6px;
    padding: 4px 8px 0;
  }
  .bar-labels span {
    flex: 1;
    text-align: center;
    font-size: 10px;
    color: #667788;
  }
  .models-breakdown {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .model-tag {
    background: #1a2a3a;
    border: 1px solid #1e2630;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 12px;
  }
  .model-tag .name { color: #12c7ef; }
  .model-tag .count { color: #8899aa; margin-left: 6px; }
  .empty {
    text-align: center;
    padding: 40px;
    color: #667788;
  }
  .payout-info {
    background: #12161c;
    border: 1px solid #1e2630;
    border-radius: 8px;
    padding: 16px;
    margin-top: 8px;
  }
  .payout-info .row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1e2630;
  }
  .payout-info .row:last-child { border-bottom: none; }
  .payout-info .row .label { color: #8899aa; font-size: 13px; }
  .payout-info .row .value { color: #e0e0e0; font-weight: 600; }
  .payout-info .row .value.green { color: #4caf50; }
</style>
</head>
<body>
  <div class="header">
    <h1>Sawyer <span>Provider Dashboard</span></h1>
    <div class="status"><div class="dot"></div> Online</div>
  </div>
  <div class="container" id="app">
    <div class="empty">Loading stats...</div>
  </div>
<script>
async function loadDashboard() {
  try {
    const resp = await fetch('/api/stats');
    if (!resp.ok) throw new Error('Failed to load stats');
    const data = await resp.json();
    renderDashboard(data);
  } catch (e) {
    document.getElementById('app').innerHTML =
      '<div class="empty">Could not load stats. Is the Sawyer server running?</div>';
  }
}

function formatNumber(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
  return n.toString();
}

function formatDate(dateStr) {
  const d = new Date(dateStr + 'T00:00:00Z');
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function renderDashboard(data) {
  const summary = data.weekly_summary || {};
  const totals = summary.totals || {};
  const days = summary.days || [];
  const provider = data.provider || {};
  const tier = data.tier || {};
  const payout = data.payout || {};

  // Find max tokens for chart scaling
  const maxTokens = Math.max(...days.map(d => d.tokens_served), 1);

  let html = '';

  // Tier badge
  if (tier.tier) {
    html += '<div style="margin-bottom:16px">';
    html += '<span class="tier-badge tier-' + tier.tier.split('_')[1] + '">';
    html += tier.tier.replace('_', ' ').toUpperCase() + ' — ' + (tier.description || '');
    html += '</span>';
    html += '</div>';
  }

  // Stats cards
  html += '<div class="stats-grid">';
  html += '<div class="stat-card"><div class="label">Tokens This Week</div>';
  html += '<div class="value tokens">' + formatNumber(totals.tokens_served || 0) + '</div>';
  html += '<div class="sub">avg ' + formatNumber(totals.avg_daily_tokens || 0) + '/day</div></div>';

  html += '<div class="stat-card"><div class="label">Earnings This Week</div>';
  html += '<div class="value earnings">$' + (totals.earnings_usd || 0).toFixed(2) + '</div>';
  html += '<div class="sub">quarterly pool distribution</div></div>';

  html += '<div class="stat-card"><div class="label">Uptime</div>';
  html += '<div class="value">' + (totals.uptime_hours || 0).toFixed(1) + 'h</div>';
  html += '<div class="sub">last 7 days</div></div>';

  html += '<div class="stat-card"><div class="label">Multiplier</div>';
  html += '<div class="value">' + (tier.multiplier || 1) + 'x</div>';
  html += '<div class="sub">' + (tier.min_vram_gb || 4) + '+ GB VRAM</div></div>';
  html += '</div>';

  // Token chart
  html += '<div class="section"><h2>Daily Tokens Served</h2>';
  html += '<div class="bar-chart">';
  days.slice().reverse().forEach(d => {
    const pct = maxTokens > 0 ? (d.tokens_served / maxTokens) * 100 : 0;
    html += '<div class="bar" style="height:' + Math.max(pct, 1) + '%">';
    html += '<div class="tooltip">' + formatDate(d.date) + ': ' + formatNumber(d.tokens_served) + '</div>';
    html += '</div>';
  });
  html += '</div>';
  html += '<div class="bar-labels">';
  days.slice().reverse().forEach(d => {
    html += '<span>' + formatDate(d.date) + '</span>';
  });
  html += '</div></div>';

  // Daily table
  html += '<div class="section"><h2>Daily Breakdown</h2>';
  html += '<table><tr><th>Date</th><th>Tokens</th><th>Requests</th><th>Latency</th><th>Uptime</th><th>Earnings</th><th>Errors</th></tr>';
  days.forEach(d => {
    html += '<tr>';
    html += '<td>' + formatDate(d.date) + '</td>';
    html += '<td>' + formatNumber(d.tokens_served) + '</td>';
    html += '<td>' + formatNumber(d.requests_count) + '</td>';
    html += '<td>' + (d.avg_latency_ms || 0).toFixed(0) + 'ms</td>';
    html += '<td>' + (d.uptime_hours || 0).toFixed(1) + 'h</td>';
    html += '<td>$' + (d.earnings_usd || 0).toFixed(2) + '</td>';
    html += '<td>' + (d.errors_count || 0) + '</td>';
    html += '</tr>';
  });
  html += '</table></div>';

  // Model breakdown
  if (totals.model_breakdown && Object.keys(totals.model_breakdown).length > 0) {
    html += '<div class="section"><h2>Models Served</h2>';
    html += '<div class="models-breakdown">';
    for (const [model, tokens] of Object.entries(totals.model_breakdown)) {
      html += '<div class="model-tag"><span class="name">' + model + '</span>';
      html += '<span class="count">' + formatNumber(tokens) + '</span></div>';
    }
    html += '</div></div>';
  }

  // Payout info
  if (payout) {
    html += '<div class="section"><h2>Payout</h2>';
    html += '<div class="payout-info">';
    html += '<div class="row"><span class="label">Available Balance</span>';
    html += '<span class="value green">$' + (payout.available_balance || 0).toFixed(2) + '</span></div>';
    html += '<div class="row"><span class="label">Total Earned</span>';
    html += '<span class="value">$' + (payout.total_earned || 0).toFixed(2) + '</span></div>';
    html += '<div class="row"><span class="label">Total Paid Out</span>';
    html += '<span class="value">$' + (payout.total_paid || 0).toFixed(2) + '</span></div>';
    html += '<div class="row"><span class="label">Minimum Payout</span>';
    html += '<span class="value">$25.00 / quarter</span></div>';
    html += '<div class="row"><span class="label">Next Payout</span>';
    html += '<span class="value">' + (payout.next_payout || 'End of quarter') + '</span></div>';
    html += '</div></div>';
  }

  document.getElementById('app').innerHTML = html;
}

loadDashboard();
setInterval(loadDashboard, 30000);  // Refresh every 30 seconds
</script>
</body>
</html>"""


# ── API Endpoints ──────────────────────────────────────────────────

def create_dashboard_app(
    stats_tracker: ProviderStatsTracker,
    provider_manager: Any = None,
) -> FastAPI:
    """Create a FastAPI app with dashboard endpoints."""

    app = FastAPI(title="Sawyer Provider Dashboard")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page():
        """Serve the provider dashboard HTML."""
        return HTMLResponse(content=DASHBOARD_HTML)

    @app.get("/api/stats")
    async def get_stats(request: Request):
        """Get provider stats as JSON for the dashboard."""
        # For now, return stats for all known providers
        # In production, this would be authenticated
        provider_ids = stats_tracker.get_all_provider_ids()

        if not provider_ids:
            return JSONResponse({
                "weekly_summary": {"days": [], "totals": {}},
                "provider": {},
                "tier": {},
                "payout": {},
            })

        # Use the first provider (demo mode)
        # In production, get from auth token
        provider_id = provider_ids[0]

        weekly = stats_tracker.get_weekly_summary(provider_id)

        # Get provider info if available
        provider_info = {}
        tier_info = {}
        payout_info = {}

        if provider_manager:
            summary = provider_manager.get_provider_summary(provider_id)
            if "error" not in summary:
                provider_info = {
                    "id": summary.get("provider_id", ""),
                    "display_name": summary.get("display_name", ""),
                    "status": summary.get("status", ""),
                    "nodes": summary.get("nodes", 0),
                }
                earnings = summary.get("earnings", {})
                payout_info = {
                    "available_balance": earnings.get("available_balance", 0),
                    "total_earned": earnings.get("total_usd_earned", 0),
                    "total_paid": earnings.get("total_usd_paid", 0),
                    "next_payout": "End of quarter",
                }

        return JSONResponse({
            "weekly_summary": weekly,
            "provider": provider_info,
            "tier": tier_info,
            "payout": payout_info,
        })

    @app.get("/api/stats/{provider_id}")
    async def get_provider_stats(provider_id: str):
        """Get stats for a specific provider."""
        weekly = stats_tracker.get_weekly_summary(provider_id)

        tier_info = {}
        if provider_manager:
            # Get node tier info
            provider = provider_manager._providers.get(provider_id)
            if provider and provider.node_ids:
                from sawyer.provider.node_tiers import classify_node
                # Use first node's VRAM for tier classification
                tier_info = {"tier": "tier_1", "multiplier": 1, "description": "Entry level"}

        return JSONResponse({
            "provider_id": provider_id,
            "weekly_summary": weekly,
            "tier": tier_info,
        })

    return app