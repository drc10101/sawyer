"""Tests for provider dashboard and stats tracking."""

import pytest

from sawyer.provider.dashboard import DailyStats, ProviderStatsTracker


class TestDailyStats:
    """Test DailyStats dataclass."""

    def test_default_values(self):
        """DailyStats defaults to zeros."""
        stats = DailyStats(date="2026-06-30")
        assert stats.date == "2026-06-30"
        assert stats.tokens_served == 0
        assert stats.requests_count == 0
        assert stats.earnings_usd == 0.0


class TestProviderStatsTracker:
    """Test stats tracking and aggregation."""

    def test_record_tokens(self):
        """Record token serving events."""
        tracker = ProviderStatsTracker()
        tracker.record_tokens("prov_1", tokens=1000, model="mixtral-8x7b")
        tracker.record_tokens("prov_1", tokens=500, model="mixtral-8x7b")

        daily = tracker.get_daily_stats("prov_1", days=1)
        assert len(daily) == 1
        assert daily[0]["tokens_served"] == 1500
        assert daily[0]["requests_count"] == 2

    def test_record_tokens_with_expert(self):
        """Track which experts are serving tokens."""
        tracker = ProviderStatsTracker()
        tracker.record_tokens("prov_1", tokens=100, expert_id="expert-0")
        tracker.record_tokens("prov_1", tokens=200, expert_id="expert-3")
        tracker.record_tokens("prov_1", tokens=50, expert_id="expert-0")

        daily = tracker.get_daily_stats("prov_1", days=1)
        assert daily[0]["expert_breakdown"]["expert-0"] == 150
        assert daily[0]["expert_breakdown"]["expert-3"] == 200

    def test_record_tokens_with_model_breakdown(self):
        """Track tokens per model."""
        tracker = ProviderStatsTracker()
        tracker.record_tokens("prov_1", tokens=500, model="mixtral-8x7b")
        tracker.record_tokens("prov_1", tokens=300, model="deepseek-v2-lite")
        tracker.record_tokens("prov_1", tokens=200, model="mixtral-8x7b")

        daily = tracker.get_daily_stats("prov_1", days=1)
        assert daily[0]["model_breakdown"]["mixtral-8x7b"] == 700
        assert daily[0]["model_breakdown"]["deepseek-v2-lite"] == 300

    def test_record_errors(self):
        """Track error counts."""
        tracker = ProviderStatsTracker()
        tracker.record_tokens("prov_1", tokens=100, is_error=True)
        tracker.record_tokens("prov_1", tokens=200, is_error=False)
        tracker.record_tokens("prov_1", tokens=50, is_error=True)

        daily = tracker.get_daily_stats("prov_1", days=1)
        assert daily[0]["errors_count"] == 2
        assert daily[0]["tokens_served"] == 350

    def test_record_latency(self):
        """Track average latency."""
        tracker = ProviderStatsTracker()
        tracker.record_tokens("prov_1", tokens=100, latency_ms=50.0)
        tracker.record_tokens("prov_1", tokens=200, latency_ms=100.0)

        daily = tracker.get_daily_stats("prov_1", days=1)
        assert daily[0]["avg_latency_ms"] > 0

    def test_record_uptime(self):
        """Track uptime hours."""
        tracker = ProviderStatsTracker()
        tracker.record_uptime("prov_1", hours=24.0)

        daily = tracker.get_daily_stats("prov_1", days=1)
        assert daily[0]["uptime_hours"] == 24.0

    def test_record_earnings(self):
        """Track daily earnings."""
        tracker = ProviderStatsTracker()
        tracker.record_earnings("prov_1", usd=12.50)

        daily = tracker.get_daily_stats("prov_1", days=1)
        assert daily[0]["earnings_usd"] == 12.50

    def test_multiple_providers(self):
        """Track stats for multiple providers independently."""
        tracker = ProviderStatsTracker()
        tracker.record_tokens("prov_1", tokens=1000)
        tracker.record_tokens("prov_2", tokens=500)

        prov1 = tracker.get_daily_stats("prov_1", days=1)
        prov2 = tracker.get_daily_stats("prov_2", days=1)

        assert prov1[0]["tokens_served"] == 1000
        assert prov2[0]["tokens_served"] == 500

    def test_get_all_provider_ids(self):
        """List all providers with stats."""
        tracker = ProviderStatsTracker()
        tracker.record_tokens("prov_1", tokens=100)
        tracker.record_tokens("prov_2", tokens=200)
        tracker.record_tokens("prov_3", tokens=300)

        ids = tracker.get_all_provider_ids()
        assert len(ids) == 3
        assert "prov_1" in ids
        assert "prov_2" in ids
        assert "prov_3" in ids


class TestWeeklySummary:
    """Test weekly aggregation."""

    def test_weekly_summary_aggregates_daily(self):
        """Weekly summary totals all daily stats."""
        tracker = ProviderStatsTracker()

        # Simulate a week of activity
        for _ in range(7):
            tracker.record_tokens("prov_1", tokens=1000)
            tracker.record_earnings("prov_1", usd=1.50)
            tracker.record_uptime("prov_1", hours=24.0)

        summary = tracker.get_weekly_summary("prov_1")

        assert summary["totals"]["tokens_served"] == 7000
        assert summary["totals"]["earnings_usd"] == pytest.approx(10.50, abs=0.01)
        assert summary["totals"]["uptime_hours"] == 168.0
        assert summary["totals"]["requests_count"] == 7
        assert len(summary["days"]) == 7

    def test_weekly_summary_model_breakdown(self):
        """Weekly summary aggregates model breakdown across all days."""
        tracker = ProviderStatsTracker()

        tracker.record_tokens("prov_1", tokens=500, model="mixtral-8x7b")
        tracker.record_tokens("prov_1", tokens=300, model="deepseek-v2-lite")

        summary = tracker.get_weekly_summary("prov_1")

        assert summary["totals"]["model_breakdown"]["mixtral-8x7b"] == 500
        assert summary["totals"]["model_breakdown"]["deepseek-v2-lite"] == 300

    def test_weekly_avg_daily_tokens(self):
        """Average daily tokens calculated correctly."""
        tracker = ProviderStatsTracker()

        for _ in range(7):
            tracker.record_tokens("prov_1", tokens=1000)

        summary = tracker.get_weekly_summary("prov_1")
        assert summary["totals"]["avg_daily_tokens"] == 1000

    def test_empty_week(self):
        """Weekly summary with no data returns zeros."""
        tracker = ProviderStatsTracker()

        summary = tracker.get_weekly_summary("unknown_provider")

        assert summary["totals"]["tokens_served"] == 0
        assert summary["totals"]["earnings_usd"] == 0
        assert len(summary["days"]) == 7  # Still returns 7 days of zeros


class TestDashboardApp:
    """Test the FastAPI dashboard endpoints."""

    def test_dashboard_html_page(self):
        """GET / returns the dashboard HTML."""
        from sawyer.provider.dashboard import create_dashboard_app

        tracker = ProviderStatsTracker()
        app = create_dashboard_app(stats_tracker=tracker)

        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        assert "Sawyer" in response.text
        assert "Provider Dashboard" in response.text

    def test_api_stats_empty(self):
        """GET /api/stats returns empty data when no providers."""
        from sawyer.provider.dashboard import create_dashboard_app

        tracker = ProviderStatsTracker()
        app = create_dashboard_app(stats_tracker=tracker)

        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "weekly_summary" in data
        assert data["weekly_summary"]["days"] == []

    def test_api_stats_with_data(self):
        """GET /api/stats returns provider stats."""
        from sawyer.provider.dashboard import create_dashboard_app

        tracker = ProviderStatsTracker()
        tracker.record_tokens("prov_1", tokens=5000, model="mixtral-8x7b")
        tracker.record_earnings("prov_1", usd=3.50)

        app = create_dashboard_app(stats_tracker=tracker)

        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["weekly_summary"]["totals"]["tokens_served"] == 5000
        assert data["weekly_summary"]["totals"]["earnings_usd"] == pytest.approx(3.50, abs=0.01)

    def test_api_stats_by_provider(self):
        """GET /api/stats/{provider_id} returns per-provider stats."""
        from sawyer.provider.dashboard import create_dashboard_app

        tracker = ProviderStatsTracker()
        tracker.record_tokens("prov_1", tokens=1000)
        tracker.record_tokens("prov_2", tokens=5000)

        app = create_dashboard_app(stats_tracker=tracker)

        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.get("/api/stats/prov_2")
        assert response.status_code == 200
        data = response.json()
        assert data["weekly_summary"]["totals"]["tokens_served"] == 5000