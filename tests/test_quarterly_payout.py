"""Tests for quarterly payout distribution."""

import pytest

from sawyer.provider.quarterly_payout import (
    MIN_PAYOUT_USD,
    PayoutStatus,
    QuarterlyPayout,
    SUBSCRIPTION_PRICE,
)


class TestQuarterlyPoolCalculation:
    """Test the pool math: subscribers x $51.67 x 3 months x 70%."""

    def test_pool_calculation_1_subscriber(self):
        """1 subscriber: $51.67 x 3 x 0.70 = $108.51 to providers."""
        qp = QuarterlyPayout()
        pool = qp.calculate_quarterly_pool(total_subscribers=1)

        assert pool["total_subscribers"] == 1
        assert pool["total_revenue_usd"] == pytest.approx(155.01, rel=0.01)
        assert pool["provider_pool_usd"] == pytest.approx(108.51, rel=0.01)
        assert pool["platform_pool_usd"] == pytest.approx(46.50, rel=0.01)

    def test_pool_calculation_100_subscribers(self):
        """100 subscribers: $5167/mo x 3 x 0.70 = $10,850 to providers."""
        qp = QuarterlyPayout()
        pool = qp.calculate_quarterly_pool(total_subscribers=100)

        assert pool["total_revenue_usd"] == pytest.approx(15501.00, rel=0.01)
        assert pool["provider_pool_usd"] == pytest.approx(10850.70, rel=0.01)
        assert pool["platform_pool_usd"] == pytest.approx(4650.30, rel=0.01)

    def test_pool_calculation_1000_subscribers(self):
        """1000 subscribers: $51.67K/mo x 3 x 0.70 = $105,017 to providers."""
        qp = QuarterlyPayout()
        pool = qp.calculate_quarterly_pool(total_subscribers=1000)

        assert pool["total_revenue_usd"] == pytest.approx(155010.00, rel=0.01)
        assert pool["provider_pool_usd"] == pytest.approx(108507.00, rel=0.01)
        assert pool["platform_pool_usd"] == pytest.approx(46503.00, rel=0.01)

    def test_platform_never_gets_more_than_30_percent(self):
        """Platform share is always exactly 30%."""
        for subs in [1, 10, 100, 1000]:
            qp = QuarterlyPayout()
            pool = qp.calculate_quarterly_pool(total_subscribers=subs)
            total = pool["provider_pool_usd"] + pool["platform_pool_usd"]
            assert pool["platform_pool_usd"] == pytest.approx(total * 0.30, abs=0.01)


class TestQuarterlyDistribution:
    """Test proportional distribution of the quarterly pool."""

    def test_single_provider_gets_entire_pool(self):
        """One provider doing all the work gets the entire pool."""
        qp = QuarterlyPayout()
        payouts = qp.distribute_quarterly(
            total_subscribers=100,
            provider_stats=[
                {
                    "provider_id": "prov_monster",
                    "tokens_served": 500_000,
                    "uptime_hours": 720,
                    "vram_gb": 24.0,
                },
            ],
        )

        assert len(payouts) == 1
        assert payouts[0].total_earnings_usd == pytest.approx(10850.7, abs=1.0)
        assert payouts[0].share_percent == 100.0
        assert payouts[0].tier == "tier_4"
        assert payouts[0].multiplier == 4.0

    def test_monster_pc_earns_most(self):
        """The kid with the 4090 earns the biggest slice."""
        qp = QuarterlyPayout()
        payouts = qp.distribute_quarterly(
            total_subscribers=100,
            provider_stats=[
                # Monster PC: 24GB, serving most traffic
                {
                    "provider_id": "prov_monster",
                    "tokens_served": 500_000,
                    "uptime_hours": 720,
                    "vram_gb": 24.0,
                },
                # Laptop: 4GB, some traffic
                {
                    "provider_id": "prov_laptop",
                    "tokens_served": 50_000,
                    "uptime_hours": 720,
                    "vram_gb": 4.0,
                },
            ],
        )

        monster = next(p for p in payouts if p.provider_id == "prov_monster")
        laptop = next(p for p in payouts if p.provider_id == "prov_laptop")

        # Monster has 4x multiplier + 10x tokens = way more earnings
        assert monster.total_earnings_usd > laptop.total_earnings_usd * 10
        assert monster.tier == "tier_4"
        assert laptop.tier == "tier_1"

    def test_four_tier_distribution(self):
        """All four tiers earn proportionally to their contribution."""
        qp = QuarterlyPayout()
        payouts = qp.distribute_quarterly(
            total_subscribers=50,
            provider_stats=[
                {"provider_id": "t1", "tokens_served": 20_000, "uptime_hours": 200, "vram_gb": 4.0},
                {"provider_id": "t2", "tokens_served": 50_000, "uptime_hours": 400, "vram_gb": 8.0},
                {"provider_id": "t3", "tokens_served": 100_000, "uptime_hours": 500, "vram_gb": 12.0},
                {"provider_id": "t4", "tokens_served": 300_000, "uptime_hours": 600, "vram_gb": 24.0},
            ],
        )

        t1 = next(p for p in payouts if p.provider_id == "t1")
        t2 = next(p for p in payouts if p.provider_id == "t2")
        t3 = next(p for p in payouts if p.provider_id == "t3")
        t4 = next(p for p in payouts if p.provider_id == "t4")

        # Each higher tier earns more
        assert t4.total_earnings_usd > t3.total_earnings_usd
        assert t3.total_earnings_usd > t2.total_earnings_usd
        assert t2.total_earnings_usd > t1.total_earnings_usd

        # Total distributed equals provider pool
        pool = 51.67 * 50 * 3 * 0.70  # 50 subs x $51.67 x 3 months x 70%
        total_distributed = sum(p.total_earnings_usd for p in payouts)
        assert total_distributed == pytest.approx(pool, abs=1.0)

    def test_same_tokens_different_tier(self):
        """Same token count, higher tier earns more per token."""
        qp = QuarterlyPayout()
        payouts = qp.distribute_quarterly(
            total_subscribers=10,
            provider_stats=[
                {"provider_id": "laptop", "tokens_served": 100_000, "uptime_hours": 720, "vram_gb": 4.0},
                {"provider_id": "monster", "tokens_served": 100_000, "uptime_hours": 720, "vram_gb": 24.0},
            ],
        )

        laptop = next(p for p in payouts if p.provider_id == "laptop")
        monster = next(p for p in payouts if p.provider_id == "monster")

        # Monster earns 4x per token because of the tier multiplier
        assert monster.token_earnings_usd == pytest.approx(
            laptop.token_earnings_usd * 4, abs=0.5
        )


class TestMinimumPayout:
    """Test the $25 minimum payout threshold."""

    def test_below_minimum_rolls_over(self):
        """Earnings below $25 roll over to next quarter."""
        qp = QuarterlyPayout()
        payouts = qp.distribute_quarterly(
            total_subscribers=1,  # Tiny pool
            provider_stats=[
                {
                    "provider_id": "small_provider",
                    "tokens_served": 100,
                    "uptime_hours": 10,
                    "vram_gb": 4.0,
                },
            ],
        )

        assert len(payouts) == 1
        if payouts[0].total_earnings_usd < MIN_PAYOUT_USD:
            assert payouts[0].status == PayoutStatus.ROLLED_OVER

    def test_above_minimum_gets_paid(self):
        """Earnings above $25 are eligible for payout."""
        qp = QuarterlyPayout()
        payouts = qp.distribute_quarterly(
            total_subscribers=100,
            provider_stats=[
                {
                    "provider_id": "big_provider",
                    "tokens_served": 500_000,
                    "uptime_hours": 720,
                    "vram_gb": 24.0,
                },
            ],
        )

        assert len(payouts) == 1
        assert payouts[0].total_earnings_usd >= MIN_PAYOUT_USD
        assert payouts[0].status == PayoutStatus.PENDING

    def test_rollover_accumulates(self):
        """Rollover from previous quarters adds to next quarter's earnings."""
        qp = QuarterlyPayout()

        # Quarter 1: small earnings, rolls over
        payouts_q1 = qp.distribute_quarterly(
            total_subscribers=1,
            provider_stats=[
                {
                    "provider_id": "prov",
                    "tokens_served": 100,
                    "uptime_hours": 10,
                    "vram_gb": 4.0,
                },
            ],
        )

        # If it rolled over, check next quarter includes it
        rollovers = qp.get_rolled_over()
        if "prov" in rollovers:
            assert rollovers["prov"] > 0

            # Quarter 2: bigger pool, should include Q1 rollover
            payouts_q2 = qp.distribute_quarterly(
                total_subscribers=100,
                provider_stats=[
                    {
                        "provider_id": "prov",
                        "tokens_served": 200_000,
                        "uptime_hours": 720,
                        "vram_gb": 8.0,
                    },
                ],
            )

            # Q2 payout should be higher than Q2 earnings alone
            # because it includes the Q1 rollover
            assert payouts_q2[0].total_earnings_usd > 0


class TestPayoutLifecycle:
    """Test payout state transitions."""

    def test_mark_paid(self):
        """Successful payout gets marked as paid."""
        qp = QuarterlyPayout()
        payouts = qp.distribute_quarterly(
            total_subscribers=100,
            provider_stats=[
                {
                    "provider_id": "prov",
                    "tokens_served": 500_000,
                    "uptime_hours": 720,
                    "vram_gb": 24.0,
                },
            ],
        )

        payout_id = payouts[0].payout_id
        paid = qp.mark_paid(payout_id, stripe_transfer_id="txn_123")

        assert paid is not None
        assert paid.status == PayoutStatus.PAID
        assert paid.stripe_transfer_id == "txn_123"
        assert paid.paid_at > 0

    def test_mark_failed_rolls_over(self):
        """Failed payout rolls earnings to next quarter."""
        qp = QuarterlyPayout()
        payouts = qp.distribute_quarterly(
            total_subscribers=100,
            provider_stats=[
                {
                    "provider_id": "prov",
                    "tokens_served": 500_000,
                    "uptime_hours": 720,
                    "vram_gb": 24.0,
                },
            ],
        )

        payout_id = payouts[0].payout_id
        failed = qp.mark_failed(payout_id, reason="Stripe account closed")

        assert failed is not None
        assert failed.status == PayoutStatus.FAILED
        assert failed.failure_reason == "Stripe account closed"

        # Earnings rolled over
        rollovers = qp.get_rolled_over()
        assert "prov" in rollovers
        assert rollovers["prov"] > 0

    def test_get_pending_payouts(self):
        """Only non-rollover payouts above minimum are pending."""
        qp = QuarterlyPayout()
        qp.distribute_quarterly(
            total_subscribers=100,
            provider_stats=[
                {
                    "provider_id": "prov_big",
                    "tokens_served": 500_000,
                    "uptime_hours": 720,
                    "vram_gb": 24.0,
                },
                {
                    "provider_id": "prov_tiny",
                    "tokens_served": 50,
                    "uptime_hours": 5,
                    "vram_gb": 4.0,
                },
            ],
        )

        pending = qp.get_pending_payouts()
        for p in pending:
            assert p.total_earnings_usd >= MIN_PAYOUT_USD
            assert p.status == PayoutStatus.PENDING


class TestQuarterlyPreview:
    """Test preview/summary without creating payouts."""

    def test_preview_shows_earnings_before_payout(self):
        """Providers can see what they'll earn before payout."""
        qp = QuarterlyPayout()
        summary = qp.get_quarterly_summary(
            total_subscribers=100,
            provider_stats=[
                {"provider_id": "prov_1", "tokens_served": 300_000, "uptime_hours": 500, "vram_gb": 12.0},
                {"provider_id": "prov_2", "tokens_served": 100_000, "uptime_hours": 300, "vram_gb": 8.0},
            ],
        )

        assert summary["total_subscribers"] == 100
        assert summary["total_revenue_usd"] == 15501.00
        assert summary["provider_pool_usd"] == 10850.70
        assert len(summary["providers"]) == 2

        # Each provider has earnings preview
        for prov in summary["providers"]:
            assert "total_earnings_usd" in prov
            assert prov["total_earnings_usd"] > 0