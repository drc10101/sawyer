"""Tests for Sawyer Stripe integration — subscription management."""

import os
from unittest.mock import MagicMock, patch

import pytest

from sawyer.token.budget import TIER_PRICING, TIER_TOKENS, SubscriptionTier
from sawyer.token.stripe import SawyerStripe, SawyerSubscription


class TestSawyerStripeInit:
    """Test SawyerStripe initialization and price ID loading."""

    def test_init_no_api_key(self):
        """SawyerStripe initializes without a Stripe key (dev mode)."""
        with patch.dict(os.environ, {}, clear=True):
            ss = SawyerStripe(stripe_secret_key="")
            assert ss.api_key == ""

    def test_init_with_api_key(self):
        """SawyerStripe sets the Stripe API key."""
        ss = SawyerStripe(stripe_secret_key="sk_test_123")
        assert ss.api_key == "sk_test_123"

    def test_load_price_ids(self):
        """Price IDs are loaded from environment variables."""
        env = {
            "SAWYER_STRIPE_PRICE_EXPLORER": "price_explorer",
            "SAWYER_STRIPE_PRICE_PRO": "price_pro",
            "SAWYER_STRIPE_PRICE_ENTERPRISE": "price_enterprise",
        }
        with patch.dict(os.environ, env, clear=True):
            ss = SawyerStripe(stripe_secret_key="sk_test")
            assert ss.get_price_id(SubscriptionTier.EXPLORER) == "price_explorer"
            assert ss.get_price_id(SubscriptionTier.PRO) == "price_pro"
            assert ss.get_price_id(SubscriptionTier.ENTERPRISE) == "price_enterprise"

    def test_missing_price_ids(self):
        """Missing price IDs return None."""
        with patch.dict(os.environ, {}, clear=True):
            ss = SawyerStripe(stripe_secret_key="sk_test")
            assert ss.get_price_id(SubscriptionTier.EXPLORER) is None


class TestSawyerStripeCustomer:
    """Test customer creation."""

    @patch("stripe.Customer.create")
    def test_create_customer(self, mock_create):
        """Create a Stripe customer with Sawyer metadata."""
        mock_create.return_value = MagicMock(id="cus_123", email="test@example.com")

        ss = SawyerStripe(stripe_secret_key="sk_test")
        ss.create_customer(
            email="test@example.com",
            name="Test User",
            user_id="sawyer-user-1",
        )
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["email"] == "test@example.com"
        assert call_kwargs["metadata"]["sawyer_user_id"] == "sawyer-user-1"


class TestSawyerStripeCheckout:
    """Test checkout session creation."""

    @patch("stripe.checkout.Session.create")
    def test_create_checkout_session(self, mock_create):
        """Create a checkout session for Explorer tier."""
        mock_create.return_value = MagicMock(id="cs_123", url="https://checkout.stripe.com/123")

        env = {"SAWYER_STRIPE_PRICE_EXPLORER": "price_abc123"}
        with patch.dict(os.environ, env, clear=True):
            ss = SawyerStripe(stripe_secret_key="sk_test")
            ss.create_checkout_session(
                stripe_customer_id="cus_123",
                tier=SubscriptionTier.EXPLORER,
                success_url="https://sawyer.dev/success",
                cancel_url="https://sawyer.dev/cancel",
            )
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["mode"] == "subscription"
            assert call_kwargs["line_items"][0]["price"] == "price_abc123"

    def test_create_checkout_no_price(self):
        """Checkout raises ValueError when price ID is missing."""
        with patch.dict(os.environ, {}, clear=True):
            ss = SawyerStripe(stripe_secret_key="sk_test")
            with pytest.raises(ValueError, match="No Stripe price ID"):
                ss.create_checkout_session(
                    stripe_customer_id="cus_123",
                    tier=SubscriptionTier.EXPLORER,
                    success_url="https://sawyer.dev/success",
                    cancel_url="https://sawyer.dev/cancel",
                )


class TestSawyerStripeSubscription:
    """Test subscription management."""

    def test_sawyer_subscription_dataclass(self):
        """SawyerSubscription holds expected fields."""
        sub = SawyerSubscription(
            user_id="user-1",
            stripe_customer_id="cus_123",
            stripe_subscription_id="sub_456",
            tier=SubscriptionTier.PRO,
            status="active",
            current_period_end=1700000000,
            token_budget=TIER_TOKENS[SubscriptionTier.PRO],
        )
        assert sub.tier == SubscriptionTier.PRO
        assert sub.token_budget == 2_000_000
        assert sub.tokens_used == 0

    @patch("stripe.Subscription.retrieve")
    def test_get_subscription(self, mock_retrieve):
        """Get a SawyerSubscription from Stripe."""
        mock_retrieve.return_value = MagicMock(
            id="sub_789",
            customer="cus_123",
            status="active",
            current_period_end=1700000000,
            metadata={"sawyer_tier": "builder", "sawyer_user_id": "user-1"},
        )
        mock_retrieve.return_value.__getitem__ = lambda self, key: {
            "id": "sub_789",
            "customer": "cus_123",
            "status": "active",
        }[key]

        env = {"SAWYER_STRIPE_PRICE_PRO": "price_pro"}
        with patch.dict(os.environ, env, clear=True):
            ss = SawyerStripe(stripe_secret_key="sk_test")
            sub = ss.get_subscription("sub_789")
            assert sub is not None
            assert sub.tier == SubscriptionTier.PRO


class TestSawyerStripePortal:
    """Test customer portal session."""

    @patch("stripe.billing_portal.Session.create")
    def test_create_portal_session(self, mock_create):
        """Create a billing portal session."""
        mock_create.return_value = MagicMock(url="https://billing.stripe.com/portal")

        ss = SawyerStripe(stripe_secret_key="sk_test")
        ss.create_portal_session(
            stripe_customer_id="cus_123",
            return_url="https://sawyer.dev/dashboard",
        )
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["customer"] == "cus_123"
        assert call_kwargs["return_url"] == "https://sawyer.dev/dashboard"


class TestSawyerStripeTiers:
    """Verify tier configuration consistency."""

    def test_tier_pricing_matches(self):
        """SawyerStripe tier prices match Sawyer budget module."""
        assert TIER_PRICING[SubscriptionTier.EXPLORER] == 0
        assert TIER_PRICING[SubscriptionTier.PRO] == 15
        assert TIER_PRICING[SubscriptionTier.PIONEER] == 40
        assert TIER_PRICING[SubscriptionTier.ENTERPRISE] == 100

    def test_tier_tokens_match(self):
        """SawyerStripe token budgets match Sawyer budget module."""
        assert TIER_TOKENS[SubscriptionTier.EXPLORER] == 0
        assert TIER_TOKENS[SubscriptionTier.PRO] == 2_000_000
        assert TIER_TOKENS[SubscriptionTier.PIONEER] == 5_000_000
        assert TIER_TOKENS[SubscriptionTier.ENTERPRISE] == 25_000_000
