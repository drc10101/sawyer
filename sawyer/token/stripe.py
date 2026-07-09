"""Sawyer Stripe Integration — subscription management via Stripe.

Maps Sawyer subscription tiers to Stripe prices and handles:
- Creating Stripe customers
- Creating checkout sessions for subscriptions
- Managing subscription lifecycle (activate, cancel, upgrade/downgrade)
- Webhook event processing
- Token quota enforcement via Stripe metadata

Stripe prices are looked up by environment variable per tier:
  SAWYER_STRIPE_PRICE_EXPLORER  → Explorer (free trial, unlimited tokens)
  SAWYER_STRIPE_PRICE_PRO       → Pro ($15/mo, 2M tokens)
  SAWYER_STRIPE_PRICE_PIONEER   → Pioneer ($40/mo, 5M tokens)
  SAWYER_STRIPE_PRICE_ENTERPRISE → Enterprise ($100/mo, 25M tokens)
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import stripe

from sawyer.token.budget import TIER_TOKENS, SubscriptionTier

logger = logging.getLogger(__name__)


@dataclass
class SawyerSubscription:
    """A Sawyer subscription backed by a Stripe subscription."""

    user_id: str
    stripe_customer_id: str
    stripe_subscription_id: str
    tier: SubscriptionTier
    status: str  # active, past_due, canceled, etc.
    current_period_end: int  # Unix timestamp
    token_budget: int
    tokens_used: int = 0


class SawyerStripe:
    """Manages Sawyer subscriptions through Stripe.

    Requires environment variables:
      STRIPE_SECRET_KEY — Stripe secret key
      SAWYER_STRIPE_PRICE_EXPLORER — Stripe price ID for Explorer tier
      SAWYER_STRIPE_PRICE_PRO — Stripe price ID for Pro tier
      SAWYER_STRIPE_PRICE_PIONEER — Stripe price ID for Pioneer tier
      SAWYER_STRIPE_PRICE_ENTERPRISE — Stripe price ID for Enterprise tier
    """

    # Map Sawyer tiers to environment variable names for Stripe price IDs
    TIER_PRICE_ENV = {
        SubscriptionTier.EXPLORER: "SAWYER_STRIPE_PRICE_EXPLORER",
        SubscriptionTier.PRO: "SAWYER_STRIPE_PRICE_PRO",
        SubscriptionTier.PIONEER: "SAWYER_STRIPE_PRICE_PIONEER",
        SubscriptionTier.ENTERPRISE: "SAWYER_STRIPE_PRICE_ENTERPRISE",
    }

    def __init__(self, stripe_secret_key: str | None = None) -> None:
        self.api_key = stripe_secret_key or os.environ.get("STRIPE_SECRET_KEY", "")
        if self.api_key:
            stripe.api_key = self.api_key
        self._price_ids: dict[SubscriptionTier, str] = {}
        self._load_price_ids()

    def _load_price_ids(self) -> None:
        """Load Stripe price IDs from environment variables."""
        for tier, env_var in self.TIER_PRICE_ENV.items():
            price_id = os.environ.get(env_var, "")
            if price_id:
                self._price_ids[tier] = price_id
                logger.debug("Loaded %s price: %s", tier.value, price_id)
            else:
                logger.warning(
                    "No Stripe price ID for %s tier (set %s)",
                    tier.value,
                    env_var,
                )

    def get_price_id(self, tier: SubscriptionTier) -> str | None:
        """Get the Stripe price ID for a Sawyer tier."""
        return self._price_ids.get(tier)

    def create_customer(
        self,
        email: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a Stripe customer for a Sawyer user.

        Args:
            email: Customer email
            name: Customer display name
            user_id: Sawyer user ID (stored in metadata)
            metadata: Additional metadata

        Returns:
            Stripe customer object
        """
        customer_metadata = metadata or {}
        if user_id:
            customer_metadata["sawyer_user_id"] = user_id

        customer = stripe.Customer.create(
            email=email,
            name=name or "",
            metadata=customer_metadata,
        )
        logger.info("Created Stripe customer %s for %s", customer.id, email)
        return customer

    def create_checkout_session(
        self,
        stripe_customer_id: str,
        tier: SubscriptionTier,
        success_url: str,
        cancel_url: str,
    ) -> dict[str, Any]:
        """Create a Stripe Checkout session for a Sawyer subscription.

        Args:
            stripe_customer_id: Stripe customer ID
            tier: Sawyer subscription tier
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel

        Returns:
            Stripe Checkout session object
        """
        price_id = self.get_price_id(tier)
        if not price_id:
            raise ValueError(f"No Stripe price ID configured for {tier.value} tier")

        session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "sawyer_tier": tier.value,
                "sawyer_token_budget": str(TIER_TOKENS[tier]),
            },
            subscription_data={
                "metadata": {
                    "sawyer_tier": tier.value,
                    "sawyer_token_budget": str(TIER_TOKENS[tier]),
                },
            },
        )
        logger.info(
            "Created checkout session %s for %s tier",
            session.id,
            tier.value,
        )
        return session

    def create_portal_session(
        self,
        stripe_customer_id: str,
        return_url: str,
    ) -> dict[str, Any]:
        """Create a Stripe Customer Portal session for subscription management.

        Args:
            stripe_customer_id: Stripe customer ID
            return_url: URL to redirect after portal session

        Returns:
            Stripe billing portal session object
        """
        session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=return_url,
        )
        return session

    def get_subscription(self, stripe_subscription_id: str) -> SawyerSubscription | None:
        """Get a Sawyer subscription from a Stripe subscription ID.

        Args:
            stripe_subscription_id: Stripe subscription ID

        Returns:
            SawyerSubscription or None if not found
        """
        try:
            sub = stripe.Subscription.retrieve(stripe_subscription_id)
        except stripe.error.InvalidRequestError:
            logger.warning("Subscription %s not found", stripe_subscription_id)
            return None

        tier_str = sub.metadata.get("sawyer_tier", "explorer")
        try:
            tier = SubscriptionTier(tier_str)
        except ValueError:
            tier = SubscriptionTier.EXPLORER

        customer_id = sub.customer
        user_id = sub.metadata.get("sawyer_user_id", "")

        return SawyerSubscription(
            user_id=user_id,
            stripe_customer_id=customer_id,
            stripe_subscription_id=sub.id,
            tier=tier,
            status=sub.status,
            current_period_end=sub.current_period_end,
            token_budget=TIER_TOKENS[tier],
        )

    def cancel_subscription(
        self, stripe_subscription_id: str, immediately: bool = False
    ) -> SawyerSubscription | None:
        """Cancel a Sawyer subscription.

        Args:
            stripe_subscription_id: Stripe subscription ID
            immediately: If True, cancel immediately; otherwise at period end

        Returns:
            Updated SawyerSubscription or None
        """
        try:
            if immediately:
                sub = stripe.Subscription.delete(stripe_subscription_id)
            else:
                sub = stripe.Subscription.modify(
                    stripe_subscription_id,
                    cancel_at_period_end=True,
                )
        except stripe.error.InvalidRequestError:
            logger.warning("Cannot cancel subscription %s", stripe_subscription_id)
            return None

        return self.get_subscription(sub.id)

    def update_subscription_tier(
        self,
        stripe_subscription_id: str,
        new_tier: SubscriptionTier,
    ) -> SawyerSubscription | None:
        """Change a subscription's tier (upgrade or downgrade).

        Args:
            stripe_subscription_id: Stripe subscription ID
            new_tier: New Sawyer tier

        Returns:
            Updated SawyerSubscription or None
        """
        new_price_id = self.get_price_id(new_tier)
        if not new_price_id:
            raise ValueError(f"No Stripe price ID for {new_tier.value} tier")

        try:
            sub = stripe.Subscription.retrieve(stripe_subscription_id)
        except stripe.error.InvalidRequestError:
            return None

        # Update the existing subscription item to the new price
        items = [
            {
                "id": sub["items"]["data"][0].id,
                "price": new_price_id,
            }
        ]

        updated = stripe.Subscription.modify(
            stripe_subscription_id,
            items=items,
            metadata={
                "sawyer_tier": new_tier.value,
                "sawyer_token_budget": str(TIER_TOKENS[new_tier]),
            },
        )
        logger.info(
            "Updated subscription %s to %s tier",
            stripe_subscription_id,
            new_tier.value,
        )
        return self.get_subscription(updated.id)

    def process_webhook(
        self,
        payload: str | bytes,
        sig_header: str,
        webhook_secret: str,
    ) -> dict[str, Any]:
        """Process a Stripe webhook event.

        Args:
            payload: Raw request body
            sig_header: Stripe-Signature header
            webhook_secret: Stripe webhook signing secret

        Returns:
            Parsed event dict
        """
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)

        event_type = event["type"]
        logger.info("Processing Stripe webhook: %s", event_type)

        if event_type == "checkout.session.completed":
            self._handle_checkout_complete(event)
        elif event_type == "customer.subscription.updated":
            self._handle_subscription_updated(event)
        elif event_type == "customer.subscription.deleted":
            self._handle_subscription_deleted(event)
        elif event_type == "invoice.payment_failed":
            self._handle_payment_failed(event)
        else:
            logger.debug("Unhandled webhook type: %s", event_type)

        return event

    def _handle_checkout_complete(self, event: dict) -> None:
        """Handle checkout.session.completed webhook."""
        session = event["data"]["object"]
        tier_str = session.get("metadata", {}).get("sawyer_tier", "explorer")
        logger.info(
            "Checkout completed: customer=%s tier=%s",
            session.get("customer"),
            tier_str,
        )

    def _handle_subscription_updated(self, event: dict) -> None:
        """Handle customer.subscription.updated webhook."""
        sub = event["data"]["object"]
        tier_str = sub.get("metadata", {}).get("sawyer_tier", "unknown")
        logger.info(
            "Subscription updated: %s tier=%s status=%s",
            sub.get("id"),
            tier_str,
            sub.get("status"),
        )

    def _handle_subscription_deleted(self, event: dict) -> None:
        """Handle customer.subscription.deleted webhook."""
        sub = event["data"]["object"]
        logger.info("Subscription deleted: %s", sub.get("id"))

    def _handle_payment_failed(self, event: dict) -> None:
        """Handle invoice.payment_failed webhook."""
        invoice = event["data"]["object"]
        logger.warning(
            "Payment failed: customer=%s invoice=%s",
            invoice.get("customer"),
            invoice.get("id"),
        )
