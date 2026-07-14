#!/usr/bin/env python3
"""Stripe Product & Price Setup for Sawyer Inference Network.

Creates the Sawyer subscription products and prices in Stripe and outputs
the environment variables needed by the Sawyer API server.

Prerequisites:
  - Stripe CLI installed and authenticated: https://docs.stripe.com/stripe-cli
  - Or set STRIPE_SECRET_KEY env var with a Stripe secret key

Usage:
  python scripts/stripe_setup.py               # Create products & prices
  python scripts/stripe_setup.py --dry-run       # Preview without creating
  python scripts/stripe_setup.py --show-env      # Output env vars only

Products created:
  - Sawyer Explorer (Free, 14-day trial)
  - Sawyer Pro ($15/mo, 2M tokens)
  - Sawyer Pioneer ($40/mo, 5M tokens)
  - Sawyer Enterprise ($200/mo, 10M tokens)

After running, add the output env vars to your .env or deployment config.
"""

import argparse
import json
import os
import sys

try:
    import stripe
except ImportError:
    print("ERROR: stripe package required. Install with: pip install stripe")
    sys.exit(1)

# ── Tier Definitions ──────────────────────────────────────────────

TIERS = [
    {
        "name": "Sawyer Explorer",
        "description": "14-day free trial with unlimited tokens. No credit card required.",
        "price": 0,
        "interval": "month",
        "metadata": {
            "sawyer_tier": "explorer",
            "sawyer_token_budget": "0",  # Unlimited during trial
            "sawyer_trial_days": "14",
        },
        "env_var": "SAWYER_STRIPE_PRICE_EXPLORER",
    },
    {
        "name": "Sawyer Pro",
        "description": "Production inference — 2,000,000 tokens/month with priority routing.",
        "price": 1500,  # $15.00 in cents
        "interval": "month",
        "metadata": {
            "sawyer_tier": "pro",
            "sawyer_token_budget": "2000000",
        },
        "env_var": "SAWYER_STRIPE_PRICE_PRO",
    },
    {
        "name": "Sawyer Pioneer",
        "description": "Scale inference — 5,000,000 tokens/month with adaptive routing.",
        "price": 4000,  # $40.00 in cents
        "interval": "month",
        "metadata": {
            "sawyer_tier": "pioneer",
            "sawyer_token_budget": "5000000",
        },
        "env_var": "SAWYER_STRIPE_PRICE_PIONEER",
    },
    {
        "name": "Sawyer Enterprise",
        "description": "Unlimited inference — 10,000,000 tokens/month with dedicated routing and custom SLA.",
        "price": 20000,  # $200.00 in cents
        "interval": "month",
        "metadata": {
            "sawyer_tier": "enterprise",
            "sawyer_token_budget": "10000000",
        },
        "env_var": "SAWYER_STRIPE_PRICE_ENTERPRISE",
    },
]


def setup_stripe(dry_run: bool = False) -> dict[str, str]:
    """Create Stripe products and prices, return env var mapping."""
    api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    if not api_key:
        print("ERROR: STRIPE_SECRET_KEY environment variable not set.")
        print("Get your key from https://dashboard.stripe.com/apikeys")
        sys.exit(1)

    stripe.api_key = api_key
    env_vars = {}

    for tier in TIERS:
        print(f"\n{'[DRY-RUN] ' if dry_run else ''}Creating: {tier['name']}")

        if dry_run:
            print(f"  Price: ${tier['price'] / 100:.2f}/{tier['interval']}")
            print(f"  Env var: {tier['env_var']}=price_DRY_RUN_{tier['metadata']['sawyer_tier']}")
            env_vars[tier["env_var"]] = f"price_DRY_RUN_{tier['metadata']['sawyer_tier']}"
            continue

        # Create product
        product = stripe.Product.create(
            name=tier["name"],
            description=tier["description"],
            metadata=tier["metadata"],
        )
        print(f"  Product: {product.id}")

        # Create price
        if tier["price"] == 0:
            # Free tier -- $0 recurring price
            price = stripe.Price.create(
                product=product.id,
                unit_amount=0,
                currency="usd",
                recurring={"interval": tier["interval"]},
                metadata=tier["metadata"],
            )
        else:
            price = stripe.Price.create(
                product=product.id,
                unit_amount=tier["price"],
                currency="usd",
                recurring={"interval": tier["interval"]},
                metadata=tier["metadata"],
            )
        print(f"  Price: {price.id} (${tier['price'] / 100:.2f}/{tier['interval']})")

        env_vars[tier["env_var"]] = price.id

    return env_vars


def show_env_vars(env_vars: dict[str, str]) -> None:
    """Print env vars in shell-compatible format."""
    print("\n# Add these to your .env or deployment config:")
    print("# Stripe Price IDs for Sawyer tiers")
    for var, val in env_vars.items():
        print(f"export {var}={val}")


def check_existing() -> None:
    """Check if price env vars are already set."""
    found = []
    for tier in TIERS:
        val = os.environ.get(tier["env_var"], "")
        if val:
            found.append(f"  {tier['env_var']}={val}")

    if found:
        print("Existing Stripe price IDs found in environment:")
        for line in found:
            print(line)
        print("\nRe-running will create NEW products/prices. Delete old ones in Stripe Dashboard.")
    else:
        print("No existing Stripe price IDs found in environment.")


def main():
    parser = argparse.ArgumentParser(description="Set up Stripe products and prices for Sawyer")
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating")
    parser.add_argument("--show-env", action="store_true", help="Show current env vars only")
    parser.add_argument("--check", action="store_true", help="Check existing env vars")
    args = parser.parse_args()

    if args.check:
        check_existing()
        return

    if args.show_env:
        # Show existing env vars
        found = False
        for tier in TIERS:
            val = os.environ.get(tier["env_var"], "")
            if val:
                print(f"{tier['env_var']}={val}")
                found = True
        if not found:
            print("No Stripe price IDs found in environment. Run setup first.")
        return

    print("Sawyer Stripe Product & Price Setup")
    print("=" * 40)

    env_vars = setup_stripe(dry_run=args.dry_run)
    show_env_vars(env_vars)

    if not args.dry_run:
        print("\nSetup complete. Configure your Stripe webhook endpoint at:")
        print("  https://api.sawyer.infill.systems/api/stripe/webhook")
        print("\nEvents to subscribe:")
        print("  - checkout.session.completed")
        print("  - customer.subscription.updated")
        print("  - customer.subscription.deleted")
        print("  - invoice.payment_failed")


if __name__ == "__main__":
    main()