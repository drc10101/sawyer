"""Sawyer Gateway — main router server entry point.

Receives inference requests, runs the gating network to select experts,
routes to the appropriate nodes, aggregates results, and returns to the user.
"""

import logging

from sawyer.config import SawyerConfig
from sawyer.router.scheduler import ExpertScheduler

logger = logging.getLogger(__name__)


class SawyerGateway:
    """Main Sawyer router server.

    Authenticates users, checks token balances, runs gating networks
    to select experts, routes to nodes, and aggregates results.
    """

    def __init__(self, config: SawyerConfig) -> None:
        self.config = config
        self.scheduler = ExpertScheduler()
        self._running = False

    async def start(self) -> None:
        """Start the gateway server."""
        logger.info("Starting Sawyer Gateway on port %d", self.config.inference_port)
        self._running = True
        # TODO: gRPC/QUIC server setup

    async def stop(self) -> None:
        """Stop the gateway server."""
        logger.info("Stopping Sawyer Gateway")
        self._running = False

    async def handle_request(
        self,
        model: str,
        tokens: list,
        user_id: str,
        token_balance: float,
    ) -> dict:
        """Handle an inference request.

        1. Validate user token balance
        2. Run gating network to select experts
        3. Route to available nodes
        4. Aggregate expert outputs
        5. Debit token balance

        Args:
            model: Model name (e.g., "mixtral-8x7b")
            tokens: Input token embeddings
            user_id: Authenticated user ID
            token_balance: User's remaining token balance

        Returns:
            Aggregated inference result
        """
        if not self._running:
            raise RuntimeError("Gateway is not running")

        # Check token balance
        if token_balance <= 0:
            raise ValueError("Insufficient token balance")

        # Route to experts via scheduler
        result = await self.scheduler.route(model, tokens, user_id)

        logger.info("Request completed for user %s, model %s", user_id, model)
        return result
