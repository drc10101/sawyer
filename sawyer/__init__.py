"""
Sawyer — Distributed MoE Inference Network.

The load is split. Friends help.

Sawyer distributes Mixture-of-Experts model inference across a network of
volunteer-hosted nodes. Each node hosts one or more expert weight files, and
a central router activates only the relevant experts per token. Users pay a
low monthly subscription ($15/mo) for a token budget — cheap enough to
experiment, paid enough to sustain the network. Hosts earn credits
proportional to compute contributed.

Trust is provided by Bedrock: cryptographic node identity, consent-gated
routing, and a tamper-evident audit chain.

SPDX-License-Identifier: BSL-1.1 — See LICENSE for details.
"""

__version__ = "0.6.0"
__author__ = "InFill Systems, LLC"

from sawyer.server import SawyerServer

__all__ = ["SawyerServer"]
