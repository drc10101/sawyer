"""
Sawyer CLI — Register nodes, serve experts, check status.

Usage:
    sawyer register     Register this machine as a Sawyer node
    sawyer serve        Start serving expert inference requests
    sawyer status       Show network status and token balance
    sawyer models       List available models and expert layouts
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sawyer",
        description="Sawyer — Distributed MoE Inference Network",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # register
    reg_parser = subparsers.add_parser("register", help="Register this machine as a Sawyer node")
    reg_parser.add_argument("--name", default=None, help="Node name (default: hostname)")
    reg_parser.add_argument("--gpu", action="store_true", help="Auto-detect GPU capabilities")
    reg_parser.add_argument(
        "--experts", nargs="+", help="Expert IDs to host (default: auto-assign)"
    )

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start serving expert inference requests")
    serve_parser.add_argument("--port", type=int, default=8444, help="Port for inference server")
    serve_parser.add_argument(
        "--router", default="https://router.sawyer.dev", help="Router endpoint"
    )

    # status
    subparsers.add_parser("status", help="Show network status and token balance")

    # models
    subparsers.add_parser("models", help="List available models and expert layouts")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # TODO: Implement command handlers
    print(f"sawyer {args.command}: not yet implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
