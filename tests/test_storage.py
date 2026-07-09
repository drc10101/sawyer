"""Tests for Sawyer Storage — SQLite persistence."""

import tempfile
import time
from pathlib import Path

import pytest

from sawyer.storage.accountant import PersistedAccountant
from sawyer.storage.database import SawyerStorage
from sawyer.token.accounting import InsufficientTokens
from sawyer.token.budget import SubscriptionTier


class TestSawyerStorage:
    """Test SawyerStorage — SQLite CRUD operations."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_sawyer.db"
        self.storage = SawyerStorage(str(self.db_path))

    def teardown_method(self):
        self.storage.close()

    def test_database_creates_on_init(self):
        """Database file is created on initialization."""
        assert self.db_path.exists()

    def test_upsert_and_get_node(self):
        """Insert a node and retrieve it."""
        node = {
            "node_id": "node-1",
            "node_name": "test-node",
            "gpu": "rtx-4090",
            "vram_gb": 24.0,
            "bandwidth_mbps": 1000.0,
            "latency_ms": 15.0,
            "experts": [0, 2],
            "healthy": True,
        }
        self.storage.upsert_node(node)

        result = self.storage.get_node("node-1")
        assert result is not None
        assert result["node_id"] == "node-1"
        assert result["gpu"] == "rtx-4090"
        assert result["experts"] == [0, 2]

    def test_upsert_updates_existing_node(self):
        """Upserting a node with same ID updates it."""
        self.storage.upsert_node(
            {
                "node_id": "node-1",
                "gpu": "rtx-3090",
                "latency_ms": 20.0,
                "experts": [0],
            }
        )
        self.storage.upsert_node(
            {
                "node_id": "node-1",
                "gpu": "rtx-4090",
                "latency_ms": 10.0,
                "experts": [0, 2],
            }
        )

        result = self.storage.get_node("node-1")
        assert result["gpu"] == "rtx-4090"
        assert result["latency_ms"] == 10.0
        assert result["experts"] == [0, 2]

    def test_delete_node(self):
        """Delete a node from the registry."""
        self.storage.upsert_node(
            {
                "node_id": "node-1",
                "gpu": "rtx-4090",
                "latency_ms": 15.0,
                "experts": [0],
            }
        )
        assert self.storage.delete_node("node-1") is True
        assert self.storage.get_node("node-1") is None

    def test_delete_nonexistent_node(self):
        """Deleting a non-existent node returns False."""
        assert self.storage.delete_node("nope") is False

    def test_list_nodes(self):
        """List all registered nodes."""
        for i in range(3):
            self.storage.upsert_node(
                {
                    "node_id": f"node-{i}",
                    "gpu": f"gpu-{i}",
                    "latency_ms": float(i * 10),
                    "experts": [0],
                }
            )
        nodes = self.storage.list_nodes()
        assert len(nodes) == 3

    def test_update_heartbeat(self):
        """Update a node's heartbeat timestamp."""
        self.storage.upsert_node(
            {
                "node_id": "node-1",
                "gpu": "rtx-4090",
                "latency_ms": 20.0,
                "experts": [0],
            }
        )
        self.storage.update_heartbeat("node-1", latency_ms=15.0)

        node = self.storage.get_node("node-1")
        assert node["latency_ms"] == 15.0
        assert node["last_heartbeat"] > 0

    def test_save_and_load_account(self):
        """Save and load a token account."""
        from sawyer.token.accounting import UserAccount
        from sawyer.token.budget import TokenBalance

        balance = TokenBalance(
            tier=SubscriptionTier.EXPLORER,
            monthly_budget=2_000_000,
            current_balance=400_000,
            rollover=50_000,
        )
        account = UserAccount(
            user_id="user-1",
            tier=SubscriptionTier.EXPLORER,
            balance=balance,
            total_tokens_used=100_000,
            total_inferences=50,
            last_inference_at=time.time(),
        )

        self.storage.save_account(account)
        loaded = self.storage.load_account("user-1")

        assert loaded is not None
        assert loaded.user_id == "user-1"
        assert loaded.tier == SubscriptionTier.EXPLORER
        assert loaded.balance.current_balance == 400_000
        assert loaded.balance.rollover == 50_000
        assert loaded.total_tokens_used == 100_000

    def test_load_nonexistent_account(self):
        """Loading a non-existent account returns None."""
        assert self.storage.load_account("nobody") is None

    def test_save_and_load_host_earnings(self):
        """Save and load host earnings."""
        from sawyer.token.budget import HostEarnings

        earnings = HostEarnings(
            node_id="node-1",
            tokens_served=1_000_000,
            credits_earned=1000.0,
            usd_earned=2.0,
        )
        self.storage.save_host_earnings(earnings)

        loaded = self.storage.load_host_earnings("node-1")
        assert loaded is not None
        assert loaded.tokens_served == 1_000_000
        assert loaded.usd_earned == 2.0

    def test_inference_records(self):
        """Save and query inference records."""
        from sawyer.token.accounting import InferenceRecord

        record = InferenceRecord(
            record_id="inf-00000001",
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0, 2],
            input_tokens=800,
            output_tokens=200,
            total_tokens=1000,
            latency_ms=45.0,
            timestamp=time.time(),
            node_id="node-a",
            routing_strategy="adaptive",
        )
        self.storage.save_inference_record(record)

        records = self.storage.get_inference_records("user-1")
        assert len(records) == 1
        assert records[0].record_id == "inf-00000001"
        assert records[0].total_tokens == 1000
        assert records[0].expert_ids == [0, 2]

    def test_audit_log(self):
        """Append and query audit log entries."""
        self.storage.append_audit(
            action="inference",
            actor_id="user-1",
            target_id="node-1",
            silo="tokens",
            details={"model": "mixtral-8x7b", "tokens": 1000},
        )

        entries = self.storage.query_audit(actor_id="user-1")
        assert len(entries) == 1
        assert entries[0]["action"] == "inference"
        assert entries[0]["details"]["model"] == "mixtral-8x7b"

    def test_wal_mode_enabled(self):
        """Database is in WAL mode for concurrent access."""
        conn = self.storage._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


class TestPersistedAccountant:
    """Test PersistedAccountant — in-memory + SQLite sync."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_persisted.db"
        self.storage = SawyerStorage(str(self.db_path))
        self.accountant = PersistedAccountant(self.storage)

    def teardown_method(self):
        self.storage.close()

    def test_create_account_persists(self):
        """Created account is persisted to SQLite."""
        account = self.accountant.create_account("user-1", SubscriptionTier.EXPLORER)
        assert account.balance.total_available == 2_000_000

        # Verify in database directly
        loaded = self.storage.load_account("user-1")
        assert loaded is not None
        assert loaded.user_id == "user-1"
        assert loaded.tier == SubscriptionTier.EXPLORER

    def test_inference_persists_record_and_balance(self):
        """Recording inference persists both the record and updated balance."""
        self.accountant.create_account("user-1", SubscriptionTier.EXPLORER)

        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=400,
            output_tokens=100,
            latency_ms=50.0,
            node_id="node-a",
        )

        # Verify record in database
        records = self.storage.get_inference_records("user-1")
        assert len(records) == 1
        assert records[0].total_tokens == 500

        # Verify balance updated in database
        loaded = self.storage.load_account("user-1")
        assert loaded.balance.current_balance == 2_000_000 - 500

    def test_host_earnings_persist(self):
        """Host earnings are persisted to SQLite."""
        self.accountant.create_account("user-1", SubscriptionTier.EXPLORER)

        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=500,
            output_tokens=500,
            latency_ms=100.0,
            node_id="node-a",
        )

        # Verify earnings in database
        earnings = self.storage.load_host_earnings("node-a")
        assert earnings is not None
        assert earnings.tokens_served == 1000

    def test_survives_restart(self):
        """Account state survives a database reload."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=2_000_000,
            output_tokens=2_000_000,
            latency_ms=100.0,
        )

        # Create a new accountant from the same database
        new_accountant = PersistedAccountant(self.storage)

        # Account should be loaded from DB
        account = new_accountant.get_account("user-1")
        assert account is not None
        assert account.balance.total_available == 1_000_000  # 2M - 1M

    def test_billing_cycle_persists(self):
        """Billing cycle rollover is persisted."""
        self.accountant.create_account("user-1", SubscriptionTier.EXPLORER)
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=200_000,
            output_tokens=100_000,
            latency_ms=100.0,
        )

        self.accountant.process_billing_cycle("user-1")

        # Verify rollover in database
        loaded = self.storage.load_account("user-1")
        assert loaded is not None
        assert loaded.balance.rollover == 200_000

    def test_insufficient_tokens_still_works(self):
        """InsufficientTokens exception works with persisted accountant."""
        self.accountant.create_account("user-1", SubscriptionTier.EXPLORER)
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=400_000,
            output_tokens=100_000,
            latency_ms=100.0,
        )

        with pytest.raises(InsufficientTokens):
            self.accountant.record_inference(
                user_id="user-1",
                model_name="mixtral-8x7b",
                expert_ids=[0],
                input_tokens=1,
                output_tokens=1,
                latency_ms=50.0,
            )

    def test_audit_trail_on_inference(self):
        """Every inference creates an audit log entry."""
        self.accountant.create_account("user-1", SubscriptionTier.EXPLORER)
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=100,
            output_tokens=50,
            latency_ms=50.0,
            node_id="node-a",
        )

        entries = self.storage.query_audit(actor_id="user-1")
        assert len(entries) == 1
        assert entries[0]["action"] == "inference"

    def test_inference_history(self):
        """get_inference_history returns records from database."""
        self.accountant.create_account("user-1", SubscriptionTier.EXPLORER)

        for i in range(5):
            self.accountant.record_inference(
                user_id="user-1",
                model_name="mixtral-8x7b",
                expert_ids=[0],
                input_tokens=100,
                output_tokens=50,
                latency_ms=float(i * 10),
            )

        history = self.accountant.get_inference_history("user-1", limit=3)
        assert len(history) == 3
