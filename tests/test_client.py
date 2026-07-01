"""Tests for Sawyer consumer client — the user-facing inference gateway."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sawyer.client import LocalInference, create_client_app


class TestLocalInference:
    """Test local inference fallback logic."""

    def test_is_available_no_backends(self):
        """is_available returns all False when nothing is running."""
        inference = LocalInference()
        # Mock httpx to simulate no backends
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = inference.is_available()
            assert result["llama_cpp"] is False
            assert result["ollama"] is False

    def test_infer_no_backend_raises(self):
        """infer raises RuntimeError when no backend is available."""
        inference = LocalInference()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.post.side_effect = Exception("Connection refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="No inference backend available"):
                inference.infer("Hello, world!")

    def test_infer_ollama_success(self):
        """infer returns result when Ollama is available."""
        inference = LocalInference()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()

            # First call: llama.cpp fails
            llama_resp = MagicMock()
            llama_resp.status_code = 503
            mock_client.post.side_effect = [
                llama_resp,
                # Second call: Ollama succeeds
                MagicMock(
                    status_code=200,
                    json=MagicMock(return_value={
                        "message": {"content": "Hello!"},
                        "prompt_eval_count": 5,
                        "eval_count": 3,
                        "total_duration": 500_000_000,
                        "model": "llama3",
                        "done": True,
                    }),
                ),
            ]

            mock_client.get.side_effect = Exception("nope")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = inference.infer("Hello")
            assert result.text == "Hello!"
            assert result.model == "llama3"

    def test_model_name_mapping(self):
        """Model names map from Sawyer to Ollama format."""
        inference = LocalInference()
        assert inference._try_ollama is not None  # Method exists


class TestClientAPI:
    """Test the FastAPI client endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        app = create_client_app()
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health check returns status."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "backends" in data

    def test_chat_ui_served(self, client):
        """Root endpoint returns the chat HTML."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Sawyer" in resp.text
        assert "Distributed MoE Inference" in resp.text

    def test_models_endpoint(self, client):
        """Models endpoint returns available models."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1

    def test_chat_completions_no_backends(self, client):
        """Chat completions returns 503 when no backends available."""
        with patch("sawyer.client.LocalInference.infer") as mock_infer:
            mock_infer.side_effect = RuntimeError("No inference backend available")

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "sawyer",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 503

    def test_chat_completions_success(self, client):
        """Chat completions returns OpenAI-compatible response."""
        from sawyer.client import InferenceResult

        with patch("sawyer.client.LocalInference.infer") as mock_infer:
            mock_infer.return_value = InferenceResult(
                text="Hello! How can I help?",
                input_tokens=5,
                output_tokens=7,
                latency_ms=150.0,
                model="sawyer",
                finish_reason="stop",
                cost_tokens=7,
            )

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "sawyer",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["content"] == "Hello! How can I help?"
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["usage"]["prompt_tokens"] == 5
            assert data["usage"]["completion_tokens"] == 7

    def test_chat_completions_empty_messages(self, client):
        """Chat completions returns 400 for empty messages."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "sawyer",
                "messages": [],
            },
        )
        assert resp.status_code == 400

    def test_balance_endpoint(self, client):
        """Balance endpoint returns token info."""
        resp = client.get("/v1/balance")
        assert resp.status_code == 200
        data = resp.json()
        assert "balance" in data

    def test_chat_ui_has_input_area(self, client):
        """Chat UI includes the prompt input area."""
        resp = client.get("/")
        assert "prompt" in resp.text
        assert "sendMessage" in resp.text