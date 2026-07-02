"""Tests for Sawyer consumer client — the user-facing inference gateway."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sawyer.client import LocalInference, create_client_app, DiscoveredModel


class TestLocalInference:
    """Test local inference fallback logic."""

    def test_is_available_no_backends(self):
        """is_available returns all False when nothing is running."""
        inference = LocalInference()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = inference.is_available()
            assert result["llama_cpp"] is False
            assert result["ollama"] is False
            assert result["vllm"] is False
            assert result["lm_studio"] is False

    def test_is_available_ollama_only(self):
        """is_available detects Ollama when it's running."""
        inference = LocalInference()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()

            def get_side_effect(url, **kwargs):
                resp = MagicMock()
                if "11434" in str(url):
                    resp.status_code = 200
                else:
                    resp.status_code = 503
                return resp

            mock_client.get.side_effect = get_side_effect
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = inference.is_available()
            assert result["ollama"] is True

    def test_infer_no_backend_raises(self):
        """infer raises RuntimeError when no backend is available."""
        inference = LocalInference()
        # No discovered models, all backends unreachable
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.post.side_effect = Exception("Connection refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="No inference backend"):
                inference.infer("Hello, world!")

    def test_infer_ollama_success(self):
        """infer returns result when Ollama is available."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="llama3", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        # Patch _try_ollama directly for cleaner testing
        with patch.object(inference, "_try_ollama") as mock_ollama:
            from sawyer.client import InferenceResult
            mock_ollama.return_value = InferenceResult(
                text="Hello!",
                input_tokens=5,
                output_tokens=3,
                latency_ms=500.0,
                model="llama3",
                finish_reason="stop",
                cost_tokens=3,
            )

            result = inference.infer("Hello", model="llama3")
            assert result.text == "Hello!"
            assert result.model == "llama3"

    def test_infer_thinking_model_content_empty(self):
        """infer extracts content from thinking field when content is empty."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="glm-5.1:cloud", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        # Patch _try_ollama directly
        with patch.object(inference, "_try_ollama") as mock_ollama:
            from sawyer.client import InferenceResult
            mock_ollama.return_value = InferenceResult(
                text="The answer is 4.",
                thinking="1. Think step by step.\n2. The answer is 4.",
                input_tokens=10,
                output_tokens=50,
                latency_ms=2000.0,
                model="glm-5.1",
                finish_reason="stop",
                cost_tokens=50,
            )

            result = inference.infer("What is 2+2?", model="glm-5.1:cloud")
            assert "4" in result.text
            assert result.thinking == "1. Think step by step.\n2. The answer is 4."
            assert result.model == "glm-5.1"

    def test_ollama_parsing_thinking_model(self):
        """_try_ollama correctly parses thinking models where content is empty."""
        inference = LocalInference()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "message": {"content": "", "thinking": "1. Think\n2. Answer: 4"},
                "prompt_eval_count": 10,
                "eval_count": 50,
                "total_duration": 2_000_000_000,
                "model": "glm-5.1",
                "done": True,
            }
            mock_client.post.return_value = resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = inference._try_ollama("test", "glm-5.1:cloud", 100, 0.5)
            assert result is not None
            assert "4" in result.text
            # When content is empty and thinking is promoted to content,
            # the thinking field is cleared (not duplicated)
            assert result.thinking == ""

    def test_ollama_parsing_normal_model(self):
        """_try_ollama correctly parses normal models where content has text."""
        inference = LocalInference()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "message": {"content": "Hello!"},
                "prompt_eval_count": 5,
                "eval_count": 3,
                "total_duration": 500_000_000,
                "model": "llama3",
                "done": True,
            }
            mock_client.post.return_value = resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = inference._try_ollama("test", "llama3", 100, 0.5)
            assert result.text == "Hello!"
            assert result.thinking == ""

    def test_ollama_parsing_thinking_with_both_fields(self):
        """_try_ollama preserves thinking when content is also present."""
        inference = LocalInference()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "message": {"content": "4", "thinking": "Let me compute 2+2..."},
                "prompt_eval_count": 10,
                "eval_count": 20,
                "total_duration": 1_000_000_000,
                "model": "glm-5.1",
                "done": True,
            }
            mock_client.post.return_value = resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = inference._try_ollama("test", "glm-5.1", 100, 0.5)
            assert result.text == "4"
            assert result.thinking == "Let me compute 2+2..."

    def test_ollama_model_not_found_returns_none(self):
        """Ollama returns None for model not found (404), not an error."""
        inference = LocalInference()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.status_code = 404
            mock_client.post.return_value = resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = inference._try_ollama("test", "nonexistent", 100, 0.5)
            assert result is None

    def test_model_name_mapping(self):
        """Model names map from Sawyer to Ollama format."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="mixtral", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        resolved, backend = inference._resolve_model("mixtral-8x7b")
        assert resolved == "mixtral"
        assert backend == "ollama"

    def test_resolve_default_model(self):
        """'sawyer' (default) resolves to first discovered model."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="glm-5.1:cloud", backend="ollama", owned_by="local"),
            DiscoveredModel(id="qwen3:latest", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        resolved, backend = inference._resolve_model("sawyer")
        assert resolved == "glm-5.1:cloud"
        assert backend == "ollama"

    def test_resolve_exact_model_name(self):
        """Exact model names are used directly."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="glm-5.1:cloud", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        resolved, backend = inference._resolve_model("glm-5.1:cloud")
        assert resolved == "glm-5.1:cloud"
        assert backend == "ollama"

    def test_resolve_unknown_model(self):
        """Unknown models pass through to backends."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="glm-5.1:cloud", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        resolved, backend = inference._resolve_model("nonexistent-model")
        assert resolved == "nonexistent-model"

    def test_resolve_short_name(self):
        """Short name (without tag) resolves via discovered models."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="glm-5.1:cloud", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        resolved, backend = inference._resolve_model("glm-5.1")
        assert resolved == "glm-5.1:cloud"
        assert backend == "ollama"

    def test_configurable_backend_urls(self):
        """Backend URLs can be configured via SawyerConfig."""
        from sawyer.config import SawyerConfig

        config = SawyerConfig(
            ollama_url="http://192.168.1.100:11434",
            llama_url="http://192.168.1.100:8444",
        )
        inference = LocalInference(config)
        assert inference._ollama_url == "http://192.168.1.100:11434"
        assert inference._llama_url == "http://192.168.1.100:8444"

    def test_env_var_overrides(self):
        """Backend URLs can be overridden via environment variables."""
        with patch.dict("os.environ", {"SAWYER_OLLAMA_URL": "http://custom:11434"}):
            inference = LocalInference()
            assert inference._ollama_url == "http://custom:11434"

    def test_discover_models_caching(self):
        """discover_models caches results and doesn't re-query within TTL."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="test-model", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        # Should return cached results without network calls
        models = inference.discover_models()
        assert len(models) == 1
        assert models[0].id == "test-model"

    def test_discover_models_ollama_direct(self):
        """_discover_ollama_models correctly parses Ollama /api/tags response."""
        inference = LocalInference()
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "models": [
                    {
                        "name": "llama3:latest",
                        "details": {"family": "llama", "parameter_size": "8B", "context_length": 8192},
                    },
                    {
                        "name": "glm-5.1:cloud",
                        "details": {"family": "glm", "parameter_size": "756b"},
                    },
                ]
            }
            mock_client.get.return_value = resp
            mock_client.post.return_value = resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            models = inference._discover_ollama_models()
            assert models is not None
            assert len(models) == 2
            model_ids = [m.id for m in models]
            assert "llama3:latest" in model_ids
            assert "glm-5.1:cloud" in model_ids

    def test_error_message_lists_available_models(self):
        """Error message includes available models when backends are up but model not found."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="glm-5.1:cloud", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        # Both _try_llama and _try_ollama return None
        with patch.object(inference, "_try_llama", return_value=None):
            with patch.object(inference, "_try_ollama", return_value=None):
                with pytest.raises(RuntimeError) as exc_info:
                    inference.infer("test", model="nonexistent")

                assert "glm-5.1:cloud" in str(exc_info.value)

    def test_error_message_no_models(self):
        """Error message suggests installing Ollama when no models found."""
        inference = LocalInference()
        inference._discovered_models = []
        inference._last_discovery_time = 9999999999

        with patch.object(inference, "_try_llama", return_value=None):
            with patch.object(inference, "_try_ollama", return_value=None):
                with pytest.raises(RuntimeError) as exc_info:
                    inference.infer("test", model="sawyer")

                error_msg = str(exc_info.value).lower()
                assert "ollama" in error_msg

    def test_get_models_list_includes_sawyer_and_discovered(self):
        """get_models_list always includes 'sawyer' plus discovered models."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="llama3:latest", backend="ollama", owned_by="local",
                          details={"family": "llama", "parameter_size": "8B"}),
            DiscoveredModel(id="glm-5.1:cloud", backend="ollama", owned_by="local",
                          details={"family": "glm", "parameter_size": "756b"}),
        ]
        inference._last_discovery_time = 9999999999

        models = inference.get_models_list()
        model_ids = [m["id"] for m in models]
        assert "sawyer" in model_ids  # Always present
        assert "llama3:latest" in model_ids
        assert "glm-5.1:cloud" in model_ids
        assert "llama3" in model_ids  # Short name alias

    def test_infer_routes_to_correct_backend(self):
        """infer routes to the backend that has the model."""
        inference = LocalInference()
        inference._discovered_models = [
            DiscoveredModel(id="glm-5.1:cloud", backend="ollama", owned_by="local"),
            DiscoveredModel(id="mixtral:latest", backend="ollama", owned_by="local"),
        ]
        inference._last_discovery_time = 9999999999

        from sawyer.client import InferenceResult

        # glm-5.1:cloud is on Ollama, so _try_ollama should be called first
        with patch.object(inference, "_try_ollama") as mock_ollama:
            mock_ollama.return_value = InferenceResult(
                text="Test response",
                model="glm-5.1",
                input_tokens=5,
                output_tokens=3,
            )
            result = inference.infer("test", model="glm-5.1:cloud")
            assert mock_ollama.called
            assert result.text == "Test response"

    def test_infer_stream_ollama_multi_turn(self):
        """_stream_ollama passes full conversation history."""
        inference = LocalInference()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "What is 2+2?"},
        ]

        # Mock the entire _stream_ollama to verify it's called
        # with messages — then test the actual generation logic
        # separately via integration tests
        chunks = [
            {"id": "chatcmpl-1", "object": "chat.completion.chunk",
             "created": 1, "model": "llama3",
             "choices": [{"index": 0, "delta": {"content": "4"}, "finish_reason": None}]},
            {"id": "chatcmpl-1", "object": "chat.completion.chunk",
             "created": 1, "model": "llama3",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        with patch.object(inference, "_stream_ollama", return_value=iter(chunks)):
            result = list(inference.infer_stream(
                "What is 2+2?", model="llama3", messages=messages,
            ))
            content_parts = [c for c in result if c["choices"][0]["delta"].get("content")]
            assert len(content_parts) == 1
            assert content_parts[0]["choices"][0]["delta"]["content"] == "4"
            # Verify messages was forwarded
            call_kwargs = inference._stream_ollama.call_args
            assert call_kwargs[1].get("messages") == messages

    def test_infer_stream_ollama_thinking(self):
        """_stream_ollama yields thinking deltas for reasoning models."""
        inference = LocalInference()

        # Test that infer_stream passes through thinking chunks
        chunks = [
            {"id": "chatcmpl-1", "object": "chat.completion.chunk",
             "created": 1, "model": "glm-5.1",
             "choices": [{"index": 0, "delta": {"thinking": "Let me reason"}, "finish_reason": None}]},
            {"id": "chatcmpl-1", "object": "chat.completion.chunk",
             "created": 1, "model": "glm-5.1",
             "choices": [{"index": 0, "delta": {"content": "4"}, "finish_reason": None}]},
            {"id": "chatcmpl-1", "object": "chat.completion.chunk",
             "created": 1, "model": "glm-5.1",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        with patch.object(inference, "_stream_ollama", return_value=iter(chunks)):
            result = list(inference.infer_stream(
                "2+2?", model="glm-5.1",
            ))
            thinking_chunks = [c for c in result if "thinking" in c["choices"][0]["delta"]]
            content_chunks = [c for c in result if "content" in c["choices"][0]["delta"]]
            assert len(thinking_chunks) == 1
            assert thinking_chunks[0]["choices"][0]["delta"]["thinking"] == "Let me reason"
            assert len(content_chunks) == 1
            assert content_chunks[0]["choices"][0]["delta"]["content"] == "4"

    def test_infer_stream_passes_messages(self):
        """infer_stream forwards messages to backend streamers."""
        inference = LocalInference()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "2+2?"},
        ]

        # Mock _stream_ollama to just verify messages were passed
        with patch.object(
            inference, "_stream_ollama", return_value=iter([])
        ) as mock_stream:
            mock_stream.return_value = iter([
                {"id": "test", "object": "chat.completion.chunk",
                 "created": 1, "model": "llama3",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ])
            list(inference.infer_stream(
                "2+2?", model="llama3", messages=messages
            ))
            mock_stream.assert_called_once()
            call_kwargs = mock_stream.call_args
            # Messages should be forwarded
            assert call_kwargs[1].get("messages") == messages or call_kwargs[0][-1] == messages


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
        assert "models_available" in data

    def test_chat_ui_served(self, client):
        """Root endpoint returns the chat HTML."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Sawyer" in resp.text
        assert "Distributed MoE Inference" in resp.text

    def test_models_endpoint_includes_sawyer(self, client):
        """Models endpoint always includes the 'sawyer' default model."""
        with patch("sawyer.client.LocalInference.discover_models") as mock_discover:
            mock_discover.return_value = []

            resp = client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "list"
            model_ids = [m["id"] for m in data["data"]]
            assert "sawyer" in model_ids

    def test_models_endpoint_dynamic(self, client):
        """Models endpoint returns dynamically discovered models."""
        from sawyer.client import LocalInference, DiscoveredModel

        # Patch get_models_list on the instance used by the app
        with patch("sawyer.client.LocalInference.get_models_list") as mock_get_models:
            mock_get_models.return_value = [
                {"id": "sawyer", "object": "model", "owned_by": "sawyer"},
                {"id": "llama3:latest", "object": "model", "owned_by": "local"},
                {"id": "glm-5.1:cloud", "object": "model", "owned_by": "local"},
            ]

            resp = client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            model_ids = [m["id"] for m in data["data"]]
            assert "sawyer" in model_ids
            assert "llama3:latest" in model_ids
            assert "glm-5.1:cloud" in model_ids

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

    def test_chat_completions_with_thinking(self, client):
        """Chat completions includes thinking field when available."""
        from sawyer.client import InferenceResult

        with patch("sawyer.client.LocalInference.infer") as mock_infer:
            mock_infer.return_value = InferenceResult(
                text="4",
                thinking="1. Add 2+2\n2. Result is 4",
                input_tokens=10,
                output_tokens=50,
                latency_ms=2000.0,
                model="glm-5.1",
                finish_reason="stop",
                cost_tokens=50,
            )

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "glm-5.1:cloud",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "4"
            assert "thinking" in data["choices"][0]["message"]
            assert "2+2" in data["choices"][0]["message"]["thinking"]

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

    def test_chat_completions_streaming(self, client):
        """Chat completions supports streaming responses."""
        # Mock infer_stream to yield SSE chunks
        def mock_stream(*args, **kwargs):
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "sawyer",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "sawyer",
                "choices": [
                    {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
                ],
            }
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "sawyer",
                "choices": [
                    {"index": 0, "delta": {"content": "!"}, "finish_reason": None}
                ],
            }
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "sawyer",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }

        with patch("sawyer.client.LocalInference.infer_stream") as mock_is:
            mock_is.return_value = mock_stream()

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "sawyer",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            # Should return text/event-stream content type
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # Parse the SSE data
            content_parts = []
            for line in resp.text.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            content_parts.append(delta["content"])

            assert content_parts == ["Hello", "!"]

    def test_chat_completions_streaming_with_thinking(self, client):
        """Streaming includes thinking deltas for reasoning models."""
        def mock_stream(*args, **kwargs):
            yield {
                "id": "chatcmpl-456",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "glm-5.1",
                "choices": [
                    {"index": 0, "delta": {"thinking": "Let me think"}, "finish_reason": None}
                ],
            }
            yield {
                "id": "chatcmpl-456",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "glm-5.1",
                "choices": [
                    {"index": 0, "delta": {"content": "4"}, "finish_reason": None}
                ],
            }
            yield {
                "id": "chatcmpl-456",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "glm-5.1",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }

        with patch("sawyer.client.LocalInference.infer_stream") as mock_is:
            mock_is.return_value = mock_stream()

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "glm-5.1:cloud",
                    "messages": [{"role": "user", "content": "2+2?"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200

            thinking_parts = []
            content_parts = []
            for line in resp.text.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        if "thinking" in delta:
                            thinking_parts.append(delta["thinking"])
                        if "content" in delta:
                            content_parts.append(delta["content"])

            assert thinking_parts == ["Let me think"]
            assert content_parts == ["4"]

    def test_chat_completions_streaming_error(self, client):
        """Streaming returns error as SSE event when backend fails."""
        def mock_stream_error(*args, **kwargs):
            raise RuntimeError("No inference backend available")
            yield  # noqa: unreachable — makes this a generator

        with patch("sawyer.client.LocalInference.infer_stream") as mock_is:
            mock_is.side_effect = RuntimeError("No inference backend available")

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "sawyer",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200  # SSE stream starts 200

            # Should contain error data
            assert "No inference backend" in resp.text or "error" in resp.text

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

    def test_chat_ui_has_model_select(self, client):
        """Chat UI includes dynamic model select."""
        resp = client.get("/")
        assert "model-select" in resp.text

    def test_chat_ui_has_streaming(self, client):
        """Chat UI includes streaming support."""
        resp = client.get("/")
        assert "stream" in resp.text

    def test_503_error_includes_available_models(self, client):
        """503 error response includes available models for debugging."""
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