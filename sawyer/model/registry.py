"""Sawyer model registry — supported MoE models and expert layouts."""

from dataclasses import dataclass


@dataclass
class ExpertLayout:
    """Description of a single expert within a MoE model."""

    expert_id: int
    param_count: float  # billions
    size_gb_q4: float  # size in GB at Q4_K_M quantization
    layers: int


@dataclass
class MoEModel:
    """A supported Mixture-of-Experts model."""

    name: str
    display_name: str
    total_params_b: float  # total parameters in billions
    num_experts: int  # total number of experts
    active_experts: int  # experts activated per token (gating sparsity)
    model_size_gb_q4: float  # total Q4_K_M size
    expert_size_gb_q4: float  # per-expert Q4_K_M size
    context_length: int  # maximum context window
    gating_type: str  # "top_k", "top_n", "shared"
    hf_repo: str  # HuggingFace repo ID
    experts: list[ExpertLayout] | None = None  # detailed expert layout (if available)


# Registry of supported models
MODELS: dict[str, MoEModel] = {
    "mixtral-8x7b": MoEModel(
        name="mixtral-8x7b",
        display_name="Mixtral 8x7B",
        total_params_b=46.7,
        num_experts=8,
        active_experts=2,
        model_size_gb_q4=24.0,
        expert_size_gb_q4=1.5,
        context_length=32768,
        gating_type="top_k",
        hf_repo="TheBloke/Mixtral-8x7B-v0.1-GGUF",
    ),
    "deepseek-v2-lite": MoEModel(
        name="deepseek-v2-lite",
        display_name="DeepSeek-V2 Lite",
        total_params_b=15.7,
        num_experts=64,
        active_experts=6,
        model_size_gb_q4=9.0,
        expert_size_gb_q4=0.8,
        context_length=131072,
        gating_type="shared",
        hf_repo="deepseek-ai/DeepSeek-V2-Lite-Chat-GGUF",
    ),
    "qwen1.5-moe-a2.7b": MoEModel(
        name="qwen1.5-moe-a2.7b",
        display_name="Qwen1.5-MoE-A2.7B",
        total_params_b=14.3,
        num_experts=60,
        active_experts=4,
        model_size_gb_q4=7.0,
        expert_size_gb_q4=0.5,
        context_length=32768,
        gating_type="top_k",
        hf_repo="Qwen/Qwen1.5-MoE-A2.7B-GGUF",
    ),
    "dbrx": MoEModel(
        name="dbrx",
        display_name="DBRX",
        total_params_b=132.0,
        num_experts=16,
        active_experts=4,
        model_size_gb_q4=65.0,
        expert_size_gb_q4=2.5,
        context_length=32768,
        gating_type="top_k",
        hf_repo="databricks/dbrx-instruct-GGUF",
    ),
}


def get_model(name: str) -> MoEModel:
    """Look up a model by name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name]


def list_models() -> list[MoEModel]:
    """Return all supported models."""
    return list(MODELS.values())


def can_host_expert(model: MoEModel, available_vram_gb: float) -> bool:
    """Check if a node has enough VRAM to host an expert for this model."""
    # Leave 1GB headroom for KV cache and overhead
    return available_vram_gb >= model.expert_size_gb_q4 + 1.0
