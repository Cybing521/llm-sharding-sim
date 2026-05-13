"""LLM model parameters: layers, sizes, KV cache growth."""

from dataclasses import dataclass, field
import yaml


@dataclass
class LayerInfo:
    index: int
    size_mb: float
    layer_type: str  # "embedding", "transformer", "lm_head"


@dataclass
class ModelConfig:
    name: str
    precision: str
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    kv_cache_bytes_per_layer_per_token: int
    activation_size_bytes: int
    layers: list[LayerInfo] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def total_size_mb(self) -> float:
        return sum(l.size_mb for l in self.layers)

    def layer_size_mb(self, index: int) -> float:
        return self.layers[index].size_mb

    def kv_cache_mb_per_token(self, num_layers_on_device: int) -> float:
        """KV cache memory growth per token for a given number of layers."""
        return (self.kv_cache_bytes_per_layer_per_token * num_layers_on_device) / (1024 * 1024)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)

        layers = []
        layers.append(LayerInfo(
            index=cfg["layers"]["embedding"]["index"],
            size_mb=cfg["layers"]["embedding"]["size_mb"],
            layer_type="embedding",
        ))
        t = cfg["layers"]["transformer"]
        for i in range(t["count"]):
            layers.append(LayerInfo(
                index=t["index_start"] + i,
                size_mb=t["size_per_layer_mb"],
                layer_type="transformer",
            ))
        layers.append(LayerInfo(
            index=cfg["layers"]["lm_head"]["index"],
            size_mb=cfg["layers"]["lm_head"]["size_mb"],
            layer_type="lm_head",
        ))

        return cls(
            name=cfg["name"],
            precision=cfg["precision"],
            hidden_size=cfg["hidden_size"],
            num_attention_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            head_dim=cfg["head_dim"],
            kv_cache_bytes_per_layer_per_token=cfg["kv_cache"]["bytes_per_layer_per_token"],
            activation_size_bytes=cfg["activation_size_bytes"],
            layers=layers,
        )
