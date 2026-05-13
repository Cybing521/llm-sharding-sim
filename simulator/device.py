"""Edge device model: memory tracking, layer placement, KV cache."""

from dataclasses import dataclass, field
import yaml
import numpy as np


@dataclass
class Device:
    index: int
    name: str
    memory_mb: float
    compute_capability_gflops: float
    tokens_per_second_per_layer: float

    placed_layers: list[int] = field(default_factory=list)
    kv_cache_mb: float = 0.0

    @property
    def used_memory_mb(self) -> float:
        return self._static_memory_mb + self.kv_cache_mb

    @property
    def _static_memory_mb(self) -> float:
        return self._layer_memory_mb

    @_static_memory_mb.setter
    def _static_memory_mb(self, _):
        pass

    def set_layer_memory(self, total_layer_mb: float):
        self._layer_memory_mb = total_layer_mb

    @property
    def free_memory_mb(self) -> float:
        return self.memory_mb - self.used_memory_mb

    def can_fit_layer(self, layer_size_mb: float) -> bool:
        return self.free_memory_mb >= layer_size_mb

    def reset_kv_cache(self):
        self.kv_cache_mb = 0.0

    def __post_init__(self):
        self._layer_memory_mb = 0.0


@dataclass
class DeviceCluster:
    devices: list[Device]
    bandwidth_mbps: np.ndarray   # [K, K] matrix
    latency_ms: np.ndarray       # [K, K] matrix

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    def transfer_time_s(self, src: int, dst: int, data_bytes: int) -> float:
        """Total transfer time in seconds between two devices."""
        if src == dst:
            return 0.0
        data_mb = data_bytes / (1024 * 1024)
        bw = self.bandwidth_mbps[src, dst]
        if bw <= 0:
            return float("inf")
        return data_mb / bw + self.latency_ms[src, dst] / 1000.0

    @classmethod
    def from_yaml(cls, path: str) -> "DeviceCluster":
        with open(path) as f:
            cfg = yaml.safe_load(f)

        devices = []
        for i, d in enumerate(cfg["devices"]):
            devices.append(Device(
                index=i,
                name=d["name"],
                memory_mb=d["memory_mb"],
                compute_capability_gflops=d["compute_capability_gflops"],
                tokens_per_second_per_layer=d["tokens_per_second_per_layer"],
            ))

        bw = np.array(cfg["bandwidth_mbps"], dtype=float)
        lat = np.array(cfg["latency_ms"], dtype=float)
        return cls(devices=devices, bandwidth_mbps=bw, latency_ms=lat)
