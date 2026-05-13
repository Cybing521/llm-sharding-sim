"""User request model for LLM inference."""

from dataclasses import dataclass
import yaml


@dataclass
class Request:
    id: str
    prompt_length: int       # |r_q|
    output_length: int       # g_q (estimated)
    arrival_device: int      # which device the request arrives at
    arrival_time: float      # seconds

    @property
    def total_tokens(self) -> int:
        return self.prompt_length + self.output_length


def load_requests(path: str) -> list[Request]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    requests = []
    for r in cfg["requests"]:
        requests.append(Request(
            id=r["id"],
            prompt_length=r["prompt_length"],
            output_length=r["estimated_output_length"],
            arrival_device=r["arrival_device"],
            arrival_time=r.get("arrival_time", 0.0),
        ))
    return requests
