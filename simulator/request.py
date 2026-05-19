"""LLM 推理的用户请求模型。"""

from dataclasses import dataclass
import yaml


@dataclass
class Request:
    id: str
    prompt_length: int       # 提示词长度 |r_q|
    output_length: int       # 预估输出长度 g_q
    arrival_device: int      # 请求到达的设备编号
    arrival_time: float      # 到达时间（秒）

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
