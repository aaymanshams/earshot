"""In-memory quality, latency and cost tracking for recent turns."""
from collections import Counter, deque

import numpy as np

from .schemas import TurnResult


class Metrics:
    def __init__(self, window: int = 1000):
        self.turns: deque[TurnResult] = deque(maxlen=window)

    def record(self, turn: TurnResult) -> None:
        self.turns.append(turn)

    def summary(self) -> dict:
        turns = list(self.turns)
        if not turns:
            return {"turns": 0}
        totals = [t.total_ms for t in turns]
        by_node: dict[str, list[float]] = {}
        for turn in turns:
            for step in turn.trace:
                by_node.setdefault(step.node, []).append(step.ms)
        kinds = Counter(s.kind for t in turns if t.intent == "personal_question" for s in t.suggestions[:1])
        return {
            "turns": len(turns),
            "latency_ms": _percentiles(totals),
            "nodes_ms": {name: _percentiles(ms) for name, ms in by_node.items()},
            "intents": dict(Counter(t.intent for t in turns)),
            "personal_replies": dict(kinds),
            "tokens": sum(s.input_tokens + s.output_tokens for t in turns for s in t.trace),
            "cost_usd": round(sum(t.cost_usd for t in turns), 6),
        }


def _percentiles(values: list[float]) -> dict:
    return {
        "p50": round(float(np.percentile(values, 50)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
    }
