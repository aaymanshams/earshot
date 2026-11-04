"""The LangGraph workflow that handles one spoken turn.

                       +-> urgent ------------------------------+
    start -> triage ---+-> simplify -> (statement/general) ------+-> finish
                       +-> retrieve -> draft -> verify -> simplify
                              ^                    |
                              +---- retry once ----+

Urgent speech skips every model call. Personal facts only reach the user after a
deterministic check against the cited notes, and nothing is ever spoken unless
the user taps it.
"""
import operator
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import hud, safety
from .brain import Brain, Usage
from .config import Settings
from .metrics import Metrics
from .retrieval import NoteStore, Reranker
from .schemas import Hit, HudPage, Intent, NodeTrace, Suggestion, TurnResult

MAX_ATTEMPTS = 2
REPEAT = Suggestion(text="Please repeat that.", kind="social")
WRITE = Suggestion(text="Please write that down for me.", kind="social")
CHECK = Suggestion(text="I'm not sure. Let me check and tell you.", kind="fallback")
SOCIAL = [
    Suggestion(text="Yes, thank you.", kind="social"),
    Suggestion(text="No, thank you.", kind="social"),
]
# Lowest evidence score allowed before abstaining, tuned on the evaluation dev split.
MIN_SCORE = {"none": 0.10, "lexical": 0.10, "cross-encoder": 0.10}


def _keep_recent(old: list[str], new: list[str]) -> list[str]:
    return (old + new)[-6:]


def _trace(old: list[NodeTrace], new: list[NodeTrace] | None) -> list[NodeTrace]:
    return [] if new is None else old + new


class TurnState(TypedDict, total=False):
    user_id: str
    text: str
    history: Annotated[list[str], _keep_recent]
    trace: Annotated[list[NodeTrace], _trace]
    attempts: Annotated[int, operator.add]
    intent: Intent
    query: str
    hits: list[Hit]
    answer: str | None
    cited: list[str]
    supported: bool
    alert: str | None
    caption: str
    hud: list[HudPage]
    suggestions: list[Suggestion]


def timed(name: str, fn: Callable[[TurnState], tuple[dict, Usage]]):
    def node(state: TurnState) -> dict:
        start = time.perf_counter()
        update, usage = fn(state)
        ms = (time.perf_counter() - start) * 1000
        step = NodeTrace(
            node=name, ms=round(ms, 2),
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        )
        return {**update, "trace": [step]}

    return node


class Earshot:
    def __init__(
        self, brain: Brain, store: NoteStore, reranker: Reranker,
        settings: Settings, metrics: Metrics | None = None,
    ):
        self.brain = brain
        self.store = store
        self.reranker = reranker
        self.settings = settings
        self.metrics = metrics or Metrics()
        self.min_score = MIN_SCORE[reranker.name]
        self.graph = self._build()

    # Nodes -------------------------------------------------------------------
    def _triage(self, state: TurnState) -> tuple[dict, Usage]:
        phrase = safety.urgent_phrase(state["text"])
        if phrase:
            return {"intent": "urgent", "alert": phrase, "query": state["text"]}, Usage()
        past = state.get("history", [])[:-1]
        result, usage = self.brain.classify(state["text"], past)
        return {"intent": result.intent, "query": result.query, "alert": None}, usage

    def _urgent(self, state: TurnState) -> tuple[dict, Usage]:
        return {
            "caption": f"ALERT: {state['text']}",
            "suggestions": [Suggestion(text="I am deaf. Please help me.", kind="social"), REPEAT],
        }, Usage()

    def _retrieve(self, state: TurnState) -> tuple[dict, Usage]:
        widen = 1 + state.get("attempts", 0)
        index = self.store.for_user(state["user_id"])
        found = index.search(state["query"], self.settings.retrieve_k * widen)
        ranked = self.reranker.rerank(state["query"], found)
        kept = [h for h in ranked if h.score >= self.min_score]
        return {"hits": kept[: self.settings.rerank_k * widen]}, Usage()

    def _draft(self, state: TurnState) -> tuple[dict, Usage]:
        draft, usage = self.brain.draft(state["query"], state.get("hits", []))
        return {"answer": draft.answer, "cited": draft.cited}, usage

    def _verify(self, state: TurnState) -> tuple[dict, Usage]:
        answer, cited = state.get("answer"), state.get("cited", [])
        by_id = {h.chunk.id: h.chunk for h in state.get("hits", [])}
        evidence = [by_id[c] for c in cited if c in by_id]
        supported = bool(
            answer and evidence and len(evidence) == len(cited)
            and safety.is_supported(
                answer, state["query"], " ".join(f"{c.title} {c.text}" for c in evidence)
            )
        )
        return {"supported": supported, "attempts": 1}, Usage()

    def _simplify(self, state: TurnState) -> tuple[dict, Usage]:
        caption, usage = self.brain.simplify(state["text"])
        return {"caption": caption}, usage

    def _finish(self, state: TurnState) -> tuple[dict, Usage]:
        suggestions = state.get("suggestions")
        if state["intent"] == "personal_question":
            if state.get("supported"):
                top = Suggestion(text=state["answer"], source_ids=state["cited"], kind="grounded")
            else:
                top = CHECK
            suggestions = [top, REPEAT, WRITE]
        elif state["intent"] == "general_question":
            suggestions = [*SOCIAL, REPEAT]
        elif state["intent"] == "statement":
            suggestions = [Suggestion(text="Okay, I understand.", kind="social"), REPEAT, WRITE]
        pages = hud.fit(state["caption"], self.settings.hud_columns, self.settings.hud_rows)
        return {"hud": pages, "suggestions": suggestions}, Usage()

    # Routing -----------------------------------------------------------------
    @staticmethod
    def _after_triage(state: TurnState) -> str:
        return {"urgent": "urgent", "personal_question": "retrieve"}.get(state["intent"], "simplify")

    @staticmethod
    def _after_verify(state: TurnState) -> str:
        if not state["supported"] and state.get("attempts", 0) < MAX_ATTEMPTS:
            return "retrieve"
        return "simplify"

    def _build(self):
        graph = StateGraph(TurnState)
        for name, fn in {
            "triage": self._triage, "urgent": self._urgent, "retrieve": self._retrieve,
            "draft": self._draft, "verify": self._verify, "simplify": self._simplify,
            "finish": self._finish,
        }.items():
            graph.add_node(name, timed(name, fn))
        graph.add_edge(START, "triage")
        graph.add_conditional_edges("triage", self._after_triage, ["urgent", "retrieve", "simplify"])
        graph.add_edge("urgent", "finish")
        graph.add_edge("retrieve", "draft")
        graph.add_edge("draft", "verify")
        graph.add_conditional_edges("verify", self._after_verify, ["retrieve", "simplify"])
        graph.add_edge("simplify", "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=MemorySaver())

    # Running -----------------------------------------------------------------
    @staticmethod
    def _turn_input(user_id: str, text: str) -> dict:
        return {
            "user_id": user_id, "text": text, "history": [text], "trace": None,
            "attempts": 0, "hits": [], "answer": None, "cited": [], "supported": False,
            "suggestions": None,
        }

    def _config(self, user_id: str, session_id: str) -> dict:
        # User id is part of the thread, so sessions cannot cross between users.
        return {"configurable": {"thread_id": f"{user_id}:{session_id}"}}

    def _result(self, state: dict, total_ms: float) -> TurnResult:
        trace = state["trace"]
        tokens_in = sum(t.input_tokens for t in trace)
        tokens_out = sum(t.output_tokens for t in trace)
        cost = (
            tokens_in * self.settings.llm_price_input_per_m
            + tokens_out * self.settings.llm_price_output_per_m
        ) / 1_000_000
        result = TurnResult(
            turn_id=uuid.uuid4().hex[:12], intent=state["intent"], caption=state["caption"],
            hud=state["hud"], suggestions=state["suggestions"], alert=state.get("alert"),
            trace=trace, total_ms=round(total_ms, 2), cost_usd=round(cost, 7),
            brain=self.brain.name,
        )
        self.metrics.record(result)
        return result

    def run(self, user_id: str, session_id: str, text: str) -> TurnResult:
        start = time.perf_counter()
        config = self._config(user_id, session_id)
        state = self.graph.invoke(self._turn_input(user_id, text), config)
        return self._result(state, (time.perf_counter() - start) * 1000)

    async def stream(self, user_id: str, session_id: str, text: str) -> AsyncIterator[dict]:
        """Yield each finished node, then the complete turn result."""
        start = time.perf_counter()
        config = self._config(user_id, session_id)
        updates = self.graph.astream(self._turn_input(user_id, text), config, stream_mode="updates")
        async for update in updates:
            for name in update:
                yield {"type": "node", "node": name}
        state = (await self.graph.aget_state(config)).values
        result = self._result(state, (time.perf_counter() - start) * 1000)
        yield {"type": "result", "result": result.model_dump()}
