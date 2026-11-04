import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from earshot.agent import Earshot
from earshot.brain import Draft, LLMBrain, OfflineBrain, Usage
from earshot.retrieval import LexicalReranker


def nodes(turn):
    return [step.node for step in turn.trace]


def test_grounded_reply_cites_a_note(agent):
    turn = agent.run("meera", "s1", "Are you allergic to any medicines?")
    assert turn.intent == "personal_question"
    assert "penicillin" in turn.suggestions[0].text
    assert turn.suggestions[0].kind == "grounded"
    assert turn.suggestions[0].source_ids


def test_unknown_fact_gets_a_safe_fallback(agent):
    turn = agent.run("meera", "s1", "What is your father's name?")
    assert turn.suggestions[0].kind == "fallback"
    assert "verify" in nodes(turn)


def test_urgent_speech_skips_retrieval_and_model(agent):
    turn = agent.run("meera", "s1", "There's a fire, get out now!")
    assert turn.intent == "urgent"
    assert turn.alert
    assert nodes(turn) == ["triage", "urgent", "finish"]


def test_small_talk_does_not_search_notes(agent):
    assert "retrieve" not in nodes(agent.run("meera", "s1", "How are you today?"))


def test_every_turn_has_a_repeat_option(agent):
    for text in ["Hello there.", "How are you?", "Where do you live?", "Fire!"]:
        replies = agent.run("meera", "s1", text).suggestions
        assert any(r.text == "Please repeat that." for r in replies)


def test_trace_resets_each_turn(agent):
    agent.run("meera", "s1", "Are you allergic to anything?")
    turn = agent.run("meera", "s1", "Hello there.")
    assert nodes(turn) == ["triage", "simplify", "finish"]


def test_follow_up_uses_conversation_history(agent):
    agent.run("meera", "s2", "What insulin do you take?")
    turn = agent.run("meera", "s2", "And at what time?")
    assert turn.intent == "personal_question"
    assert "10 pm" in turn.suggestions[0].text


def test_sessions_do_not_share_history(agent):
    agent.run("meera", "s3", "What insulin do you take?")
    agent.run("meera", "s4", "And at what time?")
    state = agent.graph.get_state({"configurable": {"thread_id": "meera:s4"}}).values
    assert state["history"] == ["And at what time?"]


class WrongDoseBrain(OfflineBrain):
    """A model that cites the right note and changes the number."""

    calls = 0

    def draft(self, question, hits):
        self.calls += 1
        return Draft(answer="I take 80 units of glargine.", cited=[hits[0].chunk.id]), Usage()


def test_verifier_blocks_a_changed_dose_after_one_retry(store, settings):
    brain = WrongDoseBrain()
    agent = Earshot(brain, store, LexicalReranker(), settings)
    turn = agent.run("meera", "s1", "How much insulin do you take?")
    assert brain.calls == 2
    assert nodes(turn).count("verify") == 2
    assert turn.suggestions[0].kind == "fallback"
    assert all("80" not in r.text for r in turn.suggestions)


class FakeCiteBrain(OfflineBrain):
    def draft(self, question, hits):
        return Draft(answer="penicillin rash", cited=["n99#s0"]), Usage()


def test_citation_outside_the_results_is_rejected(store, settings):
    agent = Earshot(FakeCiteBrain(), store, LexicalReranker(), settings)
    turn = agent.run("meera", "s1", "Are you allergic to any medicines?")
    assert turn.suggestions[0].kind == "fallback"


def reply(payload):
    message = AIMessage(content=json.dumps(payload))
    message.usage_metadata = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    return message


def test_llm_brain_records_tokens_and_cost(store, settings):
    model = FakeMessagesListChatModel(responses=[
        reply({"intent": "personal_question", "query": "Are you allergic to medicines?"}),
        reply({"answer": "Yes, penicillin gives me a skin rash.", "cited": ["n0#s0"]}),
        reply({"caption": "Are you allergic to any medicine?"}),
    ])
    priced = settings.model_copy(update={"llm_price_input_per_m": 1.0, "llm_price_output_per_m": 2.0})
    agent = Earshot(LLMBrain(model), store, LexicalReranker(), priced)
    turn = agent.run("meera", "s1", "Are you allergic to any medicines?")
    assert turn.suggestions[0].kind == "grounded"
    assert sum(s.input_tokens for s in turn.trace) == 300
    assert turn.cost_usd == pytest.approx((300 * 1 + 60 * 2) / 1_000_000)


def test_bad_model_output_falls_back_without_losing_the_caption(store, settings):
    model = FakeMessagesListChatModel(responses=[AIMessage(content="not json")])
    agent = Earshot(LLMBrain(model), store, LexicalReranker(), settings)
    turn = agent.run("meera", "s1", "The doctor will see you now.")
    assert turn.intent == "statement"
    assert turn.caption == "The doctor will see you now."
