"""Measure Earshot on the labelled sets and write evals/results/.

    python -m evals.run            # everything that is installed
    python -m evals.run --fast     # no model downloads

All data is synthetic. Thresholds are chosen on the dev split and reported on the
test split, so the final numbers are not tuned on the data they are reported for.
"""
import argparse
import json
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from earshot import hud
from earshot.agent import Earshot
from earshot.brain import build_brain
from earshot.config import Settings
from earshot.embeddings import get_embedder
from earshot.retrieval import NoteIndex, NoteStore, get_reranker
from earshot.schemas import Note

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"


def load(name: str):
    return json.loads((DATA / name).read_text())


def personas() -> dict[str, list[Note]]:
    return {k: [Note(**n) for n in v] for k, v in load("personas.json").items()}


def retrieval_grid(fast: bool) -> list[dict]:
    """Compare chunking, embedding and re-ranking on answerable test questions."""
    questions = [q for q in load("questions.json") if q["gold"] and q["split"] == "test"]
    embedders = ["hashing"] if fast else ["hashing", "minilm"]
    rerankers = ["none", "lexical"] if fast else ["none", "lexical", "cross-encoder"]
    rows = []
    for embedder_name in embedders:
        embedder = get_embedder(embedder_name)
        for chunking in ("fixed", "sentence"):
            indexes = {}
            for persona, notes in personas().items():
                indexes[persona] = NoteIndex(embedder, chunking)
                indexes[persona].add(notes)
            for reranker_name in rerankers:
                reranker = get_reranker(reranker_name)
                ranks, times = [], []
                for q in questions:
                    start = time.perf_counter()
                    hits = reranker.rerank(q["question"], indexes[q["persona"]].search(q["question"], 12))
                    times.append((time.perf_counter() - start) * 1000)
                    ranks.append(next(
                        (i + 1 for i, h in enumerate(hits)
                         if q["gold"] in f"{h.chunk.title} {h.chunk.text}"), None,
                    ))
                rows.append({
                    "embedder": embedder_name, "chunking": chunking, "reranker": reranker_name,
                    "recall@1": _recall(ranks, 1), "recall@3": _recall(ranks, 3),
                    "mrr": round(float(np.mean([1 / r if r else 0 for r in ranks])), 3),
                    "p95_ms": round(float(np.percentile(times, 95)), 1),
                })
    return rows


def _recall(ranks: list, k: int) -> float:
    return round(sum(1 for r in ranks if r and r <= k) / len(ranks), 3)


def make_agent(settings: Settings, brain, min_score: float | None = None) -> Earshot:
    store = NoteStore(get_embedder(settings.embedder))
    for persona, notes in personas().items():
        store.for_user(persona).add(notes)
    agent = Earshot(brain, store, get_reranker(settings.reranker), settings)
    if min_score is not None:
        agent.min_score = min_score
    return agent


def replies(agent: Earshot, split: str) -> dict:
    """Score the whole agent. A wrong fact is worse than saying "let me check"."""
    questions = [q for q in load("questions.json") if q["split"] == split]
    counts = dict.fromkeys(
        ["answerable", "correct", "wrong", "abstained", "unanswerable", "safe", "invented"], 0
    )
    failures, latencies = [], []
    for q in questions:
        turn = agent.run(q["persona"], f"eval-{q['id']}", q["question"])
        latencies.append(turn.total_ms)
        top = turn.suggestions[0]
        grounded = turn.intent == "personal_question" and top.kind == "grounded"
        if q["gold"]:
            counts["answerable"] += 1
            if grounded and q["gold"].lower() in _evidence(agent, q["persona"], top).lower():
                counts["correct"] += 1
            elif grounded:
                counts["wrong"] += 1
                failures.append({"q": q["question"], "got": top.text, "want": q["gold"]})
            else:
                counts["abstained"] += 1
                failures.append({"q": q["question"], "got": "(abstained)", "want": q["gold"]})
        else:
            counts["unanswerable"] += 1
            if grounded:
                counts["invented"] += 1
                failures.append({"q": q["question"], "got": top.text, "want": "(abstain)"})
            else:
                counts["safe"] += 1
    answerable, unanswerable = counts["answerable"], counts["unanswerable"]
    return {
        **counts,
        "answer_accuracy": round(counts["correct"] / answerable, 3),
        "wrong_fact_rate": round((counts["wrong"] + counts["invented"]) / len(questions), 3),
        "abstain_when_unknown": round(counts["safe"] / unanswerable, 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "failures": failures,
    }


def _evidence(agent: Earshot, persona: str, suggestion) -> str:
    chunks = {c.id: c for c in agent.store.for_user(persona).chunks}
    text = suggestion.text
    cited = " ".join(f"{chunks[i].title} {chunks[i].text}" for i in suggestion.source_ids)
    return f"{text} {cited}"


def tune_threshold(settings: Settings, brain) -> tuple[float, list[dict]]:
    """Pick the abstention threshold with the fewest wrong facts on dev.

    Ties go to the higher answer accuracy, then the lower threshold.
    """
    sweep = []
    for value in np.arange(0.1, 0.96, 0.05):
        result = replies(make_agent(settings, brain, float(value)), "dev")
        sweep.append({
            "threshold": round(float(value), 2), "accuracy": result["answer_accuracy"],
            "wrong_fact_rate": result["wrong_fact_rate"],
        })
    best = max(sweep, key=lambda r: (r["accuracy"] - 2 * r["wrong_fact_rate"], -r["threshold"]))
    return best["threshold"], sweep


def triage(brain, dataset: str = "triage.json") -> dict:
    agent = Earshot(brain, NoteStore(get_embedder("hashing")), get_reranker("none"), Settings())
    correct, total, per_class, misses = 0, 0, {}, []
    for label, utterances in load(dataset).items():
        hits = 0
        for text in utterances:
            got = agent.run("triage", f"t{total}", text).intent
            hits += got == label
            total += 1
            if got != label:
                misses.append({"text": text, "want": label, "got": got})
        correct += hits
        per_class[label] = round(hits / len(utterances), 3)
    return {"accuracy": round(correct / total, 3), "per_class": per_class, "misses": misses}


def urgent_recall(brain) -> dict:
    """Urgent speech must be flagged and must not touch the model."""
    agent = Earshot(brain, NoteStore(get_embedder("hashing")), get_reranker("none"), Settings())
    turns = [agent.run("u", f"u{i}", t) for i, t in enumerate(load("triage.json")["urgent"])]
    return {
        "recall": round(sum(t.intent == "urgent" for t in turns) / len(turns), 3),
        "p95_ms": round(float(np.percentile([t.total_ms for t in turns], 95)), 2),
        "tokens_used": sum(s.input_tokens + s.output_tokens for t in turns for s in t.trace),
    }


def caption_safety(brain) -> dict:
    cases = load("simplify.json")
    kept = 0
    for case in cases:
        caption, _ = brain.simplify(case["text"])
        kept += all(item in caption for item in case["keep"])
    fits = all(
        len(line) <= 32 and len(page.lines) <= 2
        for case in cases for page in hud.fit(case["text"]) for line in page.lines
    )
    return {"facts_preserved": round(kept / len(cases), 3), "cases": len(cases), "hud_fits": fits}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1] / max(len(reference), 1)


def speech(model: str) -> dict | None:
    """Clean macOS speech synthesis, so this is a best case, not real-world accuracy."""
    if platform.system() != "Darwin":
        return None
    from earshot.speech import transcribe

    sentences = load("asr.json")
    errors, times, words = 0.0, [], 0
    with tempfile.TemporaryDirectory() as folder:
        for index, sentence in enumerate(sentences):
            path = Path(folder) / f"{index}.wav"
            subprocess.run(
                ["say", "-o", str(path), "--data-format=LEI16@16000", sentence], check=True
            )
            audio = path.read_bytes()
            start = time.perf_counter()
            text = transcribe(audio, model)["text"]
            if index:  # The first call includes model loading.
                times.append((time.perf_counter() - start) * 1000)
            reference = _words(sentence)
            errors += word_error_rate(reference, _words(text)) * len(reference)
            words += len(reference)
    return {
        "model": model, "sentences": len(sentences), "wer": round(errors / words, 3),
        "p50_ms": round(float(np.percentile(times, 50)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
        "source": "macOS say, clean audio",
    }


def report(results: dict) -> str:
    lines = ["# Evaluation results", "", f"Brain: `{results['brain']}`. All data is synthetic.", ""]
    lines += ["## Retrieval (answerable test questions)", ""]
    lines += ["| Embedder | Chunking | Re-ranker | Recall@1 | Recall@3 | MRR | p95 ms |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for r in results["retrieval"]:
        lines.append(
            f"| {r['embedder']} | {r['chunking']} | {r['reranker']} | {r['recall@1']} "
            f"| {r['recall@3']} | {r['mrr']} | {r['p95_ms']} |"
        )
    for name, value in results["agent"].items():
        lines += ["", f"## Full agent: {name}", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in value.items() if k != "failures"]
    lines += ["", "## Triage", "", f"Accuracy: {results['triage']['accuracy']}", ""]
    lines += [f"- {k}: {v}" for k, v in results["triage"]["per_class"].items()]
    hard = results["triage_hard"]
    lines += ["", f"Hard set, written after the rules and not tuned on: {hard['accuracy']}", ""]
    lines += [f"- {k}: {v}" for k, v in hard["per_class"].items()]
    lines += ["", "## Urgent speech", "", json.dumps(results["urgent"]), ""]
    lines += ["## Caption safety", "", json.dumps(results["captions"]), ""]
    if results.get("speech"):
        lines += ["## Speech", "", json.dumps(results["speech"]), ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", default="latest")
    args = parser.parse_args()

    base = Settings()
    brain = build_brain(base)
    configs = {"baseline (hashing, lexical)": ("hashing", "lexical")}
    if not args.fast:
        configs["best (minilm, cross-encoder)"] = ("minilm", "cross-encoder")

    results = {"brain": brain.name, "retrieval": retrieval_grid(args.fast), "agent": {}, "tuning": {}}
    for name, (embedder, reranker) in configs.items():
        settings = base.model_copy(update={"embedder": embedder, "reranker": reranker})
        threshold, sweep = tune_threshold(settings, brain)
        results["tuning"][name] = {"chosen": threshold, "sweep": sweep}
        outcome = replies(make_agent(settings, brain, threshold), "test")
        results["agent"][name] = {"threshold": threshold, **outcome}
    results["triage"] = triage(brain)
    results["triage_hard"] = triage(brain, "triage_hard.json")
    results["urgent"] = urgent_recall(brain)
    results["captions"] = caption_safety(brain)
    if not args.fast:
        results["speech"] = speech(base.whisper_model)

    OUT.mkdir(exist_ok=True)
    (OUT / f"{args.out}.json").write_text(json.dumps(results, indent=2))
    (OUT / f"{args.out}.md").write_text(report(results))
    print(report(results))


if __name__ == "__main__":
    main()
