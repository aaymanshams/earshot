import pytest

from earshot.agent import Earshot
from earshot.brain import OfflineBrain
from earshot.config import Settings
from earshot.embeddings import HashingEmbedder
from earshot.retrieval import LexicalReranker, NoteStore
from earshot.schemas import Note

NOTES = [
    Note(title="Allergies", text="Medicine allergy: penicillin causes a skin rash."),
    Note(title="Current medicines", text="Insulin glargine 18 units every night at 10 pm."),
]


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_api_key="", embedder="hashing", reranker="lexical")


@pytest.fixture
def store() -> NoteStore:
    notes = NoteStore(HashingEmbedder())
    notes.for_user("meera").add(NOTES)
    return notes


@pytest.fixture
def agent(store: NoteStore, settings: Settings) -> Earshot:
    return Earshot(OfflineBrain(), store, LexicalReranker(), settings)
