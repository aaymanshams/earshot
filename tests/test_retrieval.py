from earshot.chunking import fixed_chunks, sentence_chunks
from earshot.embeddings import HashingEmbedder
from earshot.retrieval import LexicalReranker, NoteIndex, NoteStore
from earshot.schemas import Note


def test_sentence_chunks_keep_facts_whole():
    note = Note(title="Doctors", text="Family doctor: Dr. Imran Sheikh, every month.\nAudiologist: Dr. Neha Bose.")
    chunks = sentence_chunks(note, "n0")
    assert any("Imran Sheikh" in c.text for c in chunks)
    assert any("Neha Bose" in c.text for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_titles_do_not_end_a_sentence():
    note = Note(title="Doctors", text="Family doctor: Dr. Imran Sheikh visits monthly. He is kind.")
    assert any("Imran Sheikh" in c.text for c in sentence_chunks(note, "n0"))


def test_fixed_chunks_cover_the_text():
    note = Note(title="T", text="abc " * 200)
    chunks = fixed_chunks(note, "n0")
    assert len(chunks) > 1
    assert chunks[-1].text.endswith("abc")


def test_search_finds_the_right_note():
    index = NoteIndex(HashingEmbedder())
    index.add([Note(title="Food", text="I am vegetarian."), Note(title="Bank", text="My bank is Harbor Bank.")])
    assert "Harbor" in index.search("Which bank?", 1)[0].chunk.text


def test_empty_index_returns_nothing():
    assert NoteIndex(HashingEmbedder()).search("anything", 3) == []


def test_users_never_see_each_others_notes():
    store = NoteStore(HashingEmbedder())
    store.for_user("a").add([Note(title="Secret", text="My code word is pineapple.")])
    assert store.for_user("b").search("code word", 5) == []


def test_lexical_reranker_prefers_coverage():
    index = NoteIndex(HashingEmbedder())
    index.add([Note(title="A", text="blood pressure 118 over 76"), Note(title="B", text="blood sugar")])
    hits = LexicalReranker().rerank("blood pressure", index.search("blood pressure", 2))
    assert "118" in hits[0].chunk.text
