"""Two chunking strategies, compared in the evaluation suite."""
import re

from .schemas import Chunk, Note

# A full stop ends a sentence unless it belongs to a title such as "Dr." or "Mrs.".
SENTENCE_END = re.compile(r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<=[.!?])\s+(?=[A-Z])")


def fixed_chunks(note: Note, note_id: str, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Character windows: simple, but can cut a fact in half."""
    text = " ".join(note.text.split())
    chunks = []
    for index, start in enumerate(range(0, max(len(text), 1), size - overlap)):
        piece = text[start : start + size]
        if piece.strip():
            chunks.append(Chunk(id=f"{note_id}#f{index}", title=note.title, text=piece))
        if start + size >= len(text):
            break
    return chunks


def sentence_chunks(note: Note, note_id: str, max_chars: int = 240) -> list[Chunk]:
    """One fact-sized piece per line or sentence, each carrying its note title.

    Personal notes are short facts, so a chunk should never cut one in half. The
    title stays attached because "Dr. Rao, Tuesdays" means little without
    "Cardiologist" above it.
    """
    pieces: list[str] = []
    for line in note.text.splitlines():
        line = line.strip(" -*\t")
        if line:
            pieces.extend(s.strip() for s in SENTENCE_END.split(line) if s.strip())

    chunks: list[Chunk] = []
    buffer = ""
    for piece in pieces:
        if buffer and len(buffer) + len(piece) + 1 > max_chars:
            chunks.append(_make(note, note_id, len(chunks), buffer))
            buffer = ""
        # Keep each list line as its own chunk; only join very short fragments.
        if len(piece) >= 25:
            if buffer:
                chunks.append(_make(note, note_id, len(chunks), buffer))
                buffer = ""
            chunks.append(_make(note, note_id, len(chunks), piece))
        else:
            buffer = f"{buffer} {piece}".strip()
    if buffer:
        chunks.append(_make(note, note_id, len(chunks), buffer))
    return chunks


def _make(note: Note, note_id: str, index: int, text: str) -> Chunk:
    return Chunk(id=f"{note_id}#s{index}", title=note.title, text=text)


STRATEGIES = {"fixed": fixed_chunks, "sentence": sentence_chunks}
