"""HTTP and WebSocket API, used by the demo and open to any app or web client."""
import json
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .agent import Earshot
from .brain import build_brain
from .config import Settings, get_settings
from .embeddings import get_embedder
from .retrieval import NoteStore, get_reranker
from .schemas import Note, TurnResult
from .speech import SpeechUnavailable, transcribe

ROOT = Path(__file__).resolve().parent.parent
DEMO_USER = "demo"
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class TurnRequest(BaseModel):
    user_id: str = Field(DEMO_USER, max_length=64)
    session_id: str = Field("default", max_length=64)
    text: str = Field(min_length=1, max_length=500)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = NoteStore(get_embedder(settings.embedder))
    agent = Earshot(build_brain(settings), store, get_reranker(settings.reranker), settings)
    demo_notes = json.loads((ROOT / "evals/data/personas.json").read_text())["meera"]
    store.for_user(DEMO_USER).add([Note(**n) for n in demo_notes])

    app = FastAPI(title="Earshot", version="0.1.0")
    app.state.agent = agent

    def allowed(user_id: str, authorization: str | None) -> bool:
        return (
            user_id == DEMO_USER
            or not settings.api_token
            or authorization == f"Bearer {settings.api_token}"
        )

    def authorized(authorization: str | None = Header(default=None)) -> None:
        if settings.api_token and authorization != f"Bearer {settings.api_token}":
            raise HTTPException(401, "Missing or invalid API token")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok", "brain": agent.brain.name, "embedder": settings.embedder,
            "reranker": settings.reranker,
        }

    @app.post("/v1/users/{user_id}/notes", dependencies=[Depends(authorized)])
    def add_notes(user_id: str, notes: list[Note]) -> dict:
        if user_id == DEMO_USER:
            raise HTTPException(403, "The demo notes are read-only")
        return {"chunks_added": store.for_user(user_id).add(notes)}

    @app.post("/v1/turns", response_model=TurnResult)
    def turn(body: TurnRequest, authorization: str | None = Header(default=None)) -> TurnResult:
        if not allowed(body.user_id, authorization):
            raise HTTPException(401, "Missing or invalid API token")
        return agent.run(body.user_id, body.session_id, body.text)

    @app.post("/v1/transcribe")
    def speech_to_text(audio: UploadFile = File(...)) -> dict:
        data = audio.file.read(MAX_AUDIO_BYTES + 1)
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Audio is larger than 10 MB")
        try:
            return transcribe(data, settings.whisper_model)
        except SpeechUnavailable as error:
            raise HTTPException(503, str(error)) from error

    @app.websocket("/v1/stream")
    async def stream(socket: WebSocket) -> None:
        await socket.accept()
        token = socket.query_params.get("token")
        authorization = f"Bearer {token}" if token else None
        try:
            while True:
                try:
                    body = TurnRequest(**await socket.receive_json())
                except (ValidationError, ValueError, TypeError) as error:
                    await socket.send_json({"type": "error", "detail": str(error)[:200]})
                    continue
                if not allowed(body.user_id, authorization):
                    await socket.send_json({"type": "error", "detail": "Invalid API token"})
                    continue
                async for event in agent.stream(body.user_id, body.session_id, body.text):
                    await socket.send_json(event)
        except WebSocketDisconnect:
            return

    @app.get("/v1/metrics")
    def metrics() -> dict:
        return agent.metrics.summary()

    @app.get("/v1/demo/notes")
    def notes() -> list[dict]:
        return demo_notes

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(ROOT / "web/index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app


app = create_app()
