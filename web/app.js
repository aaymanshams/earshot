const SCRIPT = [
  "Good morning, how are you?",
  "Are you allergic to any medicines?",
  "How many units of insulin do you take at night?",
  "Do you have a pet?",
  "Um, basically you should take 2 tablets at 8:30 pm, you know, prior to dinner.",
  "There's a fire, get out now!",
];
const STEPS = ["triage", "urgent", "retrieve", "draft", "verify", "simplify", "finish"];
const $ = (id) => document.getElementById(id);
const session = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
let socket, hudTimer;

function buildSteps() {
  $("steps").replaceChildren(...STEPS.map((step) => {
    const item = document.createElement("li");
    item.textContent = step;
    item.dataset.step = step;
    return item;
  }));
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/v1/stream`);
  socket.onmessage = (message) => handle(JSON.parse(message.data));
  socket.onclose = () => setTimeout(connect, 1000);
}

function send(text) {
  if (!text.trim() || socket.readyState !== WebSocket.OPEN) return;
  buildSteps();
  clearTimeout(hudTimer);
  $("view").classList.remove("alert");
  $("hud").textContent = "...";
  $("dots").textContent = "";
  socket.send(JSON.stringify({ user_id: "demo", session_id: session, text }));
}

function handle(event) {
  if (event.type === "node") {
    document.querySelector(`[data-step="${event.node}"]`)?.classList.add("on");
  } else if (event.type === "result") {
    render(event.result);
  } else if (event.type === "error") {
    $("hud").textContent = "Could not read that";
  }
}

function showPages(pages, index = 0) {
  $("hud").textContent = pages[index].lines.join("\n");
  $("dots").textContent = pages.length > 1 ? `Page ${index + 1} of ${pages.length}` : "";
  if (index + 1 < pages.length) {
    hudTimer = setTimeout(() => showPages(pages, index + 1), pages[index].seconds * 1000);
  }
}

function render(turn) {
  $("view").classList.toggle("alert", turn.intent === "urgent");
  showPages(turn.hud);
  $("source").textContent = "";
  $("replies").replaceChildren(...turn.suggestions.map((reply) => {
    const button = document.createElement("button");
    button.textContent = reply.text;
    button.className = reply.kind;
    button.onclick = () => speak(reply.text);
    return button;
  }));
  const grounded = turn.suggestions.find((reply) => reply.kind === "grounded");
  if (grounded) $("source").textContent = "Checked against your notes.";
  else if (turn.suggestions[0].kind === "fallback") $("source").textContent = "Not in your notes, so no answer was suggested.";
  const rows = turn.trace.map((step) => [step.node, `${step.ms.toFixed(1)} ms`]);
  rows.push(["Total", `${turn.total_ms.toFixed(1)} ms`]);
  $("trace").replaceChildren(...rows.map(([name, value]) => {
    const row = document.createElement("tr");
    row.insertCell().textContent = name;
    row.insertCell().textContent = value;
    return row;
  }));
}

function speak(text) {
  if ("speechSynthesis" in window) speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  $("source").textContent = `Spoken: "${text}"`;
}

async function loadExtras() {
  const health = await (await fetch("/health")).json();
  const brain = health.brain === "llm" ? "LLM" : "Offline";
  const search = health.embedder === "hashing" ? "baseline search, may give wrong facts" : "semantic search";
  $("mode").textContent = `${brain}, ${search}`;
  const notes = await (await fetch("/v1/demo/notes")).json();
  $("notes").replaceChildren(...notes.flatMap((note) => {
    const title = document.createElement("h3");
    title.textContent = note.title;
    const body = document.createElement("p");
    body.textContent = note.text;
    return [title, body];
  }));
}

let recorder;
async function toggleRecording() {
  if (recorder?.state === "recording") return recorder.stop();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    $("micnote").textContent = "Microphone not available. Type instead.";
    return;
  }
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (event) => chunks.push(event.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((track) => track.stop());
    $("mic").classList.remove("recording");
    $("mic").textContent = "Record";
    $("micnote").textContent = "Transcribing...";
    const form = new FormData();
    form.append("audio", new Blob(chunks), "speech.webm");
    const response = await fetch("/v1/transcribe", { method: "POST", body: form });
    if (!response.ok) { $("micnote").textContent = "Speech is not enabled on this server."; return; }
    const { text } = await response.json();
    $("micnote").textContent = text ? `Heard: ${text}` : "No speech heard.";
    if (text) send(text);
  };
  recorder.start();
  $("mic").classList.add("recording");
  $("mic").textContent = "Stop";
}

$("script").replaceChildren(...SCRIPT.map((line) => {
  const button = document.createElement("button");
  button.textContent = line;
  button.onclick = () => send(line);
  return button;
}));
$("say").onsubmit = (event) => {
  event.preventDefault();
  send($("text").value);
  $("text").value = "";
};
$("mic").onclick = toggleRecording;
buildSteps();
connect();
loadExtras();
