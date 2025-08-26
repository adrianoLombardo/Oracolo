from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify

from .service_container import container
from .oracle import oracle_answer, transcribe
from .metrics import metrics_endpoint, health_endpoint


def create_app() -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__, template_folder="templates", static_folder="static")

    conv = container.conversation_manager
    queue = container.message_queue

    @app.get("/docs")
    def docs() -> str:
        """Render the documentation page."""
        return render_template("docs.html")

    @app.post("/chat")
    def chat_endpoint() -> "flask.Response":
        data = request.get_json(force=True) or {}
        message = data.get("message", "")
        if message:
            asyncio.run(queue.put(message))
            answer, _ = oracle_answer(message, "it", conv=conv)
        else:
            answer = ""
        return jsonify({"response": answer})

    @app.post("/voice")
    def voice_endpoint() -> "flask.Response":
        if "audio" not in request.files:
            return jsonify({"error": "missing audio"}), 400
        file = request.files["audio"]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        file.save(tmp.name)
        text = asyncio.run(transcribe(Path(tmp.name), conv=conv))
        asyncio.run(queue.put(text))
        answer, _ = oracle_answer(text, "it", conv=conv)
        return jsonify({"transcript": text, "response": answer})

    @app.get("/logs")
    def logs_endpoint() -> "flask.Response":
        return jsonify(conv.messages_for_llm() if conv else [])

    @app.get("/metrics")
    def metrics() -> "flask.Response":
        return metrics_endpoint()

    @app.get("/healthz")
    def health() -> "flask.Response":
        return health_endpoint()

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    app.run(debug=True)
