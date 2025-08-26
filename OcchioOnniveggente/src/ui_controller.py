from __future__ import annotations


from typing import Any

from src.conversation import ConversationManager
from src.ui_state import UIState


class UIController:
    """Simple controller that mediates access to :class:`UIState`.

    Components interacting with the UI can use this controller instead of
    manipulating global variables directly.  Only a subset of behaviour is
    required for the tests, so the class intentionally keeps a very small
    surface area.
    """

    def __init__(self, state: UIState | None = None) -> None:
        self.state = state or UIState()

    # ------------------------------------------------------------------
    # configuration
    @property
    def settings(self) -> dict[str, Any]:
        return self.state.settings

    def update_settings(self, new_settings: dict[str, Any]) -> None:
        self.state.settings = new_settings

    # ------------------------------------------------------------------
    # conversation
    @property
    def conversation(self) -> ConversationManager | None:
        return self.state.conversation

    def set_conversation(self, conv: ConversationManager) -> None:
        self.state.conversation = conv

    # ------------------------------------------------------------------
    # audio reference
    @property
    def audio(self) -> Any | None:
        return self.state.audio

    def set_audio(self, audio: Any | None) -> None:
        self.state.audio = audio

import copy
import re
from pathlib import Path
from typing import Any, Callable

import yaml
from openai import AsyncOpenAI

from src.chat import ChatState
from src.conversation import ConversationManager
from src.config import get_openai_api_key
from src.domain import validate_question
from src.oracle import oracle_answer
from src.retrieval import retrieve


_REASON_RE = re.compile(
    r"kw=(?P<kw>[-0-9.]+) emb=(?P<emb>[-0-9.]+) rag=(?P<rag>[-0-9.]+) score=(?P<score>[-0-9.]+) thr=(?P<thr>[-0-9.]+)"
)


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def deep_copy(d: dict) -> dict:
    return copy.deepcopy(d or {})


def load_settings_pair(root: Path) -> tuple[dict, dict, dict]:
    """Return base, local and merged settings dictionaries."""
    base_p = root / "settings.yaml"
    local_p = root / "settings.local.yaml"

    base = {}
    local = {}
    if base_p.exists():
        base = yaml.safe_load(base_p.read_text(encoding="utf-8")) or {}
    if local_p.exists():
        local = yaml.safe_load(local_p.read_text(encoding="utf-8")) or {}

    merged = deep_copy(base)
    merged = deep_update(merged, deep_copy(local))
    return base, local, merged


def routed_save(
    base_now: dict,
    local_now: dict,
    merged_new: dict,
    root: Path,
    log_fn: Callable[[str, str], None] | None = None,
) -> None:
    """Split settings saving between base and local files."""
    local_out = deep_copy(local_now)
    local_out.setdefault("audio", {})
    local_out["debug"] = bool(merged_new.get("debug", local_out.get("debug", False)))

    audio_new = deep_copy(merged_new.get("audio", {}))
    if "input_device" in audio_new:
        local_out["audio"]["input_device"] = audio_new.get(
            "input_device", local_out["audio"].get("input_device")
        )
    if "output_device" in audio_new:
        local_out["audio"]["output_device"] = audio_new.get(
            "output_device", local_out["audio"].get("output_device")
        )

    base_out = deep_copy(merged_new)
    base_out.pop("debug", None)
    if "audio" in base_out:
        base_out["audio"].pop("input_device", None)
        base_out["audio"].pop("output_device", None)

    base_changed = base_out != base_now
    local_changed = local_out != local_now

    (root / "settings.yaml").write_text(
        yaml.safe_dump(base_out, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    if base_changed and log_fn is not None:
        log_fn("Aggiornato settings.yaml\n", "MISC")
    (root / "settings.local.yaml").write_text(
        yaml.safe_dump(local_out, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    if local_changed and log_fn is not None:
        log_fn("Aggiornato settings.local.yaml\n", "MISC")


class UiController:
    """Business logic for the Tkinter UI."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.reload_settings()

    def reload_settings(self) -> None:
        self.base_settings, self.local_settings, self.settings = load_settings_pair(self.root_dir)
        chat_conf = self.settings.get("chat", {})
        self.conv = ConversationManager()
        self.chat_state: ChatState = self.conv.chat
        if chat_conf.get("enabled", True):
            self.chat_state.max_turns = int(chat_conf.get("max_turns", 10))
        else:
            self.chat_state.max_turns = 0
        pj = chat_conf.get("persist_jsonl")
        if pj:
            self.chat_state.persist_jsonl = Path(pj)

    def save_settings(self, log_fn: Callable[[str, str], None] | None = None) -> None:
        routed_save(
            self.base_settings,
            self.local_settings,
            self.settings,
            self.root_dir,
            log_fn,
        )
        # refresh references
        self.base_settings, self.local_settings, self.settings = load_settings_pair(self.root_dir)

    def send_chat(
        self,
        text: str,
        lang: str,
        mode: str,
        style_enabled: bool,
        sandbox: bool,
    ) -> tuple[str, list[dict[str, str]]]:
        self.conv.push_user(text)
        openai_conf = self.settings.get("openai", {})
        api_key = get_openai_api_key(self.settings)
        client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        style_prompt = self.settings.get("style_prompt", "") if style_enabled else ""
        docstore_path = self.settings.get("docstore_path")
        top_k = int(self.settings.get("retrieval_top_k", 3))
        ok, ctx, needs_clar, reason, _ = validate_question(
            text,
            settings=self.settings,
            client=client,
            docstore_path=docstore_path,
            top_k=top_k,
            embed_model=openai_conf.get("embed_model", "text-embedding-3-small"),
            topic=self.chat_state.topic_text,
            history=self.chat_state.history,
        )
        if not ok:
            if needs_clar:
                ans = "Potresti fornire maggiori dettagli o chiarire la tua domanda?"
            else:
                m = _REASON_RE.search(reason)
                if m:
                    score = float(m.group("score"))
                    thr = float(m.group("thr"))
                    ans = f"Richiesta fuori dominio (score {score:.2f} < {thr:.2f})."
                else:
                    ans = "Richiesta fuori dominio."
            self.conv.push_assistant(ans)
            return ans, []
        if sandbox:
            ctx = []
        else:
            dom = self.settings.get("domain", {}) or {}
            prof = dom.get("profile", "")
            if isinstance(prof, dict):
                prof = prof.get("current", "")
            ctx = retrieve(
                text,
                self.settings.get("docstore_path", ""),
                top_k=top_k,
                topic=prof,
            )
        pin_ctx = [{"id": f"pin{i}", "text": t} for i, t in enumerate(self.chat_state.pinned)]
        ctx = pin_ctx + ctx
        tone = self.settings.get("tone", "informal")
        ans, used_ctx = oracle_answer(
            text,
            lang,
            client,
            self.settings.get("llm_model", "gpt-4o"),
            style_prompt,
            tone=tone,
            context=ctx,
            history=self.chat_state.history,
            topic=self.chat_state.topic_text,
            mode=mode,
        )
        self.conv.push_assistant(ans)
        return ans, used_ctx

