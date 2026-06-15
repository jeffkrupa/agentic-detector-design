"""Pluggable LLM client.

The orchestrator is model-agnostic: it asks a client to ``propose_action`` given
the current state and the available tools, and to ``write`` free-form text
(reports, rationales). Three backends:

  * ``dry-run``  -- deterministic heuristics, NO API key. Keeps the whole repo
                    runnable offline (CI, demos, plumbing tests).
  * ``anthropic``-- Claude via the official SDK (scaffolded; finish the TODOs).
  * ``openai``   -- GPT via the official SDK (scaffolded; finish the TODOs).

An ``Action`` is a small structured decision the orchestrator executes:
  {"type": "call_tool", "tool": <name>, "args": {...}, "thought": "..."}
  {"type": "finish",     "summary": "..."}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import json
import os


@dataclass
class Action:
    type: str                       # "call_tool" | "finish"
    tool: Optional[str] = None
    args: dict = field(default_factory=dict)
    thought: str = ""
    summary: str = ""


def make_client(spec: str, temperature: float = 0.0, max_tokens: int = 4096):
    """Factory. ``spec`` is 'dry-run' | 'anthropic:<model>' | 'openai:<model>'."""
    if spec == "dry-run":
        return DryRunClient()
    backend, _, model = spec.partition(":")
    if backend == "anthropic":
        return AnthropicClient(model or "claude-sonnet-4", temperature, max_tokens)
    if backend == "openai":
        return OpenAIClient(model or "gpt-4.1", temperature, max_tokens)
    raise ValueError(f"unknown llm spec {spec!r}")


# --------------------------------------------------------------------------- #
# Dry-run (heuristic) client
# --------------------------------------------------------------------------- #
class DryRunClient:
    """Deterministic stand-in. Implements just enough 'reasoning' to exercise the
    full loop: rank sensitivities, step along the largest *reliable* gradient,
    finish when budget is low. The heuristics live in the orchestrator; this
    client only signals intent so the control flow matches the LLM path.
    """
    name = "dry-run"

    def propose_action(self, system_prompt: str, state: dict, tools: list) -> Action:
        # The orchestrator passes a 'phase' hint in state for the heuristic path.
        phase = state.get("phase", "sense")
        if phase == "sense":
            return Action("call_tool", tool="rank_sensitivities",
                          args={"observable": state["target_observable"]},
                          thought="Rank design parameters by reliable sensitivity.")
        if phase == "optimize":
            return Action("call_tool", tool="propose_step",
                          args={"loss_spec": state["loss_spec"]},
                          thought="Take a step along the largest reliable gradient.")
        if phase == "verify":
            return Action("call_tool", tool="cross_check_last_step", args={},
                          thought="Cross-check AD against finite differences.")
        return Action("finish", summary="Budget/looping policy reached; stopping.")

    def write(self, system_prompt: str, content: str) -> str:
        return content


# --------------------------------------------------------------------------- #
# Real backends (scaffolded -- finish the TODOs in CLAUDE.md task #8)
# --------------------------------------------------------------------------- #
class _ApiClient:
    def __init__(self, model, temperature, max_tokens):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"{self.backend}:{model}"

    def _system_with_protocol(self, system_prompt: str, tools: list) -> str:
        schema = json.dumps([t["schema"] for t in tools], indent=2)
        return (
            f"{system_prompt}\n\n"
            "You control a differentiable detector-design loop by emitting ONE JSON "
            "action per turn and nothing else. Allowed actions:\n"
            '  {"type":"call_tool","tool":<name>,"args":{...},"thought":<str>}\n'
            '  {"type":"finish","summary":<str>}\n\n'
            f"Available tools (JSON schema):\n{schema}\n"
        )

    @staticmethod
    def _parse_action(text: str) -> Action:
        text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start:end + 1])
        return Action(type=obj["type"], tool=obj.get("tool"),
                      args=obj.get("args", {}), thought=obj.get("thought", ""),
                      summary=obj.get("summary", ""))


class AnthropicClient(_ApiClient):
    backend = "anthropic"

    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
        # TODO(claude-code): `import anthropic; self._c = anthropic.Anthropic(
        #     api_key=os.environ[...])`. Keep import lazy so dry-run needs no dep.
        self._c = None

    def propose_action(self, system_prompt, state, tools) -> Action:
        # TODO(claude-code): call self._c.messages.create(...) with
        # self._system_with_protocol(system_prompt, tools) as system and a user
        # message = json.dumps(state); then return self._parse_action(text).
        raise NotImplementedError("Wire the Anthropic SDK (see CLAUDE.md task #8).")

    def write(self, system_prompt, content) -> str:
        raise NotImplementedError("Wire the Anthropic SDK (see CLAUDE.md task #8).")


class OpenAIClient(_ApiClient):
    backend = "openai"

    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
        self._c = None  # TODO(claude-code): lazy `from openai import OpenAI`

    def propose_action(self, system_prompt, state, tools) -> Action:
        raise NotImplementedError("Wire the OpenAI SDK (see CLAUDE.md task #8).")

    def write(self, system_prompt, content) -> str:
        raise NotImplementedError("Wire the OpenAI SDK (see CLAUDE.md task #8).")
