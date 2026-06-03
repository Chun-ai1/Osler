"""
llm_client.py — provider-agnostic LLM wrapper for the demo agent.

The LLM has two jobs, both in service of the symbolic engine, never replacing it:
  • parse a free-text case into structured fields (case_parser)
  • explain the engine's reasoning in natural language (chat)

The API key can come from the BROWSER (top-right key box, passed per request) or
from the environment. Per-request key/provider override the environment.

Env defaults:
  LLM_PROVIDER = openai (default) | gemini
  OPENAI_API_KEY / OPENAI_MODEL (default gpt-4o-mini)
  GEMINI_API_KEY / GEMINI_MODEL (default gemini-1.5-flash)

No key / SDK missing -> ok=False so callers fall back to deterministic behavior.
"""
from __future__ import annotations
import os
from typing import List, Dict, Tuple, Optional

_NO_KEY_MSG = (
    "ℹ️ No LLM key configured — chat is in fallback mode. The reasoning graph on the "
    "right is fully computed by the symbolic engine and remains valid.\n\n"
    "To enable natural-language Q&A, paste an API key in the top-right box "
    "(OpenAI by default, or choose Gemini)."
)


def resolve_provider(provider: Optional[str] = None) -> str:
    return (provider or os.environ.get("LLM_PROVIDER") or "openai").strip().lower()


def resolve_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    if api_key:
        return api_key.strip()
    return os.environ.get("GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY")


def available(api_key: Optional[str] = None, provider: Optional[str] = None) -> bool:
    p = resolve_provider(provider)
    return bool(resolve_key(p, api_key))


def chat(system: str, messages: List[Dict[str, str]],
         api_key: Optional[str] = None, provider: Optional[str] = None) -> Tuple[bool, str]:
    """messages: [{role: user|assistant, content}]. Returns (ok, text)."""
    p = resolve_provider(provider)
    key = resolve_key(p, api_key)
    if not key:
        return False, _NO_KEY_MSG
    try:
        if p == "gemini":
            return _gemini(system, messages, key)
        return _openai(system, messages, key)
    except Exception as e:  # noqa: BLE001
        return False, f"⚠️ LLM call failed: {type(e).__name__}: {e}"


def complete_json(system: str, user: str,
                  api_key: Optional[str] = None, provider: Optional[str] = None) -> Tuple[bool, str]:
    """One-shot call used for structured extraction. Returns (ok, raw_text)."""
    return chat(system, [{"role": "user", "content": user}], api_key, provider)


def _openai(system: str, messages: List[Dict[str, str]], key: str) -> Tuple[bool, str]:
    from openai import OpenAI
    client = OpenAI(api_key=key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        temperature=0.2,
    )
    return True, (resp.choices[0].message.content or "").strip()


# Model names change over time; try current ones in order (env override wins).
_GEMINI_MODELS = [m for m in [os.environ.get("GEMINI_MODEL"),
                              "gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash-001"] if m]


def _gemini(system: str, messages: List[Dict[str, str]], key: str) -> Tuple[bool, str]:
    try:  # modern SDK: pip install google-genai
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part(text=m["content"])]) for m in messages]
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=0.2)
        last = None
        for model_name in _GEMINI_MODELS:
            try:
                resp = client.models.generate_content(model=model_name, contents=contents, config=cfg)
                return True, (resp.text or "").strip()
            except Exception as e:  # try next model on 404/not-found, else surface
                last = e
                if "NOT_FOUND" in str(e) or "not found" in str(e).lower() or "404" in str(e):
                    continue
                raise
        raise last or RuntimeError("no Gemini model available")
    except ImportError:  # fall back to deprecated google-generativeai
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(_GEMINI_MODELS[0], system_instruction=system)
        history = [{"role": "model" if m["role"] == "assistant" else "user",
                    "parts": [m["content"]]} for m in messages]
        return True, (genai.GenerativeModel and model.generate_content(history).text or "").strip()
