"""
Thin wrapper so the rest of the code doesn't care which provider is behind the
LLM. Also a json helper, because the planner and reviewer want structured back
and models sometimes wrap it in prose or fences.
"""
import json
import re

from config import LLM_PROVIDER, ANTHROPIC_MODEL, OPENAI_MODEL

_llm = None


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        _llm = ChatAnthropic(model=ANTHROPIC_MODEL,  max_tokens=1024)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    else:
        raise ValueError(f"unknown LLM_PROVIDER: {LLM_PROVIDER}")
    return _llm


def ask(system: str, user: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def ask_json(system: str, user: str, fallback):
    """Invoke and parse JSON. If the model gives us something unparseable we
    return the caller's fallback instead of blowing up the whole case - a
    degraded plan is better than a crash in front of an analyst."""
    raw = ask(system + "\nReturn ONLY valid JSON, no prose, no code fences.", user)
    text = raw.strip()
    # strip ```json ... ``` if the model added it anyway
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        # last-ditch: grab the first {...} or [...] block
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return fallback
