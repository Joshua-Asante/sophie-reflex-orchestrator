"""
One-file adapter registry for all SOPHIE-Core tools.

Each tool name maps to its async `execute(parameters: dict) -> str`.
Add new tools simply by extending the `REGISTRY` dict.
"""

import asyncio
import json
import httpx
from pathlib import Path
from typing import Any, Dict

# ---------- helpers ----------
def _error(msg: str) -> str:
    return f"[{msg}]"


# ---------- tool implementations ----------
async def web_search(parameters: Dict[str, Any]) -> str:
    import googlesearch
    urls = list(
        googlesearch.search(parameters["query"], num_results=parameters.get("num_results", 5))
    )
    return json.dumps(urls, ensure_ascii=False)


async def generative_ai(parameters: Dict[str, Any]) -> str:
    provider = parameters.get("provider", "openai").lower()
    prompt = parameters["prompt"]
    api_key = parameters["api_key"]
    async with httpx.AsyncClient(timeout=30) as client:
        if provider == "gemini":
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
                f"?key={api_key}"
            )
            body = {"contents": [{"parts": [{"text": prompt}]}]}
            r = await client.post(url, json=body)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:  # openai
            body = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


async def file_tools_write(parameters: Dict[str, Any]) -> str:
    try:
        path = Path(parameters["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(parameters["content"], encoding="utf-8")
        return f"Wrote {len(parameters['content'])} chars to {path}"
    except Exception as e:
        return _error(f"file_tools_write: {e}")


async def file_tools_read(parameters: Dict[str, Any]) -> str:
    try:
        return Path(parameters["path"]).read_text(encoding="utf-8")
    except Exception as e:
        return _error(f"file_tools_read: {e}")


async def file_tools_list_dir(parameters: Dict[str, Any]) -> str:
    import json
    try:
        items = [p.name for p in Path(parameters["path"]).iterdir()]
        return json.dumps(items)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def human(parameters: Dict[str, Any]) -> str:
    return input(f"\n🤔 {parameters['message']}\n> ")


# ---------- registry ----------
REGISTRY: Dict[str, Any] = {
    "web_search": web_search,
    "generative_ai": generative_ai,
    "file_tools_write": file_tools_write,
    "file_tools_read": file_tools_read,
    "file_tools_list_dir": file_tools_list_dir,
    "human": human,
    # add more tools here—no extra files required
}


# ---------- public interface ----------
async def execute(tool_name: str, parameters: Dict[str, Any]) -> Any:
    """
    Entry point used by the generic adapter.
    """
    if tool_name not in REGISTRY:
        return _error(f"tool '{tool_name}' not defined")
    return await REGISTRY[tool_name](parameters)