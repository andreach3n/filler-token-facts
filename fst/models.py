"""Study models and the two ways we call them.

- Opus 4.5 goes through the Anthropic Message Batches API (50% cheaper; nothing needs to be live).
- Open-weight models go through OpenRouter's OpenAI-compatible endpoint, with reasoning disabled
  and hosts pinned to full-precision weights. Each response records which host served it.

Credentials: ANTHROPIC_API_KEY and OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import fst  # noqa: F401  (sets SSL_CERT_FILE)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str  # "anthropic" or "openrouter"
    model_id: str
    tokenizer: str | None  # key in fst.tokenization.TOKENIZER_REPOS; None = Anthropic count_tokens
    extra_body: dict = field(default_factory=dict)


MODELS = {
    spec.name: spec
    for spec in [
        ModelSpec("opus-4.5", "anthropic", "claude-opus-4-5", tokenizer=None),
        # "DeepSeek V4 Flash 0423" on OpenRouter: the original checkpoint the J-lens was fitted on.
        ModelSpec(
            "deepseek-v4-flash", "openrouter", "deepseek/deepseek-v4-flash", "deepseek-v4-flash",
            extra_body={"reasoning": {"enabled": False}, "provider": {"quantizations": ["fp8"]}},
        ),
        ModelSpec(
            "deepseek-v3-0324", "openrouter", "deepseek/deepseek-chat-v3-0324", "deepseek-v3-0324",
            extra_body={"provider": {"quantizations": ["fp8"]}},
        ),
        ModelSpec(
            "qwen3.6-27b", "openrouter", "qwen/qwen3.6-27b", "qwen3.6-27b",
            extra_body={"reasoning": {"enabled": False}},
        ),
    ]
}

_INT = re.compile(r"-?\d+")


def parse_int_answer(text: str) -> int | None:
    """First integer in the reply, tolerating an echoed 'Answer:' or surrounding whitespace."""
    match = _INT.search(text.replace(",", ""))
    return int(match.group()) if match else None


# --- Anthropic --------------------------------------------------------------------------

def anthropic_client():
    import anthropic

    return anthropic.Anthropic()


def anthropic_count_after(model_id: str):
    """count_after(prefix, text) for a Claude model, via the token counting endpoint (cached)."""
    client = anthropic_client()
    cache: dict[str, int] = {}

    def count(content: str) -> int:
        if content not in cache:
            cache[content] = client.messages.count_tokens(
                model=model_id, messages=[{"role": "user", "content": content}]
            ).input_tokens
        return cache[content]

    return lambda prefix, text: count(prefix + text) - count(prefix)


def run_anthropic_batch(spec: ModelSpec, requests: dict[str, dict], max_tokens: int, state_path: Path) -> dict[str, dict]:
    """Submit (or resume) one Message Batch and return {request_id: result}.

    `requests` maps our ids (<=64 chars, [A-Za-z0-9_-]) to provider-neutral prompts.
    The batch id is saved to `state_path` so an interrupted run picks the same batch back up.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic_client()
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if "batch_id" not in state:
        batch = client.messages.batches.create(requests=[
            Request(
                custom_id=request_id,
                params=MessageCreateParamsNonStreaming(
                    model=spec.model_id,
                    max_tokens=max_tokens,
                    temperature=0,
                    system=prompt["system"],
                    messages=prompt["messages"],
                ),
            )
            for request_id, prompt in requests.items()
        ])
        state = {"batch_id": batch.id, "created": time.time()}
        state_path.write_text(json.dumps(state))
        print(f"submitted batch {batch.id} with {len(requests)} requests")

    while True:
        batch = client.messages.batches.retrieve(state["batch_id"])
        if batch.processing_status == "ended":
            break
        print(f"batch {batch.id}: {batch.processing_status}, {batch.request_counts.processing} processing")
        time.sleep(60)

    results = {}
    for item in client.messages.batches.results(state["batch_id"]):
        if item.result.type == "succeeded":
            message = item.result.message
            text = next((b.text for b in message.content if b.type == "text"), "")
            results[item.custom_id] = {
                "text": text,
                "stop_reason": message.stop_reason,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
        else:
            results[item.custom_id] = {"error": item.result.type}
    return results


# --- OpenRouter -------------------------------------------------------------------------

async def _openrouter_one(client, spec: ModelSpec, prompt: dict, max_tokens: int) -> dict:
    messages = [{"role": "system", "content": prompt["system"]}, *prompt["messages"]]
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=spec.model_id, messages=messages, max_tokens=max_tokens, temperature=0,
                extra_body=spec.extra_body,
            )
            choice = response.choices[0]
            raw = response.model_dump()
            return {
                "text": choice.message.content or "",
                "stop_reason": choice.finish_reason,
                "host": raw.get("provider"),
                "input_tokens": response.usage.prompt_tokens if response.usage else None,
                "output_tokens": response.usage.completion_tokens if response.usage else None,
                "reasoning_returned": bool(raw["choices"][0]["message"].get("reasoning")),
            }
        except Exception as exc:  # retry transient failures, record the last one
            error = repr(exc)
            await asyncio.sleep(2 ** attempt)
    return {"error": error}


def run_openrouter(spec: ModelSpec, requests: dict[str, dict], max_tokens: int, out_path: Path, concurrency: int = 16) -> dict[str, dict]:
    """Run requests with bounded concurrency, appending each result to `out_path` as it lands.

    Ids already present in `out_path` are skipped, so rerunning resumes.
    """
    from openai import AsyncOpenAI

    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            if "error" not in row["result"]:
                done[row["id"]] = row["result"]
    pending = {k: v for k, v in requests.items() if k not in done}
    print(f"{spec.name}: {len(done)} done, {len(pending)} to run")

    async def main():
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        semaphore = asyncio.Semaphore(concurrency)
        with out_path.open("a") as f:
            async def worker(request_id, prompt):
                async with semaphore:
                    result = await _openrouter_one(client, spec, prompt, max_tokens)
                f.write(json.dumps({"id": request_id, "result": result}) + "\n")
                f.flush()
                done[request_id] = result

            await asyncio.gather(*(worker(k, v) for k, v in pending.items()))

    if pending:
        asyncio.run(main())
    return done


def run_requests(spec: ModelSpec, requests: dict[str, dict], max_tokens: int, out_dir: Path) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if spec.provider == "anthropic":
        results = run_anthropic_batch(spec, requests, max_tokens, out_dir / f"{spec.name}.batch.json")
        (out_dir / f"{spec.name}.jsonl").write_text(
            "".join(json.dumps({"id": k, "result": v}) + "\n" for k, v in results.items())
        )
        return results
    return run_openrouter(spec, requests, max_tokens, out_dir / f"{spec.name}.jsonl")
