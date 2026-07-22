#!/usr/bin/env python3
"""
anthropic_tabllm.py — thin Anthropic-API client for the in-context TabLLM
classifier (serialize-a-grid-cell -> classify HIGH/LOW dwelling).

Design goals (see plan jiggly-nibbling-hopcroft):
  * structured output {label, p_high} via output_config json_schema (numeric
    min/max are NOT enforced by the API -> clamp p_high client-side);
  * a sqlite RESPONSE CACHE keyed by sha256(model | system | messages |
    gen-params | vote_idx | schema) so re-runs are free and metric
    reproducibility is anchored to the committed cache, NOT to a seed
    (Claude is not byte-deterministic even at temperature 0);
  * an N-vote ensemble (Haiku only — it accepts `temperature`; Opus 4.8 rejects
    temperature/top_p/top_k with a 400, so it is single-call / spot-check only);
  * Batch API support (50% cheaper, results UNORDERED -> keyed by custom_id).

The `anthropic` SDK is imported lazily inside the methods that call it, so this
module (and the serialization / evaluation code that imports it) can be imported
and unit-tested with numpy + stdlib only, with no SDK and no API key.

Model IDs (verified against the claude-api skill):
  Haiku 4.5 = "claude-haiku-4-5"  ($1/$5 per MTok, 200K ctx, accepts temperature)
  Opus 4.8  = "claude-opus-4-8"   ($5/$25, rejects temperature/top_p/top_k)
"""

import hashlib
import json
import os
import sqlite3
import time

HAIKU = "claude-haiku-4-5"
OPUS = "claude-opus-4-8"

# Opus 4.8 / 4.7 / Fable 5 reject sampling params (400). Only Haiku here accepts them.
_ACCEPTS_TEMPERATURE = {HAIKU}

# Structured-output schema. `p_high` carries no minimum/maximum: the API does not
# enforce numeric bounds, so we clamp client-side. `reasoning` is optional and
# dropped in the bulk pass (max_tokens small) to save output tokens.
OUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "p_high"],
    "properties": {
        "reasoning": {"type": "string"},
        "label": {"type": "string", "enum": ["high", "low"]},
        "p_high": {"type": "number"},
    },
}
_SCHEMA_HASH = hashlib.sha256(
    json.dumps(OUT_SCHEMA, sort_keys=True).encode()
).hexdigest()[:16]


def _clamp01(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return min(max(x, 0.0), 1.0)


def _label_to_p(label):
    """Fallback score when p_high is missing/garbage: map the label to {0,1}."""
    return 1.0 if str(label).strip().lower() == "high" else 0.0


class TabLLMClient:
    """Caching wrapper around client.messages.create / .batches for the
    serialize-then-classify TabLLM workload.

    Parameters
    ----------
    model : str
        HAIKU (default) or OPUS.
    cache_path : str
        Path to the sqlite response cache. Created if absent. COMMIT THIS FILE —
        it is the reproducibility anchor (API output is not byte-deterministic).
    n_vote : int
        Number of samples to draw per cell (Haiku only). 1 = single deterministic
        call. >1 raises `temperature` and aggregates (vote_freq + mean p_high).
    temperature : float
        Sampling temperature used when n_vote > 1 (Haiku only). Ignored for Opus.
    max_tokens : int
        Output cap. 128 for the bulk pass (reasoning dropped), 512 for spot-checks.
    keep_reasoning : bool
        If True, request the optional `reasoning` field (Opus spot-check).
    """

    def __init__(self, model=HAIKU, cache_path="figures/tabllm_cache.sqlite",
                 n_vote=1, temperature=0.0, max_tokens=128, keep_reasoning=False,
                 max_retries=4, base_url=None, api_key=None, error_abort=25):
        self.model = model
        self.cache_path = cache_path
        self.n_vote = max(1, int(n_vote))
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.keep_reasoning = bool(keep_reasoning)
        self.max_retries = int(max_retries)
        # Circuit-breaker: abort a batched run once this many requests have
        # failed (server likely down), so a dead vLLM server can't quietly
        # poison the cache with neutral 0.0s. <=0 disables. See
        # _classify_batch_openai.
        self.error_abort = int(error_abort)
        # OpenAI-compatible backend (e.g. a self-hosted vLLM server exposing
        # /v1/chat/completions). When base_url is set we route through the
        # `openai` SDK instead of `anthropic`, and the Anthropic path is left
        # byte-identical. Falls back to $VLLM_BASE_URL so it can be exported.
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.is_openai = bool(self.base_url)
        self._client = None  # lazy
        self._db = None       # lazy
        self.calls_made = 0   # live API calls this process (cache misses)
        self.cache_hits = 0

    # ----- lazy resources -------------------------------------------------
    def _client_obj(self):
        if self._client is None:
            if self.is_openai:
                import openai  # lazy: only needed for a real vLLM call
                self._client = openai.OpenAI(
                    base_url=self.base_url, api_key=self.api_key,
                    max_retries=self.max_retries,
                )
            else:
                import anthropic  # lazy: only needed for a real call
                self._client = anthropic.Anthropic(max_retries=self.max_retries)
        return self._client

    def _db_conn(self):
        if self._db is None:
            d = os.path.dirname(self.cache_path)
            if d:
                os.makedirs(d, exist_ok=True)
            self._db = sqlite3.connect(self.cache_path)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)"
            )
            self._db.commit()
        return self._db

    # ----- cache key ------------------------------------------------------
    def cache_key(self, system, messages, temperature, vote_idx):
        """sha256 over everything that can change the response. vote_idx keeps the
        N samples of one cell distinct (else identical custom_id / cache key)."""
        payload = json.dumps(
            {
                "model": self.model,
                "system": system,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "keep_reasoning": self.keep_reasoning,
                "vote_idx": vote_idx,
                "schema": _SCHEMA_HASH,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key):
        row = self._db_conn().execute(
            "SELECT v FROM cache WHERE k=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, key, value):
        db = self._db_conn()
        db.execute(
            "INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        db.commit()

    # ----- request construction ------------------------------------------
    def _create_kwargs(self, system, messages, temperature):
        kw = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"format": {"type": "json_schema", "schema": OUT_SCHEMA}},
        )
        if self.model in _ACCEPTS_TEMPERATURE:
            kw["temperature"] = temperature
        return kw

    def _create_kwargs_openai(self, system, messages, temperature):
        """OpenAI/vLLM chat-completions request. The system prompt is prepended
        as a message (no top-level `system` kwarg), and structured output uses
        `response_format` json_schema (vLLM guided decoding). vLLM accepts
        `temperature` for every model, so it is always sent."""
        return dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}] + list(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "dwell", "schema": OUT_SCHEMA,
                                "strict": True},
            },
        )

    # ----- backend-agnostic single inference -----------------------------
    def _infer_one(self, system, messages, temperature):
        """One live model call -> parsed {label, p_high, raw}. Branches on the
        backend; both callers (classify_one, batch fan-out) go through here."""
        if self.is_openai:
            kw = self._create_kwargs_openai(system, messages, temperature)
            resp = self._client_obj().chat.completions.create(**kw)
            return self._parse_openai(resp)
        kw = self._create_kwargs(system, messages, temperature)
        resp = self._client_obj().messages.create(**kw)
        return self._parse(resp)

    def _parse_openai(self, resp):
        """Normalize an OpenAI/vLLM chat completion into the block shape `_parse`
        expects (choices[0].message.content is a JSON string)."""
        try:
            text = resp.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            text = None
        return self._parse({"content": [{"type": "text", "text": text}]})

    @staticmethod
    def _parse(message_obj_or_dict):
        """Extract {label, p_high} from a Message (object or dict). Structured
        output guarantees the first text block is valid JSON for the schema."""
        # Accept both SDK objects and the dict shape we cache.
        content = getattr(message_obj_or_dict, "content", None)
        if content is None and isinstance(message_obj_or_dict, dict):
            content = message_obj_or_dict.get("content")
        text = None
        for block in content or []:
            btype = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if btype == "text":
                text = getattr(block, "text", None) or block.get("text")
                break
        if text is None:
            return {"label": "low", "p_high": 0.0, "raw": None}
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return {"label": "low", "p_high": 0.0, "raw": text}
        label = str(obj.get("label", "low")).strip().lower()
        p = _clamp01(obj.get("p_high"))
        if p is None:
            p = _label_to_p(label)
        return {"label": label, "p_high": p, "raw": obj}

    # ----- single-cell classification (synchronous) ----------------------
    def classify_one(self, system, messages):
        """Classify one cell. With n_vote>1 (Haiku) draws N samples and returns
        an aggregated continuous score. Returns dict:
            {label, p_high (=score), p_mean, vote_freq, n_used, cached}
        """
        temps = ([0.0] if self.n_vote == 1
                 else [self.temperature] * self.n_vote)
        votes = []
        any_live = False
        for vi, t in enumerate(temps):
            key = self.cache_key(system, messages, t, vi)
            cached = self._cache_get(key)
            if cached is None:
                parsed = self._infer_one(system, messages, t)
                self._cache_put(key, parsed)
                self.calls_made += 1
                any_live = True
            else:
                parsed = cached
                self.cache_hits += 1
            votes.append(parsed)
        labels_high = [1.0 if v["label"] == "high" else 0.0 for v in votes]
        p_means = [v["p_high"] for v in votes]
        vote_freq = sum(labels_high) / len(votes)
        p_mean = sum(p_means) / len(votes)
        # single deterministic call -> use p_high directly; ensemble -> blend
        score = p_mean if self.n_vote == 1 else 0.5 * (p_mean + vote_freq)
        majority = "high" if vote_freq >= 0.5 else "low"
        return {
            "label": majority,
            "p_high": score,
            "p_mean": p_mean,
            "vote_freq": vote_freq,
            "n_used": len(votes),
            "cached": not any_live,
        }

    # ----- batch classification ------------------------------------------
    def classify_batch(self, jobs, poll_seconds=30, max_wait=7200):
        """Classify many cells via the Batch API (50% cheaper, ~1h typical).

        `jobs` : list of dicts {id, system, messages}. Returns {job_id: result}
        with the same aggregation as classify_one. Cached votes are served
        locally; only misses are submitted. Results arrive UNORDERED -> keyed
        by custom_id (we encode job_id + vote_idx into it).

        For the OpenAI/vLLM backend there is no Batch API; we fan out concurrent
        chat-completions instead (vLLM's continuous batching turns them into GPU
        batches) -> `_classify_batch_openai`.
        """
        if self.is_openai:
            return self._classify_batch_openai(jobs)
        from anthropic.types.message_create_params import (
            MessageCreateParamsNonStreaming,
        )
        from anthropic.types.messages.batch_create_params import Request

        temps_for = (lambda: [0.0]) if self.n_vote == 1 else (
            lambda: [self.temperature] * self.n_vote
        )
        # plan each (job, vote) unit; serve cache hits, collect misses
        votes = {j["id"]: [None] * self.n_vote for j in jobs}
        requests = []
        custom_map = {}  # custom_id -> (job_id, vote_idx, cache_key)
        for j in jobs:
            for vi, t in enumerate(temps_for()):
                key = self.cache_key(j["system"], j["messages"], t, vi)
                cached = self._cache_get(key)
                if cached is not None:
                    votes[j["id"]][vi] = cached
                    self.cache_hits += 1
                    continue
                cid = f"{j['id']}::{vi}"
                custom_map[cid] = (j["id"], vi, key)
                requests.append(
                    Request(
                        custom_id=cid,
                        params=MessageCreateParamsNonStreaming(
                            **self._create_kwargs(j["system"], j["messages"], t)
                        ),
                    )
                )

        if requests:
            client = self._client_obj()
            batch = client.messages.batches.create(requests=requests)
            waited = 0
            while True:
                b = client.messages.batches.retrieve(batch.id)
                if b.processing_status == "ended":
                    break
                if waited >= max_wait:
                    raise TimeoutError(f"batch {batch.id} not done after {waited}s")
                time.sleep(poll_seconds)
                waited += poll_seconds
            for result in client.messages.batches.results(batch.id):
                cid = result.custom_id
                job_id, vi, key = custom_map[cid]
                if result.result.type == "succeeded":
                    parsed = self._parse(result.result.message)
                else:
                    # errored/expired/canceled -> neutral, flagged
                    parsed = {"label": "low", "p_high": 0.0,
                              "raw": {"batch_error": result.result.type}}
                self._cache_put(key, parsed)
                self.calls_made += 1
                votes[job_id][vi] = parsed

        # aggregate per job (same logic as classify_one)
        out = {}
        for jid, vlist in votes.items():
            vlist = [v for v in vlist if v is not None]
            if not vlist:
                out[jid] = {"label": "low", "p_high": 0.0, "n_used": 0}
                continue
            labels_high = [1.0 if v["label"] == "high" else 0.0 for v in vlist]
            p_means = [v["p_high"] for v in vlist]
            vote_freq = sum(labels_high) / len(vlist)
            p_mean = sum(p_means) / len(vlist)
            score = p_mean if self.n_vote == 1 else 0.5 * (p_mean + vote_freq)
            out[jid] = {
                "label": "high" if vote_freq >= 0.5 else "low",
                "p_high": score, "p_mean": p_mean,
                "vote_freq": vote_freq, "n_used": len(vlist),
            }
        return out

    def _classify_batch_openai(self, jobs, max_workers=32):
        """vLLM/OpenAI batch: serve cache hits, then fan out the misses over a
        thread pool (vLLM batches concurrent requests server-side). sqlite is
        touched only from the main thread (get before the pool, put after)."""
        from concurrent.futures import ThreadPoolExecutor

        temps = [0.0] if self.n_vote == 1 else [self.temperature] * self.n_vote
        votes = {j["id"]: [None] * self.n_vote for j in jobs}
        misses = []  # (job_id, vote_idx, cache_key, system, messages, temperature)
        jobmap = {j["id"]: j for j in jobs}
        for j in jobs:
            for vi, t in enumerate(temps):
                key = self.cache_key(j["system"], j["messages"], t, vi)
                cached = self._cache_get(key)
                if cached is not None:
                    votes[j["id"]][vi] = cached
                    self.cache_hits += 1
                else:
                    misses.append((j["id"], vi, key, j["system"], j["messages"], t))

        if misses:
            def _do(unit):
                jid, vi, key, system, messages, t = unit
                try:
                    return unit, self._infer_one(system, messages, t), None
                except Exception as e:  # degrade one request, don't kill the batch
                    return unit, None, repr(e)

            workers = max(1, min(max_workers, len(misses)))
            # Submit in chunks so the circuit-breaker can abort *early* — with a
            # single ex.map over all misses, every task is scheduled up front and
            # a dead server would grind through the whole grid before we notice.
            chunk = max(workers * 4, workers)
            errors = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for start in range(0, len(misses), chunk):
                    part = misses[start:start + chunk]
                    for (jid, vi, key, _s, _m, _t), parsed, err in ex.map(_do, part):
                        if err is not None:
                            errors += 1
                            # Do NOT cache errors: a re-run should re-attempt
                            # them once the server is healthy again.
                            votes[jid][vi] = {"label": "low", "p_high": 0.0,
                                              "raw": {"vllm_error": err}}
                            continue
                        self._cache_put(key, parsed)  # main thread only
                        self.calls_made += 1
                        votes[jid][vi] = parsed
                    if self.error_abort > 0 and errors >= self.error_abort:
                        done = start + len(part)
                        raise RuntimeError(
                            f"TabLLM circuit-breaker: {errors} of {done} requests "
                            f"to {self.base_url or 'the API'} failed — the server "
                            f"is likely down. Aborting so the cache isn't poisoned "
                            f"with neutral 0.0s. Fix the backend and re-run; cells "
                            f"already answered are cached and will be skipped. "
                            f"(Raise --max-retries or --error-abort to be more "
                            f"tolerant of transient errors.)")

        out = {}
        for jid in jobmap:
            vlist = [v for v in votes[jid] if v is not None]
            if not vlist:
                out[jid] = {"label": "low", "p_high": 0.0, "n_used": 0}
                continue
            labels_high = [1.0 if v["label"] == "high" else 0.0 for v in vlist]
            p_means = [v["p_high"] for v in vlist]
            vote_freq = sum(labels_high) / len(vlist)
            p_mean = sum(p_means) / len(vlist)
            score = p_mean if self.n_vote == 1 else 0.5 * (p_mean + vote_freq)
            out[jid] = {
                "label": "high" if vote_freq >= 0.5 else "low",
                "p_high": score, "p_mean": p_mean,
                "vote_freq": vote_freq, "n_used": len(vlist),
            }
        return out


if __name__ == "__main__":
    # Offline smoke test of the pure-Python pieces (no SDK / no API key).
    c = TabLLMClient(cache_path="/tmp/_tabllm_selftest_cache.sqlite")
    k1 = c.cache_key("sys", [{"role": "user", "content": "x"}], 0.0, 0)
    k2 = c.cache_key("sys", [{"role": "user", "content": "x"}], 0.0, 1)
    k3 = c.cache_key("sys", [{"role": "user", "content": "y"}], 0.0, 0)
    assert k1 != k2 and k1 != k3, "cache keys must distinguish vote_idx and content"
    assert _clamp01(1.5) == 1.0 and _clamp01(-0.2) == 0.0 and _clamp01("nan") is None
    parsed = TabLLMClient._parse(
        {"content": [{"type": "text", "text": '{"label":"high","p_high":1.4}'}]}
    )
    assert parsed["label"] == "high" and parsed["p_high"] == 1.0
    print("anthropic_tabllm self-test OK (cache key, clamp, parse)")
