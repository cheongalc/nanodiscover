from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.renderer import resolve_renderer


logger = logging.getLogger(__name__)

RAY_LOCAL_SOCKET_PATH_LIMIT = 107
RAY_LOCAL_TEMP_DIR_NAME = "ray"
RAY_LOCAL_SESSION_SOCKET_SUFFIX = (
    "/session_0000-00-00_00-00-00_000000_0000000/sockets/plasma_store"
)
RAY_LOCAL_TEMP_DIR_MAX_LENGTH = (
    RAY_LOCAL_SOCKET_PATH_LIMIT - len(f"/{RAY_LOCAL_TEMP_DIR_NAME}{RAY_LOCAL_SESSION_SOCKET_SUFFIX}")
)
LOCAL_RAY_TMPDIR_ENV_CANDIDATES = (
    ("SLURM_TMPDIR", "slurm_tmpdir"),
    ("TMPDIR", "tmpdir"),
    ("TMP", "tmp"),
    ("TEMP", "temp"),
)


def estimate_local_ray_socket_path(temp_dir: Path) -> str:
    """Return the longest local Ray socket path implied by a temp root."""

    return f"{temp_dir}/{RAY_LOCAL_TEMP_DIR_NAME}{RAY_LOCAL_SESSION_SOCKET_SUFFIX}"


def validate_local_ray_temp_dir(temp_dir: Path) -> Path:
    """Normalize a Ray temp root and reject paths that exceed socket limits."""

    expanded = temp_dir.expanduser()
    if not expanded.is_absolute():
        raise RuntimeError(
            "Ray temp dir candidates must be absolute after shell expansion: "
            f"{temp_dir}"
        )
    normalized = expanded.resolve()
    predicted_socket_path = estimate_local_ray_socket_path(normalized)
    if len(predicted_socket_path) > RAY_LOCAL_SOCKET_PATH_LIMIT:
        raise RuntimeError(
            "Resolved Ray temp dir is too long for Ray's local UNIX socket limit "
            f"({len(predicted_socket_path)} > {RAY_LOCAL_SOCKET_PATH_LIMIT}): {normalized}. "
            "Set NANODISCOVER_RAY_TMPDIR to a shorter absolute path. "
            f"The temp-root portion should stay at or below {RAY_LOCAL_TEMP_DIR_MAX_LENGTH} characters."
        )
    return normalized


def local_ray_temp_dir_name(run_dir: str | None) -> str:
    """Return a short namespaced directory name for one Ray runtime."""

    hash_source = str(Path(run_dir).expanduser().resolve()) if run_dir else os.getcwd()
    run_hash = hashlib.sha1(hash_source.encode("utf-8")).hexdigest()[:6]
    slurm_job_id = (os.environ.get("SLURM_JOB_ID") or "").strip()
    if slurm_job_id:
        return f"ndray-j{slurm_job_id}-r{run_hash}"
    return f"ndray-p{os.getpid()}-r{run_hash}"


def default_local_ray_temp_dir(run_dir: str) -> Path:
    """Return the namespaced run-dir-adjacent Ray temp-root fallback."""

    run_path = Path(run_dir).expanduser().resolve()
    return run_path.parent / local_ray_temp_dir_name(str(run_path))


def prepare_local_ray_temp_dir(temp_dir: Path) -> Path:
    """Create and probe a candidate Ray temp-root parent for local runtime use."""

    normalized = validate_local_ray_temp_dir(temp_dir)
    normalized.mkdir(parents=True, exist_ok=True)
    probe_path = normalized / f".nanodiscover-write-probe-{os.getpid()}"
    try:
        with probe_path.open("w", encoding="utf-8"):
            pass
    finally:
        probe_path.unlink(missing_ok=True)
    return normalized


def automatic_local_ray_temp_dir_candidates(config: GeneratorConfig) -> list[tuple[Path, str]]:
    """Return automatic Ray temp-dir candidates ordered by locality preference."""

    temp_name = local_ray_temp_dir_name(config.run_dir)
    candidates: list[tuple[Path, str]] = []
    for env_name, source in LOCAL_RAY_TMPDIR_ENV_CANDIDATES:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        candidate_root = raw_value.strip()
        if not candidate_root:
            continue
        candidates.append((Path(candidate_root).expanduser() / temp_name, source))

    user_name = (os.environ.get("USER") or "nanodiscover").strip() or "nanodiscover"
    candidates.append((Path("/tmp") / user_name / temp_name, "tmp_user"))
    if config.run_dir:
        candidates.append((default_local_ray_temp_dir(config.run_dir), "run_dir_parent"))
    return candidates


def resolve_local_ray_temp_dir(config: GeneratorConfig) -> tuple[Path, str]:
    """Resolve a usable local Ray temp root for generator-owned runtimes."""

    if config.ray_temp_dir:
        try:
            return prepare_local_ray_temp_dir(Path(config.ray_temp_dir).expanduser()), "env"
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "Configured NANODISCOVER_RAY_TMPDIR is not usable for a local Ray runtime: "
                f"{config.ray_temp_dir}. {exc}"
            ) from exc

    failure_details: list[str] = []
    for candidate_path, source in automatic_local_ray_temp_dir_candidates(config):
        try:
            return prepare_local_ray_temp_dir(candidate_path), source
        except (OSError, RuntimeError) as exc:
            failure_details.append(f"{source}={candidate_path}: {exc}")
            logger.warning(
                "generator_ray_runtime temp_dir_candidate_skipped source=%s temp_dir=%s reason=%s",
                source,
                str(candidate_path),
                str(exc),
            )

    detail_suffix = "; ".join(failure_details) if failure_details else "no automatic candidates were available"
    raise RuntimeError(
        "Could not resolve a usable local Ray temp dir. "
        "Set NANODISCOVER_RAY_TMPDIR to a short writable path on node-local storage. "
        f"Tried: {detail_suffix}"
    )


def extract_ray_token_logprobs(raw_logprobs: Any, token_ids: list[int]) -> list[float] | None:
    """Extract per-token logprobs from Ray Data LLM response payloads."""

    if not isinstance(raw_logprobs, list) or len(raw_logprobs) != len(token_ids):
        return None
    extracted: list[float] = []
    for token_id, token_entry in zip(token_ids, raw_logprobs, strict=True):
        if not isinstance(token_entry, dict):
            return None
        details = token_entry.get(token_id)
        if details is None:
            details = token_entry.get(str(token_id))
        if not isinstance(details, dict) or not isinstance(details.get("logprob"), (float, int)):
            return None
        extracted.append(float(details["logprob"]))
    return extracted


def postprocess_ray_llm_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Ray Data LLM output row into NanoDiscover's shape."""

    raw_logprobs = row.pop("logprobs", None)
    generated_tokens = [int(token_id) for token_id in list(row.get("generated_tokens") or [])]
    completion_logprobs = extract_ray_token_logprobs(raw_logprobs, generated_tokens)
    for key in ("prompt", "tokenized_prompt", "sampling_params", "model"):
        row.pop(key, None)
    row["row_index"] = int(row.get("row_index", 0))
    row["generated_tokens"] = generated_tokens
    row["generated_text"] = str(row["generated_text"]) if row.get("generated_text") is not None else ""
    row["finish_reason"] = str(row["finish_reason"]) if row.get("finish_reason") is not None else None
    row["completion_logprobs"] = completion_logprobs
    row["logprobs"] = None
    return row


@dataclass
class GeneratorConfig:
    """Configuration for the Ray Data LLM generation stage."""

    model_name_or_path: str
    tokenizer_name_or_path: str | None
    renderer_name: str
    renderer_system_prompt: str
    renderer_stop_sequence: str
    temperature: float
    phase1_max_tokens: int
    context_window: int
    context_buffer: int
    gpu_memory_utilization: float | None
    max_num_batched_tokens: int | None
    max_num_seqs: int | None
    request_parallelism: int | None
    request_timeout_s: float | None
    backend_name: str
    data_parallel_size: int
    tensor_parallel_size: int
    final_answer_marker: str | None
    forced_final_suffix: str | None
    phase1_end_marker: str | None
    forced_final_suffix_after_phase1_end_marker: str | None
    batch_size: int | None = None
    lora_rank: int | None = None
    ray_temp_dir: str | None = None
    run_dir: str | None = None


@dataclass
class PhaseOutput:
    """One generation phase output before final assembly."""

    text: str
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str | None = None


@dataclass
class GenerationOutput:
    """Final generation payload used by downstream evaluation."""

    prompt_text: str
    response_text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    completion_mask: list[float]
    finish_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "completion_token_ids": list(self.completion_token_ids),
            "completion_logprobs": list(self.completion_logprobs),
            "completion_mask": list(self.completion_mask),
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationOutput":
        return cls(
            prompt_text=str(payload.get("prompt_text", "")),
            response_text=str(payload.get("response_text", "")),
            prompt_token_ids=[int(value) for value in payload.get("prompt_token_ids", [])],
            completion_token_ids=[int(value) for value in payload.get("completion_token_ids", [])],
            completion_logprobs=[float(value) for value in payload.get("completion_logprobs", [])],
            completion_mask=[float(value) for value in payload.get("completion_mask", [])],
            finish_reason=(str(payload["finish_reason"]) if payload.get("finish_reason") is not None else None),
        )


def contains_token_subsequence(haystack: list[int], needle: list[int]) -> bool:
    """Check if needle token IDs appear as a contiguous subsequence in haystack.

    Uses token-level matching rather than text matching so that special tokens
    (e.g. GPT-OSS Harmony markers) are handled correctly regardless of how the
    tokenizer decodes them to text.
    """
    if not needle or len(needle) > len(haystack):
        return False
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return True
    return False


class TokenizerBackend:
    """Shared tokenizer and prompt-rendering utilities for generator backends."""

    def __init__(self, config: GeneratorConfig) -> None:
        from transformers import AutoTokenizer

        self.config = config
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path or config.model_name_or_path,
            trust_remote_code=True,
        )
        self.renderer = resolve_renderer(
            config.renderer_name,
            system_prompt=config.renderer_system_prompt,
            stop_sequence=config.renderer_stop_sequence,
        )

    def render_prompt(self, prompt: str) -> str:
        return self.renderer.render_prompt(prompt)

    def stop_sequences(self) -> list[str]:
        return list(self.renderer.stop_sequences)

    def tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return str(
            self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )


class RayDataLLMBackend(TokenizerBackend):
    """Public Ray Data LLM backend used by the release path."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__(config)
        self.adapter_path: str | None = None
        self._logged_config = False
        self._owns_ray_runtime = False
        self._ray_temp_dir: Path | None = None
        self._ray_temp_dir_source: str | None = None
        self._temporary_ray_env: dict[str, str | None] = {}

    def set_temporary_ray_env(self, name: str, value: str) -> None:
        """Temporarily override one Ray env var for this backend lifetime."""

        temporary_ray_env = getattr(self, "_temporary_ray_env", None)
        if temporary_ray_env is None:
            temporary_ray_env = {}
            self._temporary_ray_env = temporary_ray_env
        if name not in temporary_ray_env:
            temporary_ray_env[name] = os.environ.get(name)
        os.environ[name] = value

    def pin_local_ray_reinit_address(self) -> None:
        """Keep nested Ray init calls from probing stale default temp dirs."""

        self.set_temporary_ray_env("RAY_ADDRESS", "local")

    def pin_local_ray_temp_dir(self, temp_dir: Path) -> None:
        """Point Ray's internal init path at the selected local temp root."""

        self.set_temporary_ray_env("RAY_TMPDIR", str(temp_dir))

    def restore_temporary_ray_env(self) -> None:
        """Restore Ray env vars overridden by this backend."""

        temporary_ray_env = getattr(self, "_temporary_ray_env", None)
        if not temporary_ray_env:
            return None
        for name, previous_value in temporary_ray_env.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value
        temporary_ray_env.clear()
        return None

    def ensure_ray_runtime(self):
        """Prepare env so Ray Data LLM owns the single Ray init."""

        import ray

        self.pin_local_ray_reinit_address()
        if ray.is_initialized():
            return ray

        if self._ray_temp_dir is None:
            ray_temp_dir, ray_temp_dir_source = resolve_local_ray_temp_dir(self.config)
            self._ray_temp_dir = ray_temp_dir
            self._ray_temp_dir_source = ray_temp_dir_source
        assert self._ray_temp_dir is not None
        assert self._ray_temp_dir_source is not None
        self.pin_local_ray_temp_dir(self._ray_temp_dir)
        logger.info(
            "generator_ray_runtime prepare ownership=ray_data_llm address=local temp_root=%s temp_dir=%s temp_dir_source=%s",
            str(self._ray_temp_dir),
            str(self._ray_temp_dir / RAY_LOCAL_TEMP_DIR_NAME),
            self._ray_temp_dir_source,
        )
        return ray

    @staticmethod
    def extract_token_logprobs(row: dict[str, Any], token_ids: list[int]) -> list[float] | None:
        """Extract completion logprobs from either verbose or compact row shapes."""

        extracted = extract_ray_token_logprobs(row.get("logprobs"), token_ids)
        if extracted is not None:
            return extracted
        compact = row.get("completion_logprobs")
        if isinstance(compact, list) and len(compact) == len(token_ids):
            try:
                return [float(value) for value in compact]
            except (TypeError, ValueError):
                return None
        return None

    def log_config_interpretation(self) -> None:
        """Log the effective Ray Data LLM configuration once per backend."""

        if self._logged_config:
            return
        self._logged_config = True
        logger.info(
            "generator_ray_config data_parallel_size=%d tensor_parallel_size=%d batch_size=%d context_window=%d prompt_format=pretokenized chat_template_stage=off tokenize_stage=off detokenize_stage=off",
            max(1, int(self.config.data_parallel_size)),
            max(1, int(self.config.tensor_parallel_size)),
            max(1, int(self.config.batch_size or 64)),
            int(self.config.context_window),
        )

    def build_processor(self, *, use_lora: bool):
        """Build the Ray Data LLM processor for this sampling request."""

        ray = self.ensure_ray_runtime()
        ray_was_initialized = ray.is_initialized()
        from ray.data.llm import build_processor, vLLMEngineProcessorConfig

        engine_kwargs: dict[str, Any] = {
            "tensor_parallel_size": max(1, int(self.config.tensor_parallel_size)),
            "max_model_len": int(self.config.context_window),
            "trust_remote_code": True,
            "enable_prefix_caching": True,
        }
        if use_lora:
            engine_kwargs["enable_lora"] = True
            if self.config.lora_rank is not None:
                engine_kwargs["max_lora_rank"] = int(self.config.lora_rank)

        # https://docs.ray.io/en/latest/data/api/doc/ray.data.llm.vLLMEngineProcessorConfig.html#ray.data.llm.vLLMEngineProcessorConfig
        processor_config = vLLMEngineProcessorConfig(
            model_source=self.config.model_name_or_path,
            concurrency=max(1, int(self.config.data_parallel_size)),
            batch_size=max(1, int(self.config.batch_size)),
            engine_kwargs=engine_kwargs,
            chat_template_stage=False,
            tokenize_stage=False,
            detokenize_stage=False,
            # Leave accelerator_type unset so the launcher does not hard-code one GPU family.
        )
        try:
            processor = build_processor(processor_config, postprocess=postprocess_ray_llm_row)
        except Exception:
            if not ray_was_initialized and ray.is_initialized():
                self._owns_ray_runtime = True
            if not ray.is_initialized():
                self.cleanup_ray_temp_dir()
                self._ray_temp_dir = None
                self._ray_temp_dir_source = None
            raise
        if not ray_was_initialized and ray.is_initialized():
            self._owns_ray_runtime = True
            logger.info("generator_ray_runtime started ownership=backend init=ray_data_llm")
        return processor

    def response_text_from_row(self, row: dict[str, Any], token_ids: list[int]) -> str:
        """Recover response text from a normalized Ray Data LLM row."""

        generated_text = row.get("generated_text")
        if isinstance(generated_text, str):
            return generated_text
        return self.decode(token_ids)

    def adapter_model_source(self) -> str | None:
        """Return the resolved LoRA adapter path for Ray request rows."""

        if not self.adapter_path:
            return None
        return str(Path(self.adapter_path).resolve())

    @staticmethod
    def build_request_row(
        *,
        index: int,
        prompt: str,
        prompt_token_ids: list[int],
        max_tokens: int,
        temperature: float,
        stop_sequences: list[str] | None,
        model_source: str | None,
    ) -> dict[str, Any]:
        """Build one Ray Data LLM request row in NanoDiscover's canonical shape."""

        sampling_params: dict[str, Any] = {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "logprobs": 1,
        }
        if stop_sequences:
            sampling_params["stop"] = list(stop_sequences)
        row: dict[str, Any] = {
            "row_index": index,
            "prompt": prompt,
            "tokenized_prompt": list(prompt_token_ids),
            "sampling_params": sampling_params,
        }
        if model_source:
            row["model"] = model_source
        return row

    def sample(
        self,
        prompts: list[str],
        *,
        max_tokens: int | list[int],
        temperature: float,
        stop_sequences: list[str] | None = None,
        prompt_token_id_lists: list[list[int]] | None = None,
    ) -> list[PhaseOutput]:
        """Sample completions.

        Args:
            prompts: String prompts (used when prompt_token_id_lists is None).
            max_tokens: Per-prompt token budget.
            temperature: Sampling temperature.
            stop_sequences: Stop strings.
            prompt_token_id_lists: If provided, send these pre-tokenized IDs
                instead of tokenizing the string prompts.  This avoids
                re-tokenization and preserves exact token boundaries (important
                for GPT-OSS special tokens in phase 2).
        """
        ray = self.ensure_ray_runtime()

        use_token_ids = prompt_token_id_lists is not None
        count = len(prompt_token_id_lists) if use_token_ids else len(prompts)
        max_tokens_list = max_tokens if isinstance(max_tokens, list) else [int(max_tokens)] * count
        if len(max_tokens_list) != count:
            raise ValueError("prompts/prompt_token_id_lists and max_tokens must be the same length")
        if count == 0:
            return []

        self.log_config_interpretation()
        prompt_rows = (
            [(self.decode(token_ids), list(token_ids)) for token_ids in prompt_token_id_lists]
            if use_token_ids
            else [(prompt, self.tokenize(prompt)) for prompt in prompts]
        )
        model_source = self.adapter_model_source()
        rows = [
            self.build_request_row(
                index=index,
                prompt=prompt_text,
                prompt_token_ids=prompt_token_ids,
                max_tokens=prompt_max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                model_source=model_source,
            )
            for index, ((prompt_text, prompt_token_ids), prompt_max_tokens) in enumerate(
                zip(prompt_rows, max_tokens_list, strict=True)
            )
        ]

        started_at = time.perf_counter()
        processor = self.build_processor(use_lora=bool(self.adapter_path))
        dataset = ray.data.from_items(rows)
        # Retry logic for vLLM EADDRINUSE race condition.
        # When multiple vLLM engines start concurrently on the same node,
        # get_open_port() in vllm/utils/network_utils.py can hand the same
        # TCP port to two engines (TOCTOU race on socket bind).  This causes
        # torch.distributed.DistNetworkError: EADDRINUSE inside the
        # EngineCore subprocess, which kills the Ray actor and propagates as
        # an exception from take_all().  The race is probabilistic and almost
        # never happens twice in a row.
        # This is a well-known upstream vLLM issue, see #14919, #21638, #28498.
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                generated_rows = processor(dataset).take_all()
                break
            except Exception:
                if attempt == max_retries:
                    raise
                logger.warning(
                    "generator_retry attempt=%d/%d — rebuilding processor after likely vLLM port collision",
                    attempt,
                    max_retries,
                )
                processor = self.build_processor(use_lora=bool(self.adapter_path))
                dataset = ray.data.from_items(rows)
        generated_rows.sort(key=lambda row: int(row["row_index"]))
        generated_token_ids = [list(row.get("generated_tokens") or []) for row in generated_rows]
        generated_logprobs: list[list[float]] = []
        for row, gen_token_ids in zip(generated_rows, generated_token_ids, strict=True):
            extracted = row.get("completion_logprobs")
            if not isinstance(extracted, list) or len(extracted) != len(gen_token_ids):
                extracted = self.extract_token_logprobs(row, gen_token_ids)
            if extracted is None:
                raise RuntimeError(
                    "Ray Data LLM response did not include per-token logprobs; "
                    "expected Ray >= 2.54 behavior"
                )
            generated_logprobs.append([float(value) for value in extracted])
        logger.info(
            "generator_progress ray_data_llm_complete total=%d elapsed_s=%.2f data_parallel_size=%d tensor_parallel_size=%d",
            len(generated_rows),
            time.perf_counter() - started_at,
            max(1, int(self.config.data_parallel_size)),
            max(1, int(self.config.tensor_parallel_size)),
        )
        return [
            PhaseOutput(
                text=self.response_text_from_row(row, gen_token_ids),
                token_ids=[int(token_id) for token_id in gen_token_ids],
                logprobs=list(logprobs),
                finish_reason=(str(row["finish_reason"]) if row.get("finish_reason") is not None else None),
            )
            for row, gen_token_ids, logprobs in zip(generated_rows, generated_token_ids, generated_logprobs, strict=True)
        ]

    def reload_adapter(self, adapter_path: str | None) -> None:
        self.adapter_path = adapter_path

    def cleanup_ray_temp_dir(self) -> None:
        """Remove auto-selected Ray temp roots after the backend is done with them."""

        if self._ray_temp_dir is None or self._ray_temp_dir_source in {None, "env"}:
            return None

        import shutil

        try:
            shutil.rmtree(self._ray_temp_dir)
            logger.info(
                "generator_ray_runtime temp_dir_cleanup temp_dir=%s temp_dir_source=%s",
                str(self._ray_temp_dir),
                self._ray_temp_dir_source,
            )
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception(
                "generator_ray_runtime temp_dir_cleanup_failed temp_dir=%s temp_dir_source=%s",
                str(self._ray_temp_dir),
                self._ray_temp_dir_source,
            )
        return None

    def teardown(self) -> None:
        if not self._owns_ray_runtime:
            self.cleanup_ray_temp_dir()
            self._ray_temp_dir = None
            self._ray_temp_dir_source = None
            self.restore_temporary_ray_env()
            return None

        import ray

        try:
            if ray.is_initialized():
                logger.info("generator_ray_runtime shutdown ownership=backend")
                ray.shutdown()
        finally:
            self.cleanup_ray_temp_dir()
            self._ray_temp_dir = None
            self._ray_temp_dir_source = None
            self._owns_ray_runtime = False
            self.restore_temporary_ray_env()
        return None


class Generator:
    """High-level generation facade that manages backend lifecycle and staging."""

    def __init__(self, config: GeneratorConfig, backend: Any | None = None) -> None:
        self.config = config
        self.backend = backend
        self.adapter_path: str | None = None

    def reload_adapter(self, adapter_path: str | None) -> None:
        """Reload the currently active LoRA adapter for subsequent sampling."""

        self.adapter_path = adapter_path
        if self.backend is None:
            self.backend = self.ensure_backend()
        self.backend.reload_adapter(adapter_path)

    def ensure_backend(self) -> Any:
        """Instantiate the configured generation backend on first use."""

        if self.backend is None:
            if (self.config.backend_name or "ray_data_llm").strip().lower() != "ray_data_llm":
                raise RuntimeError(
                    "Unsupported generator backend: "
                    f"{self.config.backend_name!r}. NanoDiscover only supports ray_data_llm."
                )
            self.backend = RayDataLLMBackend(self.config)
            self.backend.reload_adapter(self.adapter_path)
        return self.backend

    def phase1_budget(self, prompt_token_ids: list[int]) -> int:
        """Return the phase-1 token budget remaining after the prompt."""

        budget = int(self.config.phase1_max_tokens) - len(prompt_token_ids)
        if budget <= 0:
            raise ValueError(
                f"Prompt length {len(prompt_token_ids)} exceeds phase1_max_tokens {self.config.phase1_max_tokens}."
            )
        return budget

    def generate_with_backend(self, prompts: list[str], backend: Any) -> list[GenerationOutput]:
        """Run phase-1/phase-2 generation against an already prepared backend."""

        started_at = time.perf_counter()
        stop_sequences = list(backend.stop_sequences())
        stop_token_sequences = [backend.tokenize(stop_sequence) for stop_sequence in stop_sequences]
        phase1_end_marker_token_ids = backend.tokenize(self.config.phase1_end_marker) if self.config.phase1_end_marker else []
        final_answer_marker_token_ids = backend.tokenize(self.config.final_answer_marker) if self.config.final_answer_marker else []
        rendered_prompts = [backend.render_prompt(prompt) for prompt in prompts]
        prompt_token_ids_by_prompt = [backend.tokenize(rendered_prompt) for rendered_prompt in rendered_prompts]
        phase1_budgets = [self.phase1_budget(prompt_token_ids) for prompt_token_ids in prompt_token_ids_by_prompt]
        logger.info(
            "generator_progress start prompts=%d data_parallel_size=%d tensor_parallel_size=%d phase1_tokens_mean=%.1f phase1_tokens_max=%d mode=ray_data_llm",
            len(prompts),
            int(self.config.data_parallel_size),
            int(self.config.tensor_parallel_size),
            (sum(phase1_budgets) / len(phase1_budgets)) if phase1_budgets else 0.0,
            max(phase1_budgets, default=0),
        )
        phase1_started_at = time.perf_counter()
        phase1_outputs = backend.sample(
            rendered_prompts,
            max_tokens=phase1_budgets,
            temperature=self.config.temperature,
            stop_sequences=stop_sequences,
        )
        if len(phase1_outputs) != len(prompts):
            raise RuntimeError("Generator backend returned the wrong number of phase1 outputs")

        outputs: list[GenerationOutput | None] = [None] * len(prompts)
        phase2_requests: list[dict[str, Any]] = []
        phase1_completed_count = 0
        forced_suffix_count = 0
        budget_exhausted_count = 0
        for index, (prompt, rendered_prompt, prompt_token_ids, phase1) in enumerate(
            zip(prompts, rendered_prompts, prompt_token_ids_by_prompt, phase1_outputs, strict=True)
        ):
            phase1_budget = self.phase1_budget(prompt_token_ids)
            phase1_completed = len(phase1.token_ids) < phase1_budget or any(
                len(stop_ids) and len(stop_ids) <= len(phase1.token_ids) and phase1.token_ids[-len(stop_ids) :] == stop_ids
                for stop_ids in stop_token_sequences
            )
            finish_reason = (phase1.finish_reason or "").lower()
            if phase1_completed or (finish_reason and finish_reason not in {"length", "max_tokens"}):
                phase1_completed_count += 1
                outputs[index] = GenerationOutput(
                    prompt_text=prompt,
                    response_text=phase1.text,
                    prompt_token_ids=list(prompt_token_ids),
                    completion_token_ids=list(phase1.token_ids),
                    completion_logprobs=list(phase1.logprobs),
                    completion_mask=[1.0] * len(phase1.token_ids),
                    finish_reason=phase1.finish_reason,
                )
                continue

            forced_suffix = ""
            forced_token_ids: list[int] = []
            # Parity: original TTT-Discover checks for the final-answer marker
            # using token subsequence matching, not text
            # matching.  Token-level is robust to special tokens that may not
            # round-trip through decode (e.g. GPT-OSS Harmony markers).
            marker_already_present = bool(final_answer_marker_token_ids) and contains_token_subsequence(phase1.token_ids, final_answer_marker_token_ids)
            if self.config.forced_final_suffix and final_answer_marker_token_ids and not marker_already_present:
                forced_suffix = self.config.forced_final_suffix
                if phase1_end_marker_token_ids and len(phase1_end_marker_token_ids) <= len(phase1.token_ids) and phase1.token_ids[-len(phase1_end_marker_token_ids) :] == phase1_end_marker_token_ids:
                    forced_suffix = self.config.forced_final_suffix_after_phase1_end_marker or forced_suffix
                forced_token_ids = backend.tokenize(forced_suffix)
            if forced_token_ids:
                forced_suffix_count += 1

            phase2_budget = max(
                0,
                int(self.config.context_window) - len(prompt_token_ids) - len(phase1.token_ids) - len(forced_token_ids) - int(self.config.context_buffer),
            )
            if phase2_budget <= 0:
                budget_exhausted_count += 1
                outputs[index] = GenerationOutput(
                    prompt_text=prompt,
                    response_text=phase1.text + forced_suffix,
                    prompt_token_ids=list(prompt_token_ids),
                    completion_token_ids=list(phase1.token_ids) + forced_token_ids,
                    completion_logprobs=list(phase1.logprobs) + ([0.0] * len(forced_token_ids)),
                    completion_mask=([1.0] * len(phase1.token_ids)) + ([0.0] * len(forced_token_ids)),
                    finish_reason="budget_exhausted",
                )
                continue

            phase2_requests.append(
                {
                    "output_index": index,
                    "prompt_text": prompt,
                    "prompt_token_ids": list(prompt_token_ids),
                    "phase1": phase1,
                    "forced_suffix": forced_suffix,
                    "forced_token_ids": forced_token_ids,
                    "max_tokens": phase2_budget,
                }
            )

        logger.info(
            "generator_progress phase1_complete total=%d phase1_completed=%d phase2_needed=%d forced_suffixes=%d budget_exhausted=%d elapsed_s=%.2f",
            len(prompts),
            phase1_completed_count,
            len(phase2_requests),
            forced_suffix_count,
            budget_exhausted_count,
            time.perf_counter() - phase1_started_at,
        )
        if phase2_requests:
            logger.info(
                "generator_progress phase2_start total=%d phase2_tokens_mean=%.1f phase2_tokens_max=%d",
                len(phase2_requests),
                (sum(int(request["max_tokens"]) for request in phase2_requests) / len(phase2_requests)),
                max(int(request["max_tokens"]) for request in phase2_requests),
            )
            phase2_started_at = time.perf_counter()
            # Parity: original TTT-Discover builds the phase 2 prompt from
            # pre-tokenized chunks (prompt_tokens + phase1_tokens +
            # prefill_tokens) to avoid re-tokenization.  We do the same by
            # sending concatenated token ID lists via prompt_token_id_lists
            # so that special-token boundaries (especially GPT-OSS Harmony
            # markers) are preserved exactly.
            phase2_prompt_token_ids = [
                request["prompt_token_ids"] + list(request["phase1"].token_ids) + request["forced_token_ids"]
                for request in phase2_requests
            ]
            phase2_outputs = backend.sample(
                [],
                max_tokens=[request["max_tokens"] for request in phase2_requests],
                temperature=self.config.temperature,
                stop_sequences=stop_sequences,
                prompt_token_id_lists=phase2_prompt_token_ids,
            )
            if len(phase2_outputs) != len(phase2_requests):
                raise RuntimeError("Generator backend returned the wrong number of phase2 outputs")
            for request, phase2 in zip(phase2_requests, phase2_outputs, strict=True):
                outputs[request["output_index"]] = GenerationOutput(
                    prompt_text=request["prompt_text"],
                    response_text=request["phase1"].text + request["forced_suffix"] + phase2.text,
                    prompt_token_ids=list(request["prompt_token_ids"]),
                    completion_token_ids=list(request["phase1"].token_ids) + request["forced_token_ids"] + list(phase2.token_ids),
                    completion_logprobs=list(request["phase1"].logprobs) + ([0.0] * len(request["forced_token_ids"])) + list(phase2.logprobs),
                    completion_mask=([1.0] * len(request["phase1"].token_ids)) + ([0.0] * len(request["forced_token_ids"])) + ([1.0] * len(phase2.token_ids)),
                    finish_reason=phase2.finish_reason,
                )
            logger.info(
                "generator_progress phase2_complete total=%d elapsed_s=%.2f",
                len(phase2_requests),
                time.perf_counter() - phase2_started_at,
            )
        total_completion_tokens = sum(len(output.completion_token_ids) for output in outputs if output is not None)
        logger.info(
            "generator_progress completed=%d total=%d pct=100.0 completion_tokens=%d elapsed_s=%.2f",
            len(prompts),
            len(prompts),
            total_completion_tokens,
            time.perf_counter() - started_at,
        )
        return [output for output in outputs if output is not None]

    def generate(self, prompts: list[str]) -> list[GenerationOutput]:
        """Generate rollout candidates for the provided prompts."""

        return self.generate_with_backend(prompts, self.ensure_backend())

    def teardown(self) -> None:
        """Tear down the active backend and release owned runtime state."""

        if self.backend is not None:
            self.backend.teardown()
            self.backend = None